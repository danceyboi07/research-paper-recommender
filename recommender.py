"""Research-paper recommender backed by PostgreSQL + pgvector."""
from __future__ import annotations

import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def as_numpy(vector):
    """Convert NumPy/list/pgvector Vector values to a float32 NumPy array."""
    if isinstance(vector, np.ndarray):
        return vector.astype(np.float32, copy=False)

    # pgvector-python returns a Vector object for vector columns.
    # The supported conversions are to_numpy() and to_list().
    to_numpy = getattr(vector, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy(), dtype=np.float32)

    to_list = getattr(vector, "to_list", None)
    if callable(to_list):
        return np.asarray(to_list(), dtype=np.float32)

    return np.asarray(vector, dtype=np.float32)


def unit(vector):
    vector = as_numpy(vector)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


class HashEncoder:
    """Download-free lexical baseline for testing, NOT semantic embeddings."""
    identity = "hash-word-unigram-bigram-512-v1"
    dimension = 512

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.model = HashingVectorizer(
            n_features=self.dimension,
            alternate_sign=False,
            stop_words="english",
            ngram_range=(1, 2),
            norm="l2",
        )

    def encode(self, texts):
        return self.model.transform(texts).toarray().astype(np.float32)


class SemanticEncoder:
    """Sentence-Transformer encoder. Long inputs are chunked before averaging."""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        dimension_method = getattr(self.model, "get_embedding_dimension", None)
        self.dimension = (
            dimension_method()
            if dimension_method
            else self.model.get_sentence_embedding_dimension()
        )
        self.identity = "sentence-transformer:" + model_name + ":mean-chunks-v1"

    def encode(self, texts):
        tokenizer = self.model.tokenizer
        limit = self.model.max_seq_length - tokenizer.num_special_tokens_to_add(False)

        chunks, spans = [], []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            start = len(chunks)
            chunks.extend(
                tokenizer.decode(ids[i : i + limit])
                for i in range(0, len(ids), limit)
            )
            if start == len(chunks):
                chunks.append("")
            spans.append((start, len(chunks)))

        vectors = self.model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.array(
            [unit(vectors[a:b].mean(axis=0)) for a, b in spans],
            dtype=np.float32,
        )


class ArxivClient:
    """Fetch paper metadata from the arXiv Atom API with pacing and retries."""
    endpoint = "https://export.arxiv.org/api/query"

    def __init__(self):
        import requests

        self.requests = requests
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "ResearchPaperRecommender/0.2 (educational prototype)"
        )
        self.last_request = 0.0

    @staticmethod
    def parse(xml):
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml)
        if root.tag != "{http://www.w3.org/2005/Atom}feed":
            raise ValueError("arXiv returned an unexpected response.")

        papers = []
        for entry in root.findall("a:entry", ns):
            def field(name):
                return " ".join(
                    entry.findtext("a:" + name, default="", namespaces=ns).split()
                )

            url = field("id")
            if "/api/errors" in url:
                raise ValueError("arXiv rejected the query: " + field("summary"))

            identifier = re.sub(r"v\d+$", "", url.split("/abs/")[-1])
            if not identifier or not field("title") or not field("summary"):
                continue

            papers.append(
                {
                    "id": "arxiv:" + identifier,
                    "title": field("title"),
                    "abstract": field("summary"),
                    "published": field("published")[:10],
                    "url": "https://arxiv.org/abs/" + identifier,
                    "authors": [
                        x.text or "" for x in entry.findall("a:author/a:name", ns)
                    ],
                    "topics": [
                        x.attrib["term"] for x in entry.findall("a:category", ns)
                    ],
                    "source": "arxiv",
                }
            )
        return papers

    def fetch(self, query, max_results=100):
        if not query.strip() or not 1 <= max_results <= 1000:
            raise ValueError("Supply an arXiv query and max_results between 1 and 1000.")

        papers = {}
        for start in range(0, max_results, 100):
            count = min(100, max_results - start)
            batch = []
            for attempt in range(3):
                time.sleep(max(0, 3.1 - (time.monotonic() - self.last_request)))
                self.last_request = time.monotonic()
                try:
                    response = self.session.get(
                        self.endpoint,
                        params={
                            "search_query": query,
                            "start": start,
                            "max_results": count,
                            "sortBy": "relevance",
                            "sortOrder": "descending",
                        },
                        timeout=(10, 45),
                    )
                    response.raise_for_status()
                    batch = self.parse(response.content)
                    break
                except (self.requests.RequestException, ET.ParseError) as exc:
                    if attempt == 2:
                        raise RuntimeError(
                            "arXiv unavailable. Retry later or use the offline demo."
                        ) from exc
                    time.sleep(2**attempt)

            for paper in batch:
                papers[paper["id"]] = paper
            if len(batch) < count:
                break

        return list(papers.values())


class PaperRecommender:
    """PostgreSQL + pgvector storage and personalized recommendation logic."""

    def __init__(self, database_url, encoder):
        self.encoder = encoder
        self.database_url = database_url
        self._lock = threading.RLock()

        self.db = psycopg.connect(
            database_url,
            autocommit=True,
            row_factory=dict_row,
        )

        # The vector extension must exist before pgvector can register its type.
        with self.db.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.db)

        self._create_schema()
        self._check_encoder()

    def _create_schema(self):
        dim = int(self.encoder.dimension)
        statements = [
            """
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS papers (
                id BIGSERIAL PRIMARY KEY,
                paper_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                published DATE,
                url TEXT,
                authors JSONB NOT NULL DEFAULT '[]'::jsonb,
                topics JSONB NOT NULL DEFAULT '[]'::jsonb,
                source TEXT NOT NULL DEFAULT 'custom',
                embedding vector({dim}) NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS profiles (
                name TEXT PRIMARY KEY,
                interests TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS feedback (
                profile TEXT REFERENCES profiles(name) ON DELETE CASCADE,
                paper_id TEXT REFERENCES papers(paper_id) ON DELETE CASCADE,
                rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                PRIMARY KEY (profile, paper_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS feedback_profile_idx
            ON feedback(profile)
            """,
            """
            CREATE INDEX IF NOT EXISTS papers_published_idx
            ON papers(published DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS papers_embedding_hnsw
            ON papers USING hnsw (embedding vector_cosine_ops)
            """,
        ]
        with self._lock, self.db.cursor() as cur:
            for statement in statements:
                cur.execute(statement)

    def _check_encoder(self):
        signature = json.dumps([self.encoder.identity, self.encoder.dimension])
        row = self.query_one(
            "SELECT value FROM config WHERE key='encoder'"
        )
        if row and row["value"] != signature:
            raise ValueError(
                "This PostgreSQL database was created with another encoder. "
                "Use a different database or reset the schema."
            )
        self.execute(
            """
            INSERT INTO config(key, value) VALUES ('encoder', %s)
            ON CONFLICT(key) DO NOTHING
            """,
            (signature,),
        )

    def execute(self, sql, params=None):
        with self._lock, self.db.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.rowcount

    def executemany(self, sql, rows):
        rows = list(rows)
        if not rows:
            return 0
        with self._lock, self.db.cursor() as cur:
            cur.executemany(sql, rows)
            return cur.rowcount

    def query_one(self, sql, params=None):
        with self._lock, self.db.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchone()

    def query_all(self, sql, params=None):
        with self._lock, self.db.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()

    @property
    def count(self):
        return self.query_one("SELECT COUNT(*) AS n FROM papers")["n"]

    @staticmethod
    def _paper_dict(row, include_embedding=False):
        paper = dict(row)
        if isinstance(paper.get("published"), date):
            paper["published"] = paper["published"].isoformat()
        if not include_embedding:
            paper.pop("embedding", None)
        paper.pop("id", None)
        return paper

    def add_papers(self, papers):
        unique = {}
        for p in papers:
            if any(
                not isinstance(p.get(k), str) or not p[k].strip()
                for k in ("id", "title", "abstract")
            ):
                raise ValueError(
                    "Each paper needs nonempty id, title, and abstract strings."
                )
            if p.get("published"):
                date.fromisoformat(p["published"])
            unique[p["id"]] = p

        if not unique:
            return 0

        batch = list(unique.values())
        vectors = self.encoder.encode(
            [p["title"] + ". " + p["abstract"] for p in batch]
        )

        if (
            vectors.shape != (len(batch), self.encoder.dimension)
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Encoder returned invalid vectors.")
        if np.any(np.linalg.norm(vectors, axis=1) < 1e-8):
            raise ValueError(
                "Some papers produced empty vectors. Supply more descriptive text."
            )

        rows = []
        for p, vector in zip(batch, vectors):
            published = date.fromisoformat(p["published"]) if p.get("published") else None
            rows.append(
                (
                    p["id"],
                    p["title"],
                    p["abstract"],
                    published,
                    p.get("url", ""),
                    Jsonb(p.get("authors", [])),
                    Jsonb(p.get("topics", [])),
                    p.get("source", "custom"),
                    unit(vector),
                )
            )

        sql = """
        INSERT INTO papers
            (paper_id, title, abstract, published, url, authors, topics, source, embedding)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT(paper_id) DO UPDATE SET
            title=EXCLUDED.title,
            abstract=EXCLUDED.abstract,
            published=EXCLUDED.published,
            url=EXCLUDED.url,
            authors=EXCLUDED.authors,
            topics=EXCLUDED.topics,
            source=EXCLUDED.source,
            embedding=EXCLUDED.embedding
        """
        self.executemany(sql, rows)
        return len(rows)

    def set_profile(self, name, interests):
        if not name.strip() or not interests.strip():
            raise ValueError("Profile name and interests cannot be blank.")
        self.execute(
            """
            INSERT INTO profiles(name, interests) VALUES (%s, %s)
            ON CONFLICT(name) DO UPDATE SET interests=EXCLUDED.interests
            """,
            (name, interests),
        )

    def rate(self, profile, paper_id, rating):
        """1 = save/like; -1 = dismiss; 0 = remove existing feedback."""
        if rating not in (-1, 0, 1):
            raise ValueError("rating must be -1, 0, or 1")
        self._interests(profile)

        if not self.query_one(
            "SELECT 1 AS ok FROM papers WHERE paper_id=%s",
            (paper_id,),
        ):
            raise ValueError("Unknown paper ID: " + paper_id)

        if rating == 0:
            self.execute(
                "DELETE FROM feedback WHERE profile=%s AND paper_id=%s",
                (profile, paper_id),
            )
        else:
            self.execute(
                """
                INSERT INTO feedback(profile, paper_id, rating)
                VALUES (%s,%s,%s)
                ON CONFLICT(profile,paper_id)
                DO UPDATE SET rating=EXCLUDED.rating
                """,
                (profile, paper_id, rating),
            )

    def _interests(self, profile):
        row = self.query_one(
            "SELECT interests FROM profiles WHERE name=%s",
            (profile,),
        )
        if not row:
            raise ValueError("Unknown profile: " + profile)
        return row["interests"]

    def profile_vectors(self, profile):
        seed = unit(self.encoder.encode([self._interests(profile)])[0])
        if np.linalg.norm(seed) < 1e-8:
            raise ValueError(
                "Interest text produced an empty vector. Add descriptive words."
            )

        rows = self.query_all(
            """
            SELECT f.paper_id, f.rating, p.embedding, p.title
            FROM feedback f
            JOIN papers p ON p.paper_id=f.paper_id
            WHERE f.profile=%s
            """,
            (profile,),
        )

        liked = [
            as_numpy(r["embedding"])
            for r in rows
            if r["rating"] == 1
        ]
        disliked = [
            as_numpy(r["embedding"])
            for r in rows
            if r["rating"] == -1
        ]

        learned = seed.copy()
        if liked:
            learned += 0.7 * np.mean(liked, axis=0)
        if disliked:
            learned -= 0.25 * np.mean(disliked, axis=0)

        return seed, unit(learned), rows

    def recommend(
        self,
        profile,
        top_k=10,
        exploration=0.2,
        candidate_pool=200,
        min_relevance=0.15,
    ):
        """Personalized ranking with relevance, recency, and diversity.

        Base = .30*original-interest cosine + .60*learned-profile cosine
               + .10*recency
        Greedy rank = base - exploration*redundancy
        """
        if (
            top_k < 1
            or candidate_pool < top_k
            or not 0 <= exploration <= 1
            or not 0 <= min_relevance <= 1
        ):
            raise ValueError(
                "Check top_k, candidate_pool, exploration and min_relevance."
            )

        seed, profile_vector, feedback = self.profile_vectors(profile)
        if not self.count:
            return []

        seen = {r["paper_id"] for r in feedback}
        k = min(self.count, candidate_pool + len(seen))

        neighbors = self.query_all(
            """
            SELECT *,
                   embedding <=> %s AS distance
            FROM papers
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (profile_vector, profile_vector, k),
        )

        liked = [
            (r["title"], as_numpy(r["embedding"]))
            for r in feedback
            if r["rating"] == 1
        ]

        candidates = []
        for row in neighbors:
            if row["paper_id"] in seen:
                continue

            vector = as_numpy(row["embedding"])
            interest = float(np.clip(vector @ seed, 0, 1))
            relevance = float(np.clip(vector @ profile_vector, 0, 1))
            if relevance < min_relevance:
                continue

            recency = 0.0
            if row["published"]:
                age = max(0, (date.today() - row["published"]).days)
                recency = 2 ** (-age / (365.25 * 5))

            paper = self._paper_dict(row)
            paper.update(
                {
                    "vector": vector,
                    "interest_score": interest,
                    "profile_score": relevance,
                    "recency_score": recency,
                    "base_score": (
                        0.3 * interest + 0.6 * relevance + 0.1 * recency
                    ),
                }
            )
            candidates.append(paper)

        selected = []
        while candidates and len(selected) < top_k:
            for paper in candidates:
                paper["redundancy"] = max(
                    [
                        max(0.0, float(paper["vector"] @ s["vector"]))
                        for s in selected
                    ]
                    or [0.0]
                )
                paper["rank_score"] = (
                    paper["base_score"]
                    - exploration * paper["redundancy"]
                )

            winner = max(
                candidates,
                key=lambda p: (p["rank_score"], p["paper_id"]),
            )
            candidates.remove(winner)

            why = (
                f"Interest similarity {winner['interest_score']:.3f}; "
                f"learned-profile similarity {winner['profile_score']:.3f}."
            )
            if liked:
                title, similarity = max(
                    [
                        (title, float(winner["vector"] @ vector))
                        for title, vector in liked
                    ],
                    key=lambda x: x[1],
                )
                why += (
                    f' Closest saved paper: "{title}" '
                    f"(cosine {similarity:.3f})."
                )
            why += (
                f" Recency contribution {0.1 * winner['recency_score']:.3f}; "
                f"diversity penalty "
                f"{exploration * winner['redundancy']:.3f}."
            )
            winner["why"] = why
            selected.append(winner)

        return [
            {k: v for k, v in p.items() if k != "vector"}
            for p in selected
        ]

    def papers_with_vectors(self, ids=None):
        if ids is not None:
            ids = list(ids)
            if not ids:
                return []
            rows = self.query_all(
                """
                SELECT * FROM papers
                WHERE paper_id = ANY(%s)
                ORDER BY paper_id
                """,
                (ids,),
            )
        else:
            rows = self.query_all("SELECT * FROM papers ORDER BY paper_id")
        return rows

    def paper_vector(self, paper_id):
        row = self.query_one(
            "SELECT embedding FROM papers WHERE paper_id=%s",
            (paper_id,),
        )
        if not row:
            raise ValueError("Unknown paper ID: " + paper_id)
        return as_numpy(row["embedding"])

    def nearest_ids(self, vector, k=60):
        rows = self.query_all(
            """
            SELECT paper_id
            FROM papers
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (unit(vector), int(k)),
        )
        return [r["paper_id"] for r in rows]

    def close(self):
        self.db.close()


def demo_papers():
    """Fictional fixtures for offline testing only."""
    topics = [
        ("Schizophrenia classification with functional MRI",
         "Machine learning schizophrenia diagnosis using functional MRI brain connectivity biomarkers."),
        ("Graph learning for brain connectivity",
         "Graph neural networks model functional MRI brain connectivity for neurological classification."),
        ("Multimodal schizophrenia prediction",
         "Multimodal machine learning combines clinical text and MRI for schizophrenia diagnosis."),
        ("Clinical language representation",
         "Language models encode clinical text and medical records for diagnosis and healthcare."),
        ("Self-supervised MRI representations",
         "Self-supervised learning creates brain MRI representations with limited diagnostic labels."),
        ("Bias in neurological prediction",
         "Evaluation of machine learning brain diagnosis across hospitals studies data leakage and generalization."),
        ("Robotic reinforcement learning",
         "Robot navigation and control with reinforcement learning and simulated environments."),
        ("Graph models of molecules",
         "Graph neural networks predict molecular properties for drug discovery."),
        ("Neural population dynamics",
         "Computational neuroscience models neural population activity and brain representations."),
        ("Satellite crop mapping",
         "Remote sensing satellite images identify agricultural crops and vegetation."),
        ("Violin pitch estimation",
         "Audio signal processing estimates violin pitch and musical intonation."),
        ("Multimodal clinical retrieval",
         "Clinical text and medical images support multimodal representation learning and patient retrieval."),
    ]
    return [
        {
            "id": f"demo:{i:02d}",
            "title": "[FICTIONAL DEMO] " + title,
            "abstract": abstract,
            "published": "2024-01-01",
            "url": "",
            "source": "fictional_test_fixture",
            "authors": [],
            "topics": [],
        }
        for i, (title, abstract) in enumerate(topics, 1)
    ]
