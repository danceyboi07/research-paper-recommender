"""Topic tabs, paper imports, and persistent collections for the FastAPI app."""
import base64
import hashlib
import io
import re
import time
import uuid
from urllib.parse import urlparse

from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from paper_map import PaperMap
from recommender import ArxivClient


def topic_query(text):
    """Small transparent keyword translator, not an LLM query planner."""
    terms = re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", text.lower())
    stop = set(ENGLISH_STOP_WORDS) | {
        "interested",
        "papers",
        "research",
        "topic",
        "models",
        "model",
        "using",
    }
    terms = list(dict.fromkeys(t for t in terms if t not in stop))[:8]
    if not terms:
        raise ValueError(
            "Enter a specific topic, for example AI for schizophrenia diagnosis."
        )

    synonyms = {
        "ai": '(all:"artificial intelligence" OR all:"machine learning" OR all:"deep learning")',
        "fmri": '(all:fmri OR all:"functional MRI")',
        "llm": '(all:llm OR all:"large language model")',
    }
    return " AND ".join(synonyms.get(t, f'all:"{t}"') for t in terms)


def arxiv_id(value):
    value = value.strip()
    if "://" in value:
        parsed = urlparse(value)
        if parsed.hostname not in (
            "arxiv.org",
            "www.arxiv.org",
            "export.arxiv.org",
        ):
            raise ValueError(
                "For automatic import, use an arXiv link. "
                "For other sources, paste title and abstract."
            )
        value = re.sub(r"^/(abs|pdf)/", "", parsed.path)

    value = re.sub(r"\.pdf$", "", value)
    value = re.sub(r"^arxiv:", "", value, flags=re.I)
    value = re.sub(r"v\d+$", "", value)

    if not re.fullmatch(
        r"(\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})",
        value,
    ):
        raise ValueError("Enter an arXiv abstract/PDF link or paper ID.")

    return value


class ResearchWorkspace:
    def __init__(self, app, profile="neuro"):
        self.app = app
        self.client = ArxivClient()
        self.default_profile = profile
        self.maps = {}

        schema = [
            """
            CREATE TABLE IF NOT EXISTS map_topics (
                profile TEXT PRIMARY KEY REFERENCES profiles(name) ON DELETE CASCADE,
                title TEXT NOT NULL,
                query TEXT NOT NULL DEFAULT '',
                anchor TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS map_members (
                profile TEXT REFERENCES map_topics(profile) ON DELETE CASCADE,
                paper_id TEXT REFERENCES papers(paper_id) ON DELETE CASCADE,
                PRIMARY KEY(profile, paper_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS paper_files (
                paper_id TEXT PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
                data BYTEA NOT NULL
            )
            """,
        ]
        for statement in schema:
            self.app.execute(statement)

        if not self.app.query_one(
            "SELECT 1 AS ok FROM map_topics WHERE profile=%s",
            (profile,),
        ):
            self.app.execute(
                """
                INSERT INTO map_topics(profile, title)
                VALUES (%s,%s)
                """,
                (profile, "AI + neuroscience"),
            )
            rows = self.app.query_all("SELECT paper_id FROM papers")
            self.app.executemany(
                """
                INSERT INTO map_members(profile, paper_id)
                VALUES (%s,%s)
                ON CONFLICT DO NOTHING
                """,
                [(profile, row["paper_id"]) for row in rows],
            )

    def tabs(self):
        return self.app.query_all(
            """
            SELECT profile, title, query, anchor
            FROM map_topics
            ORDER BY title, profile
            """
        )

    def current_map(self, profile=None):
        profile = profile or self.default_profile
        if profile not in self.maps:
            topic = self.app.query_one(
                "SELECT * FROM map_topics WHERE profile=%s",
                (profile,),
            )
            if not topic:
                raise ValueError("This topic tab no longer exists.")

            ids = [
                r["paper_id"]
                for r in self.app.query_all(
                    """
                    SELECT paper_id
                    FROM map_members
                    WHERE profile=%s
                    """,
                    (profile,),
                )
            ]
            self.maps[profile] = PaperMap(
                self.app,
                profile,
                ids=ids,
                anchor=topic["anchor"],
            )

        return self.maps[profile]

    def payload(self, profile=None):
        profile = profile or self.default_profile
        data = self.current_map(profile).payload()
        data["tabs"] = [dict(row) for row in self.tabs()]
        data["active"] = profile

        pdf_ids = {
            row["paper_id"]
            for row in self.app.query_all(
                "SELECT paper_id FROM paper_files"
            )
        }
        for paper in data["papers"]:
            paper["has_pdf"] = paper["paper_id"] in pdf_ids

        return data

    def html(self, profile=None):
        profile = profile or self.default_profile
        return self.current_map(profile).html(self.payload(profile))

    def new_topic(self, text):
        text = str(text).strip()
        if not 3 <= len(text) <= 250:
            raise ValueError("Enter a topic of 3–250 characters.")

        query = topic_query(text)
        papers = self.client.fetch(query, max_results=80)
        if not papers:
            raise ValueError(
                "No papers matched all the keywords. "
                'Try a shorter topic such as "AI schizophrenia".'
            )

        self.app.add_papers(papers)
        profile = "topic_" + uuid.uuid4().hex[:12]
        self.app.set_profile(profile, text)

        self.app.execute(
            """
            INSERT INTO map_topics(profile, title, query)
            VALUES (%s,%s,%s)
            """,
            (profile, text, query),
        )
        self.app.executemany(
            """
            INSERT INTO map_members(profile, paper_id)
            VALUES (%s,%s)
            ON CONFLICT DO NOTHING
            """,
            [(profile, p["id"]) for p in papers],
        )

        self.maps.pop(profile, None)
        self.current_map(profile)
        return profile

    def fetch_arxiv(self, value):
        identifier = arxiv_id(value)
        for attempt in range(3):
            time.sleep(
                max(
                    0,
                    3.1
                    - (
                        time.monotonic()
                        - self.client.last_request
                    ),
                )
            )
            self.client.last_request = time.monotonic()
            try:
                response = self.client.session.get(
                    self.client.endpoint,
                    params={"id_list": identifier},
                    timeout=(10, 40),
                )
                response.raise_for_status()
                papers = self.client.parse(response.content)
                if not papers:
                    raise ValueError("That arXiv paper was not found.")
                return papers[0]
            except ValueError:
                raise
            except Exception:
                if attempt == 2:
                    raise RuntimeError(
                        "arXiv import failed. Try again, "
                        "or paste the title and abstract."
                    )
                time.sleep(2**attempt)

    def import_paper(self, profile, data):
        mode = data.get("mode", "text")
        raw = None

        if mode == "arxiv":
            paper = self.fetch_arxiv(str(data.get("arxiv", "")))
        else:
            title = str(data.get("title", "")).strip()
            abstract = str(data.get("abstract", "")).strip()
            url = str(data.get("url", "")).strip()

            if url and (
                urlparse(url).scheme not in ("http", "https")
                or not urlparse(url).netloc
            ):
                raise ValueError(
                    "Paper URL must start with https:// or http://."
                )

            if mode == "pdf":
                encoded = data.get("pdf", "")
                if (
                    not isinstance(encoded, str)
                    or len(encoded) > 14000000
                ):
                    raise ValueError("Choose a PDF smaller than 10 MB.")

                try:
                    raw = base64.b64decode(
                        encoded,
                        validate=True,
                    )
                except Exception as exc:
                    raise ValueError(
                        "The PDF upload was invalid."
                    ) from exc

                if (
                    len(raw) > 10 * 1024 * 1024
                    or not raw.startswith(b"%PDF-")
                ):
                    raise ValueError(
                        "Choose a valid PDF smaller than 10 MB."
                    )

                reader = PdfReader(io.BytesIO(raw))
                if reader.is_encrypted:
                    raise ValueError(
                        "Use an unencrypted PDF, "
                        "or paste the title and abstract."
                    )

                extracted = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages[:5]
                )

                if not abstract:
                    abstract = extracted[:14000].strip()

                if not title:
                    title = str(
                        (reader.metadata or {}).get(
                            "/Title",
                            "",
                        )
                    ).strip()

                if not title:
                    title = str(
                        data.get("filename", "Uploaded paper")
                    ).removesuffix(".pdf")

                if len(abstract) < 80:
                    raise ValueError(
                        "Not enough readable PDF text. "
                        "Paste its abstract "
                        "(scanned PDFs need OCR)."
                    )

            elif mode != "text":
                raise ValueError("Unknown import type.")

            if (
                not 3 <= len(title) <= 500
                or not 40 <= len(abstract) <= 30000
            ):
                raise ValueError(
                    "Provide a title (3–500 characters) "
                    "and abstract/text (40–30,000 characters)."
                )

            digest = hashlib.sha256(
                (title + "\n" + abstract).encode()
            ).hexdigest()[:24]

            paper = {
                "id": "custom:" + digest,
                "title": title,
                "abstract": abstract,
                "url": url,
                "source": (
                    "uploaded_pdf"
                    if raw
                    else "user_supplied"
                ),
                "published": "",
                "authors": [],
                "topics": [],
            }

        self.app.add_papers([paper])

        if raw:
            self.app.execute(
                """
                INSERT INTO paper_files(paper_id, data)
                VALUES (%s,%s)
                ON CONFLICT(paper_id)
                DO UPDATE SET data=EXCLUDED.data
                """,
                (paper["id"], raw),
            )

        if not self.app.query_one(
            "SELECT 1 AS ok FROM map_topics WHERE profile=%s",
            (profile,),
        ):
            raise ValueError("Unknown topic profile.")

        target_profile = profile

        if data.get("new_tab"):
            target_profile = (
                "seed_" + uuid.uuid4().hex[:12]
            )
            self.app.set_profile(
                target_profile,
                paper["title"] + ". " + paper["abstract"],
            )
            self.app.execute(
                """
                INSERT INTO map_topics(profile, title, anchor)
                VALUES (%s,%s,%s)
                """,
                (
                    target_profile,
                    paper["title"][:100],
                    paper["id"],
                ),
            )
        else:
            self.app.set_profile(
                profile,
                paper["title"] + ". " + paper["abstract"],
            )
            self.app.execute(
                """
                UPDATE map_topics
                SET anchor=%s
                WHERE profile=%s
                """,
                (paper["id"], profile),
            )

        vector = self.app.paper_vector(paper["id"])
        ids = self.app.nearest_ids(
            vector,
            min(self.app.count, 60),
        )

        self.app.executemany(
            """
            INSERT INTO map_members(profile, paper_id)
            VALUES (%s,%s)
            ON CONFLICT DO NOTHING
            """,
            [(target_profile, paper_id) for paper_id in ids],
        )
        self.app.execute(
            """
            INSERT INTO map_members(profile, paper_id)
            VALUES (%s,%s)
            ON CONFLICT DO NOTHING
            """,
            (target_profile, paper["id"]),
        )

        self.maps.pop(target_profile, None)
        self.current_map(target_profile).visit(paper["id"])
        return target_profile

    def get_pdf(self, paper_id):
        row = self.app.query_one(
            """
            SELECT data
            FROM paper_files
            WHERE paper_id=%s
            """,
            (paper_id,),
        )
        if not row:
            raise ValueError(
                "No uploaded PDF is stored for that paper."
            )
        return bytes(row["data"])

    def visit(self, profile, paper_id):
        return self.current_map(profile).visit(paper_id)

    def rate(self, profile, paper_id, rating):
        return self.current_map(profile).rate(
            paper_id,
            rating,
        )

    def restore(self, profile, visits):
        return self.current_map(profile).restore(visits)
