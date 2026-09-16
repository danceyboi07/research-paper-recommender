"""Interactive research-paper map served by FastAPI."""
import json
from datetime import date, datetime, timezone

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from recommender import as_numpy


class PaperMap:
    def __init__(self, app, profile="neuro", ids=None, anchor=None):
        self.app = app
        self.profile = profile
        self.anchor = anchor
        app._interests(profile)

        app.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_visits (
                profile TEXT REFERENCES profiles(name) ON DELETE CASCADE,
                paper_id TEXT REFERENCES papers(paper_id) ON DELETE CASCADE,
                accessed_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY(profile, paper_id)
            )
            """
        )

        rows = app.papers_with_vectors(ids)
        if not rows:
            raise ValueError("Add papers before opening the map.")
        if len(rows) > 1500:
            raise ValueError(
                "This prototype map supports up to 1,500 papers. "
                "Use a smaller collection."
            )

        self.papers = []
        vectors = []
        for row in rows:
            paper = {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "url": row["url"] or "",
                "published": row["published"].isoformat()
                if row["published"]
                else "",
                "source": row["source"],
                "authors": row["authors"] or [],
            }
            self.papers.append(paper)
            vectors.append(as_numpy(row["embedding"]))

        self.ids = {p["paper_id"] for p in self.papers}
        self.vectors = np.stack(vectors)
        self.similarities = np.clip(self.vectors @ self.vectors.T, -1, 1)

        if len(rows) == 1:
            coords = np.zeros((1, 3))
        else:
            coords = PCA(
                n_components=min(3, len(rows), self.vectors.shape[1]),
                random_state=0,
            ).fit_transform(self.vectors)
            coords = np.pad(
                coords,
                ((0, 0), (0, 3 - coords.shape[1])),
            )
            coords /= max(float(np.max(np.abs(coords))), 1e-8)

        if anchor:
            index = next(
                (
                    i
                    for i, paper in enumerate(self.papers)
                    if paper["paper_id"] == anchor
                ),
                None,
            )
            if index is not None:
                coords -= coords[index].copy()
                coords /= max(float(np.max(np.abs(coords))), 1e-8)

        for paper, xyz in zip(self.papers, coords):
            paper["xyz"] = [round(float(v), 5) for v in xyz]

        threshold = (
            0.15 if app.encoder.identity.startswith("hash") else 0.35
        )
        edges = {}
        for i in range(len(rows)):
            order = np.argsort(-self.similarities[i], kind="stable")
            for j in [int(j) for j in order if j != i][:3]:
                if self.similarities[i, j] >= threshold:
                    a, b = sorted((i, j))
                    edges[(a, b)] = {
                        "a": a,
                        "b": b,
                        "similarity": round(
                            float(self.similarities[a, b]), 4
                        ),
                    }

        self.edges = list(edges.values())
        self.make_clusters()
        self.rerank()

    def make_clusters(self):
        palette = [
            "#22f0d0",
            "#4f8cff",
            "#dd68ff",
            "#ffad45",
            "#ff5f8f",
        ]
        count = min(
            5,
            max(1, len(self.papers) // 12 + 1),
            len(np.unique(self.vectors, axis=0)),
        )

        labels = (
            KMeans(
                n_clusters=count,
                random_state=42,
                n_init=10,
            ).fit_predict(self.vectors)
            if count > 1
            else np.zeros(len(self.papers), dtype=int)
        )

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=3000,
        )

        try:
            words = vectorizer.fit_transform(
                [
                    p["title"] + ". " + p["abstract"][:1600]
                    for p in self.papers
                ]
            )
            vocabulary = vectorizer.get_feature_names_out()
        except ValueError:
            words = None
            vocabulary = []

        used = set()
        self.groups = []

        for group in range(count):
            members = np.flatnonzero(labels == group)
            if not len(members):
                continue

            terms = []
            if words is not None:
                scores = np.asarray(
                    words[members].mean(axis=0)
                ).ravel()
                for j in np.argsort(-scores):
                    term = vocabulary[j]
                    if (
                        term not in used
                        and term
                        not in {
                            "paper",
                            "model",
                            "models",
                            "results",
                            "learning",
                            "data",
                            "based",
                            "using",
                            "proposed",
                            "study",
                        }
                    ):
                        terms.append(term)
                        used.add(term)
                    if len(terms) == 2:
                        break

            label = " / ".join(terms).title() or f"Cluster {group + 1}"
            center = np.mean(
                [self.papers[i]["xyz"] for i in members],
                axis=0,
            ).tolist()

            self.groups.append(
                {
                    "id": group,
                    "label": label,
                    "color": palette[group],
                    "xyz": center,
                    "count": len(members),
                }
            )

            for i in members:
                self.papers[i]["group"] = group
                self.papers[i]["color"] = palette[group]
                self.papers[i]["group_label"] = label

    def rerank(self):
        seed, learned, _ = self.app.profile_vectors(self.profile)
        self.interest = np.clip(self.vectors @ seed, 0, 1)
        self.relevance = np.clip(self.vectors @ learned, 0, 1)

        recency = np.array(
            [
                2
                ** (
                    -max(
                        0,
                        (
                            date.today()
                            - date.fromisoformat(p["published"])
                        ).days,
                    )
                    / (365.25 * 5)
                )
                if p["published"]
                else 0
                for p in self.papers
            ]
        )

        base = 0.3 * self.interest + 0.6 * self.relevance + 0.1 * recency
        eligible = self.relevance >= 0.15
        ratings = self.ratings()
        eligible &= np.array(
            [p["paper_id"] not in ratings for p in self.papers]
        )

        redundancy = np.zeros(len(self.papers))
        self.ranking = []

        while eligible.any():
            scores = np.where(
                eligible,
                base - 0.2 * redundancy,
                -np.inf,
            )
            index = int(np.argmax(scores))
            self.ranking.append(self.papers[index]["paper_id"])
            eligible[index] = False
            redundancy = np.maximum(
                redundancy,
                self.similarities[index],
            )

    def ratings(self):
        rows = self.app.query_all(
            """
            SELECT paper_id, rating
            FROM feedback
            WHERE profile=%s
            """,
            (self.profile,),
        )
        return {r["paper_id"]: r["rating"] for r in rows}

    def state(self):
        visits = [
            {
                "id": r["paper_id"],
                "at": r["accessed_at"].astimezone(
                    timezone.utc
                ).isoformat(),
            }
            for r in self.app.query_all(
                """
                SELECT paper_id, accessed_at
                FROM paper_visits
                WHERE profile=%s
                ORDER BY accessed_at DESC, paper_id
                """,
                (self.profile,),
            )
        ]
        seen = {v["id"] for v in visits}
        return {
            "visits": visits,
            "ranking": self.ranking,
            "ratings": self.ratings(),
            "recommendations": [
                paper_id
                for paper_id in self.ranking
                if paper_id not in seen
            ][:12],
        }

    def visit(self, paper_id):
        if paper_id not in self.ids:
            raise ValueError("Unknown paper")
        self.app.execute(
            """
            INSERT INTO paper_visits(profile, paper_id, accessed_at)
            VALUES (%s,%s,%s)
            ON CONFLICT(profile,paper_id)
            DO UPDATE SET accessed_at=EXCLUDED.accessed_at
            """,
            (
                self.profile,
                paper_id,
                datetime.now(timezone.utc),
            ),
        )
        return self.state()

    def rate(self, paper_id, rating):
        self.app.rate(self.profile, paper_id, rating)
        self.rerank()
        return self.state()

    def restore(self, entries):
        if not isinstance(entries, list) or len(entries) > 10000:
            raise ValueError("Invalid history file")

        cleaned = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Invalid history entry")
            if entry.get("id") not in self.ids:
                continue

            at = datetime.fromisoformat(
                entry["at"].replace("Z", "+00:00")
            )
            if at.tzinfo is None:
                raise ValueError(
                    "History timestamps require a timezone"
                )
            cleaned.append(
                (
                    self.profile,
                    entry["id"],
                    at.astimezone(timezone.utc),
                )
            )

        self.app.executemany(
            """
            INSERT INTO paper_visits(profile, paper_id, accessed_at)
            VALUES (%s,%s,%s)
            ON CONFLICT(profile,paper_id)
            DO UPDATE SET accessed_at=GREATEST(
                paper_visits.accessed_at,
                EXCLUDED.accessed_at
            )
            """,
            cleaned,
        )
        return self.state()

    def payload(self):
        return {
            "papers": self.papers,
            "edges": self.edges,
            "state": self.state(),
            "groups": self.groups,
            "anchor": self.anchor,
            "profile": self.profile,
            "interests": self.app._interests(self.profile),
            "demo": any(
                p["source"] == "fictional_test_fixture"
                for p in self.papers
            ),
        }

    def html(self, payload=None):
        encoded = json.dumps(
            payload or self.payload(),
            ensure_ascii=True,
            allow_nan=False,
        ).replace("<", "\\u003c")
        return TEMPLATE.replace(
            "__PAPER_MAP_DATA__",
            encoded,
        )


TEMPLATE = r'''
<div id="paper-explorer">
<style>
#paper-explorer{--ink:#25332f;--muted:#737b77;--border:#e2e5df;--blue:#326dce;--purple:#8554bd;color:var(--ink);background:#f7f8f3;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;border:1px solid var(--border);border-radius:16px;overflow:hidden;max-width:1500px;margin:10px auto;box-shadow:0 8px 30px #22352d08}
#paper-explorer *{box-sizing:border-box}#paper-explorer button,#paper-explorer input{font:inherit}#paper-explorer button{cursor:pointer}#paper-explorer button:disabled{opacity:.5;cursor:default}
#paper-explorer .head{padding:22px 25px 17px;background:#fff;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;gap:15px;align-items:center}#paper-explorer h1{font-size:24px;letter-spacing:-.8px;line-height:1.2;margin:0 0 6px;font-weight:650}#paper-explorer .eyebrow{font-size:10px;letter-spacing:1.8px;font-weight:700;color:#66786d;margin-bottom:6px}#paper-explorer .sub{font-size:12px;color:var(--muted)}#paper-explorer .badge{border-radius:30px;background:#edf3e8;color:#4d674d;padding:6px 10px;font-size:11px;white-space:nowrap}#paper-explorer .demo-badge{background:#fff1d3;color:#84600b}
#paper-explorer .workspace{display:grid;grid-template-columns:255px minmax(0,1fr)}#paper-explorer aside{background:#fff;border-right:1px solid var(--border);padding:21px 15px;display:flex;flex-direction:column;min-width:0;height:815px}#paper-explorer .side-heading{font-size:16px;font-weight:650;display:flex;justify-content:space-between;margin-bottom:3px}#paper-explorer .count{font-size:11px;color:#69508a;background:#f2edf8;border-radius:12px;padding:2px 8px}#paper-explorer .trail{overflow:auto;flex:1;margin-top:18px;min-height:100px}#paper-explorer .empty{padding:28px 10px;color:#7f8781;font-size:12px;line-height:1.7;text-align:center}#paper-explorer .trail-item{width:100%;text-align:left;border:1px solid transparent;background:transparent;padding:11px 10px;border-radius:9px;margin-bottom:5px;color:var(--ink);display:block}#paper-explorer .trail-item:hover{background:#f7f5fa}#paper-explorer .trail-item.active{background:#f5f0fa;border-color:#e3d6f0}#paper-explorer .trail-title{font-size:12px;font-weight:550;line-height:1.5;display:block}#paper-explorer .trail-time{font-size:10px;color:#8c8196;margin-top:5px;display:block}#paper-explorer .side-foot{padding-top:15px;border-top:1px solid var(--border)}#paper-explorer .side-foot .sub{font-size:10px;margin-top:9px;line-height:1.6}#paper-explorer .actions{display:flex;gap:7px;flex-wrap:wrap}#paper-explorer .btn{border:1px solid #dce1db;border-radius:7px;background:#fff;padding:7px 10px;font-size:11px;color:#4c5d52;text-decoration:none;display:inline-block}#paper-explorer .btn:hover{background:#edf2ea}#paper-explorer .btn.primary{background:#294f3d;border-color:#294f3d;color:#fff}#paper-explorer .btn.primary:hover{background:#1d3c2d}
#paper-explorer .main{min-width:0}#paper-explorer .toolbar{padding:16px 18px 10px;display:flex;gap:10px;justify-content:space-between;align-items:center;flex-wrap:wrap}#paper-explorer .search{position:relative;min-width:160px;flex:1;max-width:370px}#paper-explorer input[type=search]{width:100%;padding:9px 12px;border:1px solid #dce2d8;border-radius:8px;background:#fff;font-size:12px;color:var(--ink)}#paper-explorer .legend{display:flex;gap:15px;padding:0 19px 12px;flex-wrap:wrap;font-size:11px;color:#657369}#paper-explorer .legend span{display:flex;gap:6px;align-items:center}#paper-explorer .swatch{width:9px;height:9px;background:var(--blue);border-radius:50%;display:inline-block}#paper-explorer .swatch.visited{background:var(--purple)}#paper-explorer .swatch.other{background:#b3bfb6}
#paper-explorer .map-area{position:relative;height:440px;margin:0 10px;overflow:hidden;border-radius:10px;background:radial-gradient(ellipse at 48% 50%,#ffffff 0%,#f4f6ef 85%)}#paper-explorer canvas{display:block;width:100%;height:100%;touch-action:none;outline:none}#paper-explorer canvas:focus-visible{outline:2px solid #326dce;outline-offset:-3px}#paper-explorer .tooltip{position:absolute;pointer-events:none;display:none;max-width:280px;padding:10px 13px;background:#253c31f5;color:#fff;border-radius:8px;font-size:11px;z-index:3;box-shadow:0 5px 18px #0002}#paper-explorer .map-note{position:absolute;left:13px;bottom:10px;font-size:10px;color:#89918a;background:#f7f8f3d9;padding:3px 7px;border-radius:4px;pointer-events:none}#paper-explorer .map-stat{position:absolute;right:13px;top:10px;font-size:10px;color:#8b948c;pointer-events:none}#paper-explorer .search-results{position:absolute;top:2px;left:9px;right:9px;z-index:5;max-height:180px;overflow:auto;background:#fff;box-shadow:0 4px 16px #25332f14;border:1px solid var(--border);border-radius:8px;display:none}#paper-explorer .result{display:block;text-align:left;width:100%;background:#fff;border:0;border-bottom:1px solid #f0f2ed;padding:9px 12px;font-size:12px;color:var(--ink)}#paper-explorer .result:hover{background:#eef3fa}
#paper-explorer .suggestions{padding:12px 19px 14px;border-bottom:1px solid var(--border)}#paper-explorer .suggest-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:7px}#paper-explorer .smallcaps{font-size:10px;font-weight:650;letter-spacing:1.1px;color:#667e6d}#paper-explorer .chips{display:flex;gap:7px;overflow:auto;padding:2px 0 4px}#paper-explorer .chip{white-space:nowrap;max-width:220px;overflow:hidden;text-overflow:ellipsis;border:1px solid #d8e3f3;background:#f0f5ff;color:#3865a5;border-radius:6px;padding:6px 9px;font-size:11px;flex-shrink:0}#paper-explorer .details{background:#fff;padding:17px 22px;height:228px;overflow:auto}#paper-explorer .details h2{font-size:17px;line-height:1.4;margin:6px 0 5px;font-weight:600;letter-spacing:-.3px}#paper-explorer .details p{font-size:12px;line-height:1.7;color:#647066;margin:9px 0}#paper-explorer .detail-meta{font-size:10px;color:#8a928c;margin-bottom:9px}#paper-explorer .detail-label{font-size:10px;color:#8258a8;font-weight:650;text-transform:uppercase;letter-spacing:1px}#paper-explorer .blank-detail{display:flex;gap:15px;align-items:center;height:100%}#paper-explorer .blank-icon{font-size:30px;color:#879f8c;border:1px solid #e0e7dc;background:#f8faf5;border-radius:13px;width:54px;height:54px;display:flex;align-items:center;justify-content:center;flex-shrink:0}#paper-explorer .footer{border-top:1px solid var(--border);padding:9px 17px;display:flex;justify-content:space-between;gap:12px;background:#fff;font-size:10px;color:#8a928c;flex-wrap:wrap}#paper-explorer .status-error{color:#ab6636}#paper-explorer .mini-btn{background:none;border:0;font-size:10px;color:#718176;text-decoration:underline;padding:0}
@media(max-width:760px){#paper-explorer .workspace{grid-template-columns:195px minmax(0,1fr)}#paper-explorer aside{padding:17px 10px}#paper-explorer .head{padding:18px 16px}#paper-explorer .head .badge{max-width:120px;white-space:normal;text-align:center}#paper-explorer .legend{gap:8px;font-size:10px}#paper-explorer .details{padding:13px 15px}#paper-explorer .toolbar{padding:12px 12px 9px}#paper-explorer h1{font-size:21px}}
@media(max-width:490px){#paper-explorer .workspace{grid-template-columns:145px minmax(0,1fr)}#paper-explorer .trail-title{font-size:11px}#paper-explorer .side-heading{font-size:13px}#paper-explorer .head .badge{display:none}#paper-explorer .legend{padding:0 12px 10px}#paper-explorer .map-area{height:410px}}

#paper-explorer{background:#f5f8fa;--border:#e1e8ed}#paper-explorer .head{background:linear-gradient(115deg,#fff,#edf6f5)}
#paper-explorer .workspace-tools{padding:15px 21px 0;background:#fff}#paper-explorer .topic-form{display:flex;gap:9px;align-items:center}#paper-explorer .topic-input{flex:1;min-width:130px;padding:10px 13px;border:1px solid #d9e3e7;border-radius:8px;font:inherit;font-size:12px}#paper-explorer .tabs{display:flex;gap:6px;overflow:auto;padding:15px 0 0}#paper-explorer .topic-tab{background:#f4f7f8;border:1px solid #e0e7e9;border-bottom:0;border-radius:9px 9px 0 0;padding:10px 14px;white-space:nowrap;font-size:12px;color:#71838a;max-width:260px;overflow:hidden;text-overflow:ellipsis}#paper-explorer .topic-tab.active{background:#e6f2f0;color:#206557;border-color:#c7e1da;font-weight:600}#paper-explorer .workspace-message{font-size:12px;color:#496e66;padding:9px 0;min-height:15px}#paper-explorer .workspace-message:empty{padding:0}#paper-explorer .workspace-message.error{color:#ab572b}
#paper-explorer .trail-item{padding:9px 10px}#paper-explorer .trail-select{border:0;background:transparent;padding:0;text-align:left;width:100%;color:inherit}#paper-explorer .trail-open{display:inline-block;color:#256c91;font-size:11px;margin-top:7px;text-decoration:none;font-weight:550}#paper-explorer .trail-open:hover{text-decoration:underline}#paper-explorer .no-link{font-size:10px;color:#939eaa}#paper-explorer .cluster-legend{display:flex;gap:6px 12px;flex-wrap:wrap;padding:0 19px 10px;font-size:10px;color:#627684}#paper-explorer .cluster-legend span{display:flex;align-items:center;gap:5px}#paper-explorer .map-area{background:radial-gradient(ellipse at center,#fcfefe,#edf3f6)}#paper-explorer aside{height:870px}#paper-explorer .map-area{height:440px}#paper-explorer .legend .ring{display:inline-block;width:12px;height:12px;border:2px solid #62879a;border-radius:50%;background:transparent}
#paper-explorer .modal-backdrop{position:fixed;inset:0;background:#1b344650;z-index:50;display:flex;align-items:center;justify-content:center;padding:18px}#paper-explorer [hidden]{display:none!important}#paper-explorer .modal{background:#fff;box-shadow:0 18px 70px #1236;border-radius:15px;width:560px;max-width:100%;max-height:90vh;overflow:auto;padding:24px}#paper-explorer .modal h2{font-size:20px;margin:0 0 6px}#paper-explorer .modal label{display:block;font-size:12px;font-weight:550;margin:12px 0 5px;color:#47625b}#paper-explorer .modal input:not([type=checkbox]),#paper-explorer .modal textarea,#paper-explorer .modal select{width:100%;border:1px solid #d4dfda;border-radius:7px;padding:9px 11px;font:inherit;font-size:12px}#paper-explorer .modal textarea{resize:vertical;min-height:110px}#paper-explorer .modal .actions{margin-top:18px;justify-content:flex-end}#paper-explorer .modal-error{color:#ad5226;font-size:12px;margin-top:9px}#paper-explorer .busy button,#paper-explorer .busy canvas{pointer-events:none}#paper-explorer .busy .workspace{opacity:.7}#paper-explorer .busy{cursor:progress}

/* Dark luminous theme. These rules cover every visible surface. */
#paper-explorer{--ink:#f4f8ff;--muted:#9da9ba;--border:#252c38;--blue:#57a6ff;--purple:#da78ff;color:var(--ink);background:#050608;border-color:#242a35;box-shadow:0 18px 55px #000c;color-scheme:dark}
#paper-explorer .head{background:linear-gradient(120deg,#050608 0%,#090d14 56%,#071719 100%);border-color:var(--border)}
#paper-explorer h1,#paper-explorer h2,#paper-explorer .side-heading,#paper-explorer .trail-title{color:#f7faff;text-shadow:0 0 18px #7dcfff18}
#paper-explorer .eyebrow,#paper-explorer .smallcaps{color:#77f4de}
#paper-explorer .sub,#paper-explorer .detail-meta,#paper-explorer .footer,#paper-explorer .map-stat{color:var(--muted)}
#paper-explorer .badge{background:#0e231f;color:#82ffe2;border:1px solid #245c50;box-shadow:0 0 20px #22f0d01c}
#paper-explorer .demo-badge{background:#261b08;color:#ffd078;border-color:#6e4a16}
#paper-explorer .workspace-tools,#paper-explorer aside,#paper-explorer .details,#paper-explorer .footer{background:#080a0f;border-color:var(--border)}
#paper-explorer .main,#paper-explorer .toolbar,#paper-explorer .suggestions{background:#06080c;border-color:var(--border)}
#paper-explorer .topic-input,#paper-explorer input[type=search],#paper-explorer .modal input:not([type=checkbox]),#paper-explorer .modal textarea,#paper-explorer .modal select{background:#0d1119;color:#f4f8ff;border-color:#30394a;box-shadow:inset 0 0 0 1px #ffffff05}
#paper-explorer input::placeholder,#paper-explorer textarea::placeholder{color:#758196}
#paper-explorer input:focus,#paper-explorer textarea:focus,#paper-explorer select:focus{outline:2px solid #22f0d070;border-color:#22f0d0}
#paper-explorer .btn{background:#10151e;color:#dce8f7;border-color:#364154}
#paper-explorer .btn:hover{background:#182131;border-color:#57a6ff;color:#fff;box-shadow:0 0 14px #57a6ff25}
#paper-explorer .btn.primary{background:linear-gradient(135deg,#0c8a75,#126b9d);border-color:#31d8cf;color:#fff;box-shadow:0 0 18px #22f0d025}
#paper-explorer .btn.primary:hover{background:linear-gradient(135deg,#10aa90,#167fb9);box-shadow:0 0 24px #22f0d044}
#paper-explorer .tabs{scrollbar-color:#37465b #080a0f}
#paper-explorer .topic-tab{background:#0b0e14;border-color:#29313f;color:#a5b2c4}
#paper-explorer .topic-tab:hover{color:#fff;background:#111824;border-color:#3b4a60}
#paper-explorer .topic-tab.active{background:linear-gradient(180deg,#10262a,#0b171d);color:#8fffe9;border-color:#247d73;box-shadow:inset 0 2px 0 #2be8ce,0 -5px 18px #22f0d012}
#paper-explorer .workspace-message{color:#75dccb}#paper-explorer .workspace-message.error,#paper-explorer .status-error,#paper-explorer .modal-error{color:#ff9f73}
#paper-explorer .count{color:#edc3ff;background:#241331;border:1px solid #673780;box-shadow:0 0 12px #dd68ff25}
#paper-explorer .trail-item{color:var(--ink);border-color:transparent}
#paper-explorer .trail-item:hover{background:#101520}
#paper-explorer .trail-item.active{background:linear-gradient(135deg,#171126,#111a27);border-color:#754b91;box-shadow:0 0 18px #dd68ff17}
#paper-explorer .trail-time{color:#a891ba}#paper-explorer .trail-open{color:#6ec9ff}#paper-explorer .trail-open:hover{color:#b9e7ff}
#paper-explorer .no-link,#paper-explorer .empty{color:#8793a5}
#paper-explorer .legend,#paper-explorer .cluster-legend{color:#aab6c7}
#paper-explorer .swatch{box-shadow:0 0 9px currentColor}#paper-explorer .legend .ring{border-color:#7bc9ff;box-shadow:0 0 10px #57a6ff88}
#paper-explorer .map-area{background:radial-gradient(ellipse at 48% 44%,#0b1520 0%,#04070b 48%,#010203 100%);border:1px solid #1d2733;box-shadow:inset 0 0 70px #000,0 0 24px #0f91b70d}
#paper-explorer canvas:focus-visible{outline-color:#57cfff}
#paper-explorer .tooltip{background:#0d121bef;color:#f8fbff;border:1px solid #41536a;box-shadow:0 8px 28px #000b,0 0 18px #57a6ff22}
#paper-explorer .map-note{color:#9aa8ba;background:#05080ce6;border:1px solid #222c38}
#paper-explorer .search-results{background:#0b0f16;border-color:#303a4a;box-shadow:0 10px 30px #000b}
#paper-explorer .result{background:#0b0f16;color:#e9f2ff;border-color:#232a36}
#paper-explorer .result:hover{background:#142033;color:#fff}
#paper-explorer .chip{border-color:#294d70;background:linear-gradient(135deg,#0c1725,#10203a);color:#84c5ff;box-shadow:0 0 12px #4f8cff12}
#paper-explorer .chip:hover{border-color:#58a9ff;color:#d8efff;box-shadow:0 0 18px #4f8cff30}
#paper-explorer .details p{color:#b5c0cf}#paper-explorer .detail-label{color:#e08bff}#paper-explorer .blank-icon{color:#7ef4dc;border-color:#28534d;background:#0b1717;box-shadow:0 0 18px #22f0d018}
#paper-explorer .modal-backdrop{background:#000c;backdrop-filter:blur(5px)}
#paper-explorer .modal{background:#0a0d13;color:#f4f8ff;border:1px solid #303949;box-shadow:0 22px 90px #000,0 0 35px #4f8cff16}
#paper-explorer .modal label{color:#c3cedc}
#paper-explorer ::-webkit-scrollbar{width:9px;height:9px}#paper-explorer ::-webkit-scrollbar-track{background:#080a0f}#paper-explorer ::-webkit-scrollbar-thumb{background:#303a49;border-radius:9px;border:2px solid #080a0f}#paper-explorer ::-webkit-scrollbar-thumb:hover{background:#46566c}
</style>
<header class="head"><div><div class="eyebrow">YOUR RESEARCH, CONNECTED</div><h1>Paper explorer</h1><div class="sub">Find your next paper. See how it connects.</div></div><span class="badge" id="source-badge">Research collection</span></header>

<div class="workspace-tools"><form id="topic-form" class="topic-form"><input id="topic-input" class="topic-input" aria-label="New research topic" placeholder="New topic, e.g. AI models for schizophrenia diagnosis" maxlength="250" required><button class="btn primary" id="new-topic" type="submit">Create topic map</button><button class="btn" id="add-paper" type="button">+ Add your paper</button></form><div id="workspace-status" class="workspace-message" role="status"></div><nav id="tabs" class="tabs" aria-label="Research topic tabs"></nav></div>
<div id="paper-modal" class="modal-backdrop" hidden><form id="paper-form" class="modal" role="dialog" aria-modal="true" aria-labelledby="import-title"><h2 id="import-title">Start from your paper</h2><div class="sub">Add a paper, then explore its closest matches in your database.</div><label for="import-mode">Import method</label><select id="import-mode"><option value="arxiv">arXiv link or ID</option><option value="text">Title and abstract</option><option value="pdf">Upload a PDF</option></select><div id="arxiv-fields"><label for="arxiv-url">arXiv abstract/PDF link or ID</label><input id="arxiv-url" placeholder="https://arxiv.org/abs/2203.11610"></div><div id="text-fields" hidden><label for="paper-title">Paper title</label><input id="paper-title" maxlength="500"><label for="paper-abstract">Abstract (optional if uploading a readable PDF)</label><textarea id="paper-abstract" maxlength="30000"></textarea><label for="paper-url">Original paper URL (optional)</label><input id="paper-url" type="url" placeholder="https://..."></div><div id="pdf-fields" hidden><label for="paper-file">PDF, up to 10 MB</label><input id="paper-file" type="file" accept="application/pdf,.pdf"><div class="sub">Uses text from the first five pages unless you supply an abstract. Scanned PDFs need an abstract pasted above.</div></div><label><input type="checkbox" id="paper-new-tab"> Open a separate tab for this paper</label><div id="import-error" class="modal-error" role="alert"></div><div class="actions"><button type="button" id="cancel-import" class="btn">Cancel</button><button type="submit" class="btn primary">Add &amp; center map</button></div></form></div>

<div class="workspace"><aside><div class="side-heading">Your reading trail <span id="visit-count" class="count">0</span></div><div class="sub">Every paper you have clicked.</div><div id="trail" class="trail"></div><div class="side-foot"><div class="actions"><button class="btn" id="export">Export history</button><button class="btn" id="restore">Restore</button><input type="file" id="history-file" accept=".json,application/json" hidden></div><div class="sub">Clicks are saved in this Colab runtime. Export your history before disconnecting.</div></div></aside>
<main class="main"><div class="toolbar"><div class="search"><input id="search" type="search" aria-label="Find a paper" placeholder="Find a paper in this collection…"></div><div class="actions"><button id="reset-view" class="btn">Reset view</button><button id="toggle-labels" class="btn" aria-pressed="false">Labels</button></div></div><div class="legend"><span><i class="ring"></i>New recommendation</span><span>✓ Visited</span><span>★ Your starting paper</span><span>Color = research cluster</span></div><div id="cluster-legend" class="cluster-legend"></div>
<div class="map-area"><canvas id="graph" tabindex="0" role="img" aria-label="Interactive 3D paper similarity map. Drag to rotate, scroll to zoom. Use search or recommendation buttons to select papers with the keyboard."></canvas><div id="tooltip" class="tooltip"></div><div id="search-results" class="search-results"></div><div class="map-note">Drag to rotate · Scroll to zoom · Click a dot</div><div class="map-stat" id="map-stat"></div></div>
<div class="suggestions"><div class="suggest-head"><span class="smallcaps">EXPLORE NEXT</span><span id="new-count" class="sub"></span></div><div class="chips" id="chips"></div></div>
<section id="details" class="details" aria-live="polite"><div class="blank-detail"><div class="blank-icon">⌘</div><div><div class="smallcaps">FOLLOW A CONNECTION</div><h2>One paper leads to another.</h2><p>Click a paper to see its abstract and connections.<br>It will gain a checkmark and join your reading trail on the left.</p></div></div></section></main></div>
<footer class="footer"><span>3D projection of embeddings · Lines show text similarity, not citations. Colors are automatic clusters.</span><span id="sync-status" role="status">Ready</span></footer>
<script type="application/json" id="paper-map-data">__PAPER_MAP_DATA__</script>
<script>
(()=>{
'use strict';
const root=document.getElementById('paper-explorer'),$=s=>root.querySelector(s);
const API_ROOT=new URL('api/',window.location.href);
const apiUrl=path=>new URL(String(path).replace(/^\/+/,''),API_ROOT).toString();
async function api(path,options={}){
 const response=await fetch(apiUrl(path),options);
 const type=response.headers.get('content-type')||'';
 const body=type.includes('application/json')?await response.json():await response.text();
 if(!response.ok)throw new Error(body?.detail||body?.error||body||('HTTP '+response.status));
 return body;
}
let data=JSON.parse($('#paper-map-data').textContent), papers=data.papers;
let byId=new Map(papers.map((p,i)=>[p.paper_id,{...p,index:i}]));
let state=data.state, selected=null, hover=null, labels=false, query='', yaw=-.55, pitch=.42, zoom=1, projected=[], dirty=true, moving=null;
let visits=new Map(), fresh=new Set(), pending=0, failedSave=false, busy=false, queue=Promise.resolve();
const colors={fresh:'#57a6ff',visited:'#da78ff',other:'#9aabc0',selected:'#ffe66d'};
const canvas=$('#graph'),ctx=canvas.getContext('2d'),tip=$('#tooltip');
if(!ctx)throw new Error('Your browser could not create the map canvas.');
let width=600,height=440;
const make=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;};
const status=(text,error=false)=>{$('#sync-status').textContent=text;$('#sync-status').classList.toggle('status-error',error);};
function recommendations(){return state.ranking.filter(id=>!visits.has(id)&&!state.ratings[id]).slice(0,12);}
function refresh(){
 visits=new Map(state.visits.filter(v=>byId.has(v.id)).map(v=>[v.id,v])); fresh=new Set(recommendations());
 $('#visit-count').textContent=visits.size;$('#new-count').textContent=fresh.size+' unseen';
 const trail=$('#trail');trail.replaceChildren();
 if(!visits.size)trail.append(make('div','empty','Your trail starts here. Click any dot on the map, or choose a paper below it.'));
 [...visits.values()].sort((a,b)=>b.at.localeCompare(a.at)).forEach(v=>{
  const p=byId.get(v.id),item=make('div','trail-item'+(selected===v.id?' active':'')),b=make('button','trail-select');
  item.dataset.paperId=v.id;b.append(make('span','trail-title','✓  '+p.title));
  b.append(make('span','trail-time',new Date(v.at).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'})));
  b.onclick=()=>select(v.id);item.append(b);item.append(paperLink(p,'trail-open'));trail.append(item);
 });
 const chips=$('#chips');chips.replaceChildren();
 [...fresh].forEach(id=>{const p=byId.get(id),b=make('button','chip',p.title);b.title=p.title;b.dataset.paperId=id;b.onclick=()=>select(id);chips.append(b);});
 if(!fresh.size)chips.append(make('span','sub','No more unseen matches in this collection.'));
 dirty=true;
}
function renderDetails(){
 if(!selected)return;
 const p=byId.get(selected),box=$('#details');box.replaceChildren();
 box.append(make('div','detail-label','✓ Visited · '+(selected===data.anchor?'★ Starting paper':state.ratings[selected]===1?'Saved to interests':p.group_label||p.paper_id)));
 box.append(make('h2','',p.title));
 box.append(make('div','detail-meta',[p.published,(p.authors||[]).slice(0,3).join(', ')].filter(Boolean).join(' · ')));
 const actions=make('div','actions');
 actions.append(paperLink(p,'btn primary'));
 const liked=state.ratings[selected]===1,save=make('button','btn',liked?'Undo save':'Save to interests');
 save.onclick=()=>mutate('rate',{id:p.paper_id,rating:liked?0:1});actions.append(save);
 box.append(actions,make('p','',p.abstract));
 const neighbors=data.edges.filter(e=>e.a===p.index||e.b===p.index).map(e=>({p:papers[e.a===p.index?e.b:e.a],s:e.similarity})).sort((a,b)=>b.s-a.s);
 if(neighbors.length){box.append(make('div','smallcaps','CONNECTED PAPERS'));neighbors.forEach(n=>{const b=make('button','btn',n.p.title);b.style.margin='7px 5px 0 0';b.onclick=()=>select(n.p.paper_id);box.append(b);});}
}
function optimistic(action,arg){
 if(action==='visit')state.visits=[{id:arg.id,at:new Date().toISOString()},...state.visits.filter(v=>v.id!==arg.id)];
 if(action==='restore'){const m=new Map(state.visits.map(v=>[v.id,v]));arg.visits.forEach(v=>{if(byId.has(v.id)&&(!m.has(v.id)||m.get(v.id).at<v.at))m.set(v.id,v);});state.visits=[...m.values()];}
}
function mutate(action,arg){
 optimistic(action,arg);refresh();renderDetails();
 pending++;status('Saving via REST API…');
 queue=queue.catch(()=>{}).then(async()=>{
  try{
   let next;
   if(action==='visit'){
    next=await api('visits',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({profile:data.profile,paper_id:arg.id})});
   }else if(action==='rate'){
    next=await api('feedback',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({profile:data.profile,paper_id:arg.id,rating:arg.rating})});
   }else if(action==='restore'){
    next=await api('history/restore',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({profile:data.profile,visits:arg.visits})});
   }else{
    throw new Error('Unknown state action: '+action);
   }
   if(!next||!Array.isArray(next.visits))throw new Error('Unexpected API response.');
   const merged=new Map(next.visits.map(v=>[v.id,v]));
   state.visits.forEach(v=>{if(!merged.has(v.id)||merged.get(v.id).at<v.at)merged.set(v.id,v);});
   state={...next,visits:[...merged.values()]};
   pending--;refresh();renderDetails();status(pending?'Saving…':'Saved · PostgreSQL');
  }catch(error){
   pending--;failedSave=true;status('Save failed · '+error.message,true);console.warn('Paper map:',error.message);
  }
 });return queue;
}
function select(id){if(busy||!byId.has(id))return;selected=id;hover=null;tip.style.display='none';$('#search-results').style.display='none';mutate('visit',{id});}

function paperLink(p,cls){
 try{const url=new URL(p.url);if(['https:','http:'].includes(url.protocol)){const a=make('a',cls,'Open paper ↗');a.href=url.href;a.target='_blank';a.rel='noopener noreferrer';a.setAttribute('aria-label','Open paper: '+p.title);return a;}}catch(e){}
 if(p.has_pdf){const a=make('a',cls,'Open uploaded PDF ↗');a.href=apiUrl('pdf?paper_id='+encodeURIComponent(p.paper_id));a.target='_blank';a.rel='noopener noreferrer';return a;}
 return make('span','no-link',data.demo?'Fictional demo · no source URL':'No source URL provided');
}
function header(){
 const tabs=$('#tabs');tabs.replaceChildren();
 (data.tabs||[{profile:data.profile,title:'Current collection'}]).forEach(t=>{const b=make('button','topic-tab'+(t.profile===data.profile?' active':''),t.title);b.title=t.title;b.setAttribute('aria-pressed',String(t.profile===data.profile));b.onclick=()=>{if(t.profile!==data.profile)workspaceAction('switch_topic',{profile:t.profile},'Opening topic…')};tabs.append(b);});
 const legend=$('#cluster-legend');legend.replaceChildren();(data.groups||[]).forEach(g=>{const item=make('span'),dot=make('i','swatch');dot.style.background=g.color;item.append(dot,document.createTextNode(g.label+' · '+g.count));legend.append(item);});
 $('#source-badge').textContent=data.demo?'Includes fictional demo papers':papers.length+' research papers';$('#source-badge').classList.toggle('demo-badge',data.demo);$('#map-stat').textContent=papers.length+' papers · '+data.edges.length+' connections';
}
function loadPayload(next){
 data=next;papers=data.papers;byId=new Map(papers.map((p,i)=>[p.paper_id,{...p,index:i}]));state=data.state;
 selected=data.anchor||null;hover=null;query='';$('#search').value='';$('#search-results').style.display='none';tip.style.display='none';yaw=-.55;pitch=.42;zoom=1;failedSave=false;
 refresh();header();if(selected)renderDetails();else $('#details').replaceChildren(make('div','blank-detail','Choose a paper on this topic map to view its abstract and open its source.'));
 resize();draw();
}
async function workspaceAction(action,arg,message){
 if(busy)return false;
 busy=true;root.classList.add('busy');$('#workspace-status').classList.remove('error');$('#workspace-status').textContent=message;$('#import-error').textContent='';
 const controls=[...root.querySelectorAll('button,input,select,textarea')];controls.forEach(e=>e.disabled=true);
 try{
  await queue;
  let result;
  if(action==='new_topic'){
   result=await api('topics',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic:arg.topic})});
  }else if(action==='switch_topic'){
   result=await api('topics/switch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({profile:arg.profile})});
  }else if(action==='add_paper'){
   result=await api('papers/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...arg,profile:data.profile})});
  }else{
   throw new Error('Unknown workspace action: '+action);
  }
  if(!result?.payload)throw new Error('No topic data returned by the API.');
  loadPayload(result.payload);
  $('#workspace-status').textContent=action==='add_paper'?'Map centered on your paper. Related papers come from your indexed collection.':action==='new_topic'?'New topic ready. Papers downloaded from arXiv.':'';
  $('#paper-modal').hidden=true;status('Connected · FastAPI + PostgreSQL');return true;
 }catch(error){
  $('#workspace-status').classList.add('error');$('#workspace-status').textContent=error.message;$('#import-error').textContent=error.message;status('Request failed · previous map retained',true);return false;
 }finally{
  busy=false;root.classList.remove('busy');controls.forEach(e=>e.disabled=false);
 }
}
$('#topic-form').onsubmit=async e=>{e.preventDefault();const topic=$('#topic-input').value.trim();if(topic.length<3)return;const ok=await workspaceAction('new_topic',{topic},'Searching arXiv and embedding papers… This may take a minute.');if(ok)$('#topic-input').value='';};
$('#add-paper').onclick=()=>{$('#paper-modal').hidden=false;$('#import-error').textContent='';$('#arxiv-url').focus();};
$('#cancel-import').onclick=()=>{$('#paper-modal').hidden=true;};
$('#import-mode').onchange=()=>{const mode=$('#import-mode').value;$('#arxiv-fields').hidden=mode!=='arxiv';$('#text-fields').hidden=mode==='arxiv';$('#pdf-fields').hidden=mode!=='pdf';};
$('#paper-form').onsubmit=async e=>{e.preventDefault();const mode=$('#import-mode').value,arg={mode,arxiv:$('#arxiv-url').value,title:$('#paper-title').value,abstract:$('#paper-abstract').value,url:$('#paper-url').value,new_tab:$('#paper-new-tab').checked};
 if(mode==='pdf'){const f=$('#paper-file').files[0];if(!f||f.size>10*1024*1024){$('#import-error').textContent='Choose a PDF up to 10 MB.';return;}arg.filename=f.name;arg.pdf=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=reject;reader.readAsDataURL(f);});}
 await workspaceAction('add_paper',arg,'Reading your paper and rearranging the map…');
};
root.addEventListener('keydown',e=>{if(e.key==='Escape'&&!busy)$('#paper-modal').hidden=true;});
function project(xyz){
 const [x,y,z]=xyz,xx=x*Math.cos(yaw)+z*Math.sin(yaw),zz=-x*Math.sin(yaw)+z*Math.cos(yaw);
 const yy=y*Math.cos(pitch)-zz*Math.sin(pitch),depth=y*Math.sin(pitch)+zz*Math.cos(pitch);
 const perspective=3.8/(3.8+depth),scale=Math.min(width*.30,height*.35)*zoom;
 return {x:width/2+xx*scale*perspective,y:height*.49-yy*scale*perspective,z:depth,s:perspective};
}
function line(a,b,color,w=1){ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=color;ctx.lineWidth=w;ctx.stroke();}
function draw(){
 if(!dirty)return;dirty=false;ctx.clearRect(0,0,width,height);
 // A quiet perspective grid grounds the abstract embedding coordinates.
 for(let k=-1;k<=1.001;k+=.25){
  line(project([-1,-1,k]),project([1,-1,k]),'#6688a52b');line(project([k,-1,-1]),project([k,-1,1]),'#6688a53d');
  line(project([-1,k,-1]),project([1,k,-1]),'#6688a52b');line(project([k,-1,-1]),project([k,1,-1]),'#6688a534');
 }
 for(const group of data.groups||[]){const c=project(group.xyz);const radius=95*zoom;const glow=ctx.createRadialGradient(c.x,c.y,0,c.x,c.y,radius);glow.addColorStop(0,group.color+'2b');glow.addColorStop(.45,group.color+'12');glow.addColorStop(1,group.color+'00');ctx.fillStyle=glow;ctx.fillRect(c.x-radius,c.y-radius,radius*2,radius*2);}
 projected=papers.map((p,i)=>({...project(p.xyz),i,id:p.paper_id}));
 const focus=selected?byId.get(selected).index:-1;
 for(const edge of data.edges){const relevant=edge.a===focus||edge.b===focus,a=projected[edge.a],b=projected[edge.b],gradient=ctx.createLinearGradient(a.x,a.y,b.x,b.y);gradient.addColorStop(0,(papers[edge.a].color||'#57a6ff')+(relevant?'e0':'69'));gradient.addColorStop(1,(papers[edge.b].color||'#dd68ff')+(relevant?'e0':'69'));ctx.shadowColor=relevant?(papers[edge.a].color||'#57a6ff'):'transparent';ctx.shadowBlur=relevant?10:0;line(a,b,gradient,relevant?2.1:1);ctx.shadowBlur=0;}
 const sorted=[...projected].sort((a,b)=>b.z-a.z);
 for(const dot of sorted){
  const p=papers[dot.i],seen=visits.has(dot.id),recommended=fresh.has(dot.id),match=!query||p.title.toLowerCase().includes(query)||p.abstract.toLowerCase().includes(query);
  const r=(seen?7:recommended?7:4.4)*dot.s;dot.r=r;projected[dot.i].r=r;
  ctx.globalAlpha=match?1:.15;
  if(dot.id===selected||dot.id===hover){ctx.shadowColor=dot.id===selected?colors.selected:(p.color||colors.fresh);ctx.shadowBlur=14;ctx.beginPath();ctx.arc(dot.x,dot.y,r+5,0,Math.PI*2);ctx.strokeStyle=dot.id===selected?colors.selected:'#b9e6ff';ctx.lineWidth=2;ctx.stroke();ctx.shadowBlur=0;}
  if(recommended&&!seen){ctx.shadowColor=p.color||colors.fresh;ctx.shadowBlur=12;ctx.beginPath();ctx.arc(dot.x,dot.y,r+3,0,Math.PI*2);ctx.strokeStyle=(p.color||colors.fresh)+'d9';ctx.lineWidth=1.6;ctx.stroke();ctx.shadowBlur=0;}
  ctx.shadowColor=p.color||colors.other;ctx.shadowBlur=(recommended?17:seen?12:7)*dot.s;ctx.beginPath();ctx.arc(dot.x,dot.y,r,0,Math.PI*2);ctx.fillStyle=p.color||colors.other;ctx.fill();ctx.shadowBlur=0;ctx.strokeStyle='#ecfbff';ctx.lineWidth=1.25;ctx.stroke();
  if(dot.id===data.anchor){ctx.shadowColor='#ffd45e';ctx.shadowBlur=10;ctx.font='bold 18px sans-serif';ctx.fillStyle='#ffd45e';ctx.fillText('★',dot.x-7,dot.y-r-8);ctx.shadowBlur=0;}
  if(seen&&r>5){ctx.beginPath();ctx.moveTo(dot.x-r*.42,dot.y);ctx.lineTo(dot.x-r*.1,dot.y+r*.3);ctx.lineTo(dot.x+r*.45,dot.y-r*.33);ctx.strokeStyle='#fff';ctx.lineWidth=1.2;ctx.stroke();}
  if((labels||dot.id===selected||dot.id===hover)&&match){const t=p.title.length>46?p.title.slice(0,43)+'…':p.title;ctx.font='11px -apple-system, sans-serif';const m=ctx.measureText(t);let x=Math.min(dot.x+r+6,width-m.width-9);x=Math.max(6,x);ctx.fillStyle='#070b12ee';ctx.fillRect(x-3,dot.y-9,m.width+6,17);ctx.fillStyle='#f4f8ff';ctx.fillText(t,x,dot.y+3);}
  ctx.globalAlpha=1;
 }
}
function hit(x,y){return [...projected].sort((a,b)=>a.z-b.z).find(p=>Math.hypot(x-p.x,y-p.y)<Math.max((p.r||6)+5,10));}
function point(event){const rect=canvas.getBoundingClientRect();return {x:event.clientX-rect.left,y:event.clientY-rect.top};}
canvas.addEventListener('pointerdown',e=>{const p=point(e);moving={...p,startX:p.x,startY:p.y,drag:false};canvas.setPointerCapture(e.pointerId);tip.style.display='none';});
canvas.addEventListener('pointermove',e=>{const p=point(e);if(moving){if(Math.hypot(p.x-moving.startX,p.y-moving.startY)>5)moving.drag=true;if(moving.drag){yaw+=(p.x-moving.x)*.008;pitch=Math.max(-1.35,Math.min(1.35,pitch+(p.y-moving.y)*.008));dirty=true;}moving.x=p.x;moving.y=p.y;return;}
 const found=hit(p.x,p.y);hover=found?.id||null;canvas.style.cursor=found?'pointer':'grab';dirty=true;
 if(found){const record=byId.get(found.id);tip.textContent=(visits.has(found.id)?'✓ Visited · ':fresh.has(found.id)?'New recommendation · ':'Other paper · ')+record.title+' · '+(record.group_label||'');tip.style.display='block';tip.style.left=Math.max(5,Math.min(width-285,p.x+14))+'px';tip.style.top=Math.max(5,Math.min(height-95,p.y+14))+'px';}else tip.style.display='none';});
canvas.addEventListener('pointerup',e=>{if(moving&&!moving.drag){const p=point(e),found=hit(p.x,p.y);if(found)select(found.id);}moving=null;});
canvas.addEventListener('pointercancel',()=>{moving=null;});
canvas.addEventListener('pointerleave',()=>{hover=null;tip.style.display='none';dirty=true;});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.45,Math.min(3,zoom*Math.exp(-e.deltaY*.001)));dirty=true;},{passive:false});
$('#reset-view').onclick=()=>{yaw=-.55;pitch=.42;zoom=1;dirty=true;};
$('#toggle-labels').onclick=()=>{labels=!labels;$('#toggle-labels').setAttribute('aria-pressed',String(labels));dirty=true;};
$('#search').oninput=e=>{query=e.target.value.trim().toLowerCase();dirty=true;const box=$('#search-results');box.replaceChildren();box.style.display=query?'block':'none';if(!query)return;const matches=papers.filter(p=>p.title.toLowerCase().includes(query)||p.abstract.toLowerCase().includes(query));matches.slice(0,30).forEach(p=>{const b=make('button','result',(visits.has(p.paper_id)?'✓ ':'')+p.title);b.onclick=()=>{query='';$('#search').value='';select(p.paper_id);};box.append(b);});if(!matches.length)box.append(make('div','empty','No papers match this search.'));};
$('#export').onclick=()=>{const text=JSON.stringify({version:1,profile:data.profile,visits:state.visits},null,2),url=URL.createObjectURL(new Blob([text],{type:'application/json'})),a=make('a');a.href=url;a.download='paper_reading_history.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);status('History exported');};
$('#restore').onclick=()=>$('#history-file').click();
$('#history-file').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>2000000)throw new Error('History file is too large.');const payload=JSON.parse(await file.text());if(payload.version!==1||payload.profile!==data.profile||!Array.isArray(payload.visits)||payload.visits.length>10000)throw new Error('Use a history file for this profile.');const clean=payload.visits.map(v=>{if(!v||typeof v.id!=='string'||typeof v.at!=='string'||!Number.isFinite(Date.parse(v.at)))throw new Error('Invalid history entry.');return {id:v.id,at:new Date(v.at).toISOString()};}).filter(v=>byId.has(v.id));await mutate('restore',{visits:clean});if(!clean.length)status('No matching papers found in this history file.',true);}catch(error){status(error.message,true);}e.target.value='';};
function resize(){const rect=canvas.getBoundingClientRect();width=rect.width;height=rect.height;const dpr=Math.min(devicePixelRatio||1,2);canvas.width=width*dpr;canvas.height=height*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);dirty=true;}
new ResizeObserver(resize).observe(canvas);
function tick(){if(!root.isConnected)return;draw();requestAnimationFrame(tick);}refresh();resize();draw();tick();
if(window.google?.colab?.output?.setIframeHeight)requestAnimationFrame(()=>google.colab.output.setIframeHeight(document.documentElement.scrollHeight,true));
header();
status('Connected · FastAPI REST API · PostgreSQL + pgvector');
// Test hook: read-only projection/state for local interaction verification.
root.paperMap={getState:()=>JSON.parse(JSON.stringify(state)),getProjected:()=>projected.map(p=>({...p})),getSelected:()=>selected};
})();
</script></div>
'''
