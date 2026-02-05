from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple


# -----------------------------
# Models
# -----------------------------

@dataclass
class SimilarityHit:
    source: str                 # "internal_db" | "open_wikipedia"
    target_id: str
    target_text: str
    similarity_percent: float


@dataclass
class SimilarityResult:
    query: str
    similarity_percent: float
    best_hit: Optional[SimilarityHit]
    hits: List[SimilarityHit]
    notes: Dict[str, Any]


# -----------------------------
# Text normalization + similarity
# -----------------------------

def _norm(s: str) -> str:
    # Basic normalization for Persian/English mixed text.
    s = (s or "").strip().lower()
    s = s.replace("ي", "ی").replace("ك", "ک")  # Arabic forms to Persian forms
    s = s.replace("‌", " ")                    # ZWNJ to space
    s = s.replace("،", ",").replace("؛", ";")
    s = re.sub(r"\s+", " ", s)
    return s


def _tokens(s: str) -> List[str]:
    # Keep Persian/Latin letters and digits as tokens.
    s = _norm(s)
    toks = re.findall(r"[a-z0-9]+|[\u0600-\u06ff]+", s)
    return toks


def _char_ngrams(s: str, n: int = 3) -> List[str]:
    s = _norm(s)
    s = re.sub(r"\s+", " ", s)
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def _jaccard(a: List[str], b: List[str]) -> float:
    A = set(a)
    B = set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def similarity_percent(a: str, b: str) -> float:
    """
    Combined similarity:
      - Token Jaccard (captures lexical overlap)
      - Char 3-gram Jaccard (captures near-duplicates / minor edits)
    """
    tj = _jaccard(_tokens(a), _tokens(b))
    cj = _jaccard(_char_ngrams(a, 3), _char_ngrams(b, 3))
    score = 0.55 * cj + 0.45 * tj
    return round(max(0.0, min(1.0, score)) * 100.0, 2)


# -----------------------------
# Internal DB (SQLite)
# -----------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS internal_sentences (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    created_at_unix INTEGER NOT NULL
);
"""

def init_internal_db(db_path: str = "internal_sentences.db") -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()

def add_sentence(sentence_id: str, text: str, db_path: str = "internal_sentences.db") -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO internal_sentences (id, text, created_at_unix) VALUES (?, ?, ?)",
            (sentence_id, text, int(time.time())),
        )
        conn.commit()

def fetch_internal_candidates(limit: int = 200, db_path: str = "internal_sentences.db") -> List[Tuple[str, str]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, text FROM internal_sentences ORDER BY created_at_unix DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]


# -----------------------------
# Open sources (Wikipedia OpenSearch)
# -----------------------------

def wiki_opensearch(query: str, lang: str = "fa", timeout: float = 4.0, max_titles: int = 5) -> List[Tuple[str, str]]:
    """
    Uses Wikipedia OpenSearch to retrieve page titles and snippets (best-effort).
    Returns list of (id, text) candidates.
    If request fails, returns [].
    """
    q = _norm(query)
    if not q:
        return []

    # OpenSearch: returns [query, titles[], descriptions[], links[]]
    # We take descriptions as candidate text.
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "opensearch",
        "search": q[:200],
        "limit": str(max_titles),
        "namespace": "0",
        "format": "json",
    }
    full = url + "?" + urllib.parse.urlencode(params)

    try:
        req = urllib.request.Request(full, headers={"User-Agent": "SmartProposal/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(data)
        titles = obj[1] if len(obj) > 1 else []
        descs = obj[2] if len(obj) > 2 else []
        links = obj[3] if len(obj) > 3 else []
        out: List[Tuple[str, str]] = []
        for i in range(min(len(titles), len(descs), len(links))):
            title = str(titles[i])
            desc = str(descs[i] or "")
            link = str(links[i] or "")
            # Build a compact candidate text
            cand_text = f"{title}. {desc}".strip()
            out.append((f"wiki:{title}", cand_text))
        return out
    except Exception:
        return []


# -----------------------------
# Main checker
# -----------------------------

def check_similarity(
    sentence: str,
    *,
    internal_db_path: str = "internal_sentences.db",
    internal_limit: int = 200,
    use_open_sources: bool = True,
    wiki_lang: str = "fa",
) -> SimilarityResult:
    """
    Compare a sentence with:
      - Internal DB sentences
      - Wikipedia OpenSearch candidates (open source, best-effort)
    Output:
      - overall similarity_percent = max(best internal, best open)
      - best_hit with details
    """
    query = sentence or ""
    qn = _norm(query)
    hits: List[SimilarityHit] = []
    notes: Dict[str, Any] = {"open_sources_used": False, "internal_candidates": 0, "open_candidates": 0}

    # Internal DB candidates
    init_internal_db(internal_db_path)
    internal = fetch_internal_candidates(limit=internal_limit, db_path=internal_db_path)
    notes["internal_candidates"] = len(internal)

    best: Optional[SimilarityHit] = None
    best_score = -1.0

    for sid, txt in internal:
        sc = similarity_percent(qn, txt)
        hit = SimilarityHit(source="internal_db", target_id=str(sid), target_text=txt, similarity_percent=sc)
        hits.append(hit)
        if sc > best_score:
            best_score = sc
            best = hit

    # Open sources (Wikipedia)
    if use_open_sources:
        cands = wiki_opensearch(qn, lang=wiki_lang)
        notes["open_candidates"] = len(cands)
        notes["open_sources_used"] = True if cands else False

        for cid, ctxt in cands:
            sc = similarity_percent(qn, ctxt)
            hit = SimilarityHit(source="open_wikipedia", target_id=str(cid), target_text=ctxt, similarity_percent=sc)
            hits.append(hit)
            if sc > best_score:
                best_score = sc
                best = hit

    overall = max(0.0, best_score) if best_score >= 0 else 0.0
    # Keep only top 10 hits for output readability
    hits_sorted = sorted(hits, key=lambda h: h.similarity_percent, reverse=True)[:10]

    return SimilarityResult(
        query=query,
        similarity_percent=round(overall, 2),
        best_hit=best,
        hits=hits_sorted,
        notes=notes,
    )


# -----------------------------
# CLI usage
# -----------------------------

if __name__ == "__main__":
    # Usage:
    #   python similarity_checker.py "your sentence here"
    #
    # Optional: seed internal DB quickly:
    #   python similarity_checker.py --seed
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "--seed":
        init_internal_db()
        add_sentence("s1", "سرقت ادبی به معنای استفاده از کار دیگران بدون ذکر منبع است.")
        add_sentence("s2", "References should be listed at the end of the document.")
        print("Seeded internal DB with sample sentences.")
        raise SystemExit(0)

    if len(sys.argv) < 2:
        print("Usage: python similarity_checker.py <sentence>  (or --seed)")
        raise SystemExit(2)

    sent = " ".join(sys.argv[1:])
    res = check_similarity(sent, use_open_sources=True, wiki_lang="fa")
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
