from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple


# -----------------------------
# Data models
# -----------------------------

@dataclass
class RefItem:
    index: Optional[int]          # numeric index if available (e.g., [12] or "12.")
    raw: str                      # raw reference line
    key: str                      # normalized key used for matching
    complete_score: float         # heuristic completeness score [0..1]
    complete_flags: Dict[str, bool]


@dataclass
class InTextCitation:
    kind: str                     # "numeric" | "author_year"
    raw: str
    keys: List[str]               # one or more normalized keys (e.g., for [1,2,3])
    line_idx: int


@dataclass
class ConsistencyReport:
    in_text_count: int
    ref_list_count: int
    missing_in_ref: List[Dict[str, Any]]
    missing_in_text: List[Dict[str, Any]]
    incomplete_refs: List[Dict[str, Any]]
    metrics: Dict[str, Any]


# -----------------------------
# Heuristic parsers
# -----------------------------

HEADER_PAT = re.compile(
    r"^\s*(?:منابع|مراجع|کتابنامه|فهرست\s*منابع|فهرست\s*مراجع|references|bibliography|works\s*cited)\s*[:：]?\s*$",
    re.IGNORECASE,
)

STOP_PAT = re.compile(
    r"^\s*(?:پیوست|ضمیمه|appendix|نتیجه(?:\s*گیری)?|جمع\s*بندی|conclusion|چکیده|abstract|فصل\s*\d+|chapter\s*\d+)\s*[:：]?\s*$",
    re.IGNORECASE,
)

# In-text numeric citation patterns: [1], [1-3], [1,2,5], (1) usually ambiguous; we focus on brackets for precision.
BRACKET_NUM_BLOCK = re.compile(r"\[(?:\s*\d{1,4}\s*(?:[-–]\s*\d{1,4}\s*)?)(?:\s*,\s*\d{1,4}\s*(?:[-–]\s*\d{1,4}\s*)?)*\]")
# Reference list numeric lead: [12] or "12." or "12)" or "12-"
REF_LEAD_NUM = re.compile(r"^\s*(?:\[\s*(\d{1,4})\s*\]|(\d{1,4})\s*[\.\)\-])\s*")

# In-text author-year: (Smith, 2020) / (Smith & Wesson, 2021) / (اسمیت، ۱۳۹۹)
# This is intentionally permissive to catch incomplete cases.
AUTHOR_YEAR_PAREN = re.compile(
    r"\(([^()]{0,80}?)\b(?:19|20)\d{2}\b[^()]{0,40}?\)|"
    r"\(([^()]{0,80}?)\b1[3-4]\d{2}\b[^()]{0,40}?\)",
    re.UNICODE,
)

YEAR_PAT = re.compile(r"\b(?:19|20)\d{2}\b|\b1[3-4]\d{2}\b")
DOI_PAT = re.compile(r"\bdoi\s*:\s*\S+|\b10\.\d{4,9}/\S+", re.IGNORECASE)
URL_PAT = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _normalize_key(s: str) -> str:
    """
    Create a stable matching key.
    For numeric references: "N:12"
    For author-year: "AY:smith_2020" (best-effort)
    """
    s = _normalize_spaces(s).lower()
    s = s.replace("،", ",")
    s = re.sub(r"[“”\"'`]", "", s)
    s = re.sub(r"[\u200c\u200f\u202a-\u202e]", "", s)  # remove ZWNJ/RTL marks etc.
    s = re.sub(r"[^a-z0-9\u0600-\u06ff,\-\s]", " ", s)  # keep Latin/Persian letters, digits
    s = _normalize_spaces(s)
    return s


def _expand_numeric_block(block: str) -> List[int]:
    """
    Convert "[1,2,5-7]" into [1,2,5,6,7]
    """
    nums: List[int] = []
    inner = block.strip()[1:-1]  # remove [ ]
    inner = inner.replace("–", "-")
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = [x.strip() for x in p.split("-", 1)]
            if a.isdigit() and b.isdigit():
                ia, ib = int(a), int(b)
                if ia <= ib and (ib - ia) <= 200:  # guard
                    nums.extend(list(range(ia, ib + 1)))
                else:
                    nums.append(ia)
                    nums.append(ib)
        else:
            if p.isdigit():
                nums.append(int(p))
    # unique, stable
    out: List[int] = []
    seen = set()
    for x in nums:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _extract_author_year_keys(text: str) -> List[str]:
    """
    Extract best-effort author-year keys from an in-text citation snippet.
    Strategy:
      - Find a year
      - Take up to first 1-2 tokens before year as "author-ish"
    Output keys like: "AY:smith_2020"
    """
    t = _normalize_spaces(text).replace("،", ",")
    years = YEAR_PAT.findall(t)
    if not years:
        return []

    # Keep the first year (common case). For multiple years, we create keys per year.
    keys: List[str] = []
    for y in years[:3]:
        # Split around the year
        idx = t.find(y)
        left = t[:idx].strip(" ,;")
        # Take last 1-2 "words" from left as author token(s)
        tokens = re.split(r"[\s,;]+", left)
        tokens = [tok for tok in tokens if tok]
        author_bits = tokens[-2:] if tokens else ["unknown"]
        author = "_".join(author_bits)
        author = _normalize_key(author).replace(" ", "_")
        keys.append(f"AY:{author}_{y}")
    return keys


def detect_reference_span(lines: List[str]) -> Tuple[Optional[int], Optional[int]]:
    """
    Minimal reference span detection:
    - Prefer explicit header match.
    - Otherwise: return (None, None) and the caller can treat whole document as text (less accurate).
    """
    n = len(lines)
    header_idx = None
    for i, ln in enumerate(lines):
        if HEADER_PAT.match(_normalize_spaces(ln)):
            header_idx = i
            break

    if header_idx is None:
        return None, None

    start = header_idx + 1
    while start < n and not _normalize_spaces(lines[start]):
        start += 1
    if start >= n:
        return header_idx, header_idx

    end = start
    empty_run = 0
    for j in range(start, n):
        s = _normalize_spaces(lines[j])
        if not s:
            empty_run += 1
            if empty_run > 2:
                break
            end = j
            continue
        empty_run = 0
        if STOP_PAT.match(s) and j > start:
            break
        end = j

    return start, end


def parse_reference_list(ref_lines: List[str]) -> List[RefItem]:
    """
    Parse reference list lines into RefItem objects.
    For simplicity, each line is considered one entry (common in extracted text).
    If your extractor merges/wraps entries, you can later improve by joining continuation lines.
    """
    items: List[RefItem] = []
    for ln in ref_lines:
        raw = _normalize_spaces(ln)
        if not raw:
            continue

        idx = None
        m = REF_LEAD_NUM.match(raw)
        if m:
            idx_str = m.group(1) or m.group(2)
            if idx_str and idx_str.isdigit():
                idx = int(idx_str)

        key = ""
        if idx is not None:
            key = f"N:{idx}"
        else:
            # Build an author-year-ish key from the line itself (best-effort)
            keys = _extract_author_year_keys(raw)
            key = keys[0] if keys else f"TXT:{_normalize_key(raw)[:48]}"

        # Completeness heuristics: we want to detect incomplete refs.
        flags = {
            "has_year": bool(YEAR_PAT.search(raw)),
            "has_doi_or_url": bool(DOI_PAT.search(raw) or URL_PAT.search(raw)),
            "has_title_like": False,      # quoted title or long phrase
            "has_journal_like": False,    # presence of volume/issue/pages-ish tokens
        }

        # Title-ish: quoted or long segment with punctuation
        if re.search(r"[\"“][^\"”]{6,120}[\"”]", raw) or len(raw) >= 70:
            flags["has_title_like"] = True

        # Journal-ish: volume(issue):pages or pp. or pages-like range
        if re.search(r"\b\d+\s*\(\s*\d+\s*\)", raw) or re.search(r"\bpp?\.\s*\d+", raw, re.IGNORECASE) or re.search(r"\b\d{1,4}\s*[-–]\s*\d{1,4}\b", raw):
            flags["has_journal_like"] = True

        # Compute a completeness score (0..1). No training, just weights.
        score = 0.0
        score += 0.35 if flags["has_year"] else 0.0
        score += 0.25 if flags["has_title_like"] else 0.0
        score += 0.20 if flags["has_journal_like"] else 0.0
        score += 0.20 if flags["has_doi_or_url"] else 0.0
        score = min(1.0, score)

        items.append(RefItem(index=idx, raw=raw, key=key, complete_score=score, complete_flags=flags))

    return items


def parse_in_text_citations(text_lines: List[str], offset_line_idx: int = 0) -> List[InTextCitation]:
    """
    Extract in-text citations from lines.
    We handle:
      - Numeric: [1], [1,2,5-7]
      - Author-year: (...) containing a year
    """
    cits: List[InTextCitation] = []
    for i, ln in enumerate(text_lines):
        raw_line = ln or ""

        # Numeric blocks
        for m in BRACKET_NUM_BLOCK.finditer(raw_line):
            block = m.group(0)
            nums = _expand_numeric_block(block)
            keys = [f"N:{n}" for n in nums]
            cits.append(InTextCitation(kind="numeric", raw=block, keys=keys, line_idx=offset_line_idx + i))

        # Author-year parenthetical
        for m in AUTHOR_YEAR_PAREN.finditer(raw_line):
            snippet = m.group(0)
            keys = _extract_author_year_keys(snippet)
            if keys:
                cits.append(InTextCitation(kind="author_year", raw=snippet, keys=keys, line_idx=offset_line_idx + i))

    return cits


# -----------------------------
# Core comparison logic
# -----------------------------

def compare_citations(lines: List[str]) -> ConsistencyReport:
    """
    Main function:
      1) Detect reference section span (if possible).
      2) Parse in-text citations from the body.
      3) Parse reference entries from the reference section.
      4) Compare and produce a report.
    """
    n = len(lines)
    ref_start, ref_end = detect_reference_span(lines)

    if ref_start is None or ref_end is None:
        # No explicit reference section found: treat whole doc as body, empty ref list.
        body_lines = lines
        ref_lines: List[str] = []
        ref_start = None
        ref_end = None
    else:
        body_lines = lines[:ref_start]
        ref_lines = lines[ref_start : ref_end + 1]

    in_text = parse_in_text_citations(body_lines, offset_line_idx=0)
    refs = parse_reference_list(ref_lines)

    # Index references by key and also by numeric index.
    ref_keys = set(r.key for r in refs)
    ref_num_set = set(r.index for r in refs if r.index is not None)

    # Collect all citation keys that appear in text.
    cited_keys: List[str] = []
    cited_num_set = set()
    for c in in_text:
        cited_keys.extend(c.keys)
        if c.kind == "numeric":
            for k in c.keys:
                try:
                    cited_num_set.add(int(k.split(":", 1)[1]))
                except Exception:
                    pass

    cited_keys_set = set(cited_keys)

    # Missing in reference list: citations that have no matching ref item.
    missing_in_ref: List[Dict[str, Any]] = []
    for c in in_text:
        # If any of the keys for this citation are absent, mark them.
        # This is intentionally aggressive to catch incomplete cases.
        missing_keys = [k for k in c.keys if k not in ref_keys]
        if missing_keys:
            missing_in_ref.append(
                {
                    "line_idx": c.line_idx,
                    "citation_raw": c.raw,
                    "citation_kind": c.kind,
                    "missing_keys": missing_keys,
                }
            )

    # Missing in text: reference items that never appear in in-text citations.
    missing_in_text: List[Dict[str, Any]] = []
    for r in refs:
        if r.key.startswith("N:"):
            # Numeric refs: match by index key.
            if r.key not in cited_keys_set:
                missing_in_text.append({"ref_key": r.key, "ref_raw": r.raw})
        else:
            # Author-year refs: match by key if extracted.
            if r.key.startswith("AY:") and r.key not in cited_keys_set:
                missing_in_text.append({"ref_key": r.key, "ref_raw": r.raw})

    # Incomplete references: low completeness score OR missing year AND missing doi/url AND too short.
    incomplete_refs: List[Dict[str, Any]] = []
    for r in refs:
        too_short = len(r.raw) < 35
        looks_incomplete = (r.complete_score < 0.55) or (too_short and (not r.complete_flags["has_year"]) and (not r.complete_flags["has_doi_or_url"]))
        if looks_incomplete:
            incomplete_refs.append(
                {
                    "ref_key": r.key,
                    "ref_index": r.index,
                    "ref_raw": r.raw,
                    "complete_score": r.complete_score,
                    "flags": r.complete_flags,
                }
            )

    # Metrics for test-like expectations.
    # Since we have no ground truth, we report coverage-oriented metrics:
    # - numeric_ref_coverage: fraction of cited numeric indices that exist in ref list
    # - ref_in_text_coverage: fraction of numeric refs that are cited
    # - incomplete_ratio: how many refs look incomplete
    cited_numeric_total = len(cited_num_set) if cited_num_set else 0
    cited_numeric_found = len([x for x in cited_num_set if x in ref_num_set]) if cited_num_set else 0
    numeric_ref_coverage = (cited_numeric_found / cited_numeric_total) if cited_numeric_total else None

    numeric_refs_total = len(ref_num_set)
    numeric_refs_cited = len([x for x in ref_num_set if x in cited_num_set]) if ref_num_set else 0
    ref_in_text_coverage = (numeric_refs_cited / numeric_refs_total) if numeric_refs_total else None

    incomplete_ratio = (len(incomplete_refs) / len(refs)) if refs else None

    metrics = {
        "has_reference_section": ref_start is not None,
        "reference_span": {"start": ref_start, "end": ref_end} if ref_start is not None else None,
        "numeric_ref_coverage": numeric_ref_coverage,
        "ref_in_text_coverage": ref_in_text_coverage,
        "incomplete_ratio": incomplete_ratio,
        "missing_in_ref_count": len(missing_in_ref),
        "missing_in_text_count": len(missing_in_text),
        "incomplete_refs_count": len(incomplete_refs),
    }

    return ConsistencyReport(
        in_text_count=len(in_text),
        ref_list_count=len(refs),
        missing_in_ref=missing_in_ref,
        missing_in_text=missing_in_text,
        incomplete_refs=incomplete_refs,
        metrics=metrics,
    )


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    # Usage:
    #   python citation_reference_consistency.py < input.txt
    # Where input.txt has one extracted line per row.
    import sys
    import json

    doc_lines = [ln.rstrip("\n") for ln in sys.stdin.readlines()]
    report = compare_citations(doc_lines)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
