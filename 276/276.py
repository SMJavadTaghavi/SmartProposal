from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple


# -----------------------------
# Data models
# -----------------------------

@dataclass
class RefEntry:
    # key is a normalized identifier used for matching (e.g., "N:12" or "AY:smith_2020")
    key: str
    raw: str
    # Optional numeric index if it exists (useful for diagnostics)
    index: Optional[int] = None


@dataclass
class ScoreBreakdown:
    overall_score: float
    completeness_score: float
    correctness_score: float
    coverage_score: float
    penalties: List[Dict[str, Any]]
    metrics: Dict[str, Any]


# -----------------------------
# Heuristic helpers
# -----------------------------

YEAR_PAT = re.compile(r"\b(?:19|20)\d{2}\b|\b1[3-4]\d{2}\b")
DOI_PAT = re.compile(r"\bdoi\s*:\s*\S+|\b10\.\d{4,9}/\S+", re.IGNORECASE)
URL_PAT = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)


def _norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("،", ",")
    return s


def _ref_completeness_score(raw: str) -> Tuple[float, Dict[str, bool]]:
    """
    Compute a completeness score (0..1) for a single reference entry (heuristic).
    The aim is to catch incomplete references aggressively (test-friendly).
    """
    t = _norm(raw)

    flags = {
        "has_year": bool(YEAR_PAT.search(t)),
        "has_doi_or_url": bool(DOI_PAT.search(t) or URL_PAT.search(t)),
        "has_title_like": False,
        "has_journal_like": False,
        "long_enough": len(t) >= 45,
    }

    # Title-like: quoted title OR simply a long-ish chunk
    if re.search(r"[\"“][^\"”]{6,120}[\"”]", t) or len(t) >= 70:
        flags["has_title_like"] = True

    # Journal-like: volume(issue), pages range, pp.
    if re.search(r"\b\d+\s*\(\s*\d+\s*\)", t) or re.search(r"\bpp?\.\s*\d+", t, re.IGNORECASE) or re.search(r"\b\d{1,4}\s*[-–]\s*\d{1,4}\b", t):
        flags["has_journal_like"] = True

    # Weighted completeness (simple, interpretable)
    score = 0.0
    score += 0.35 if flags["has_year"] else 0.0
    score += 0.25 if flags["has_title_like"] else 0.0
    score += 0.20 if flags["has_journal_like"] else 0.0
    score += 0.20 if flags["has_doi_or_url"] else 0.0
    score = min(1.0, score)

    # If extremely short and missing year/doi/url, clamp down (strong incomplete signal)
    if len(t) < 30 and (not flags["has_year"]) and (not flags["has_doi_or_url"]):
        score = min(score, 0.20)

    return score, flags


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------
# Scoring design
# -----------------------------

def score_citation_quality(
    in_text_keys: List[str],
    reference_entries: List[RefEntry],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> ScoreBreakdown:
    """
    Scoring philosophy:
      - Overall score in [0..100]
      - Combine three components:
        1) Coverage (are in-text citations present in reference list? and vice versa)
        2) Correctness (missing links are severe)
        3) Completeness (reference entries should contain expected bibliographic fields)
    """
    w = weights or {
        "coverage": 0.35,       # how well the sets overlap
        "correctness": 0.40,    # missing/extra are major issues
        "completeness": 0.25,   # incomplete refs reduce score
    }

    # Prepare sets
    in_text_keys_norm = [k.strip() for k in in_text_keys if (k or "").strip()]
    in_text_set = set(in_text_keys_norm)

    ref_set = set(r.key for r in reference_entries if (r.key or "").strip())

    # Basic metrics
    missing_in_ref = sorted(list(in_text_set - ref_set))
    missing_in_text = sorted(list(ref_set - in_text_set))

    # Coverage is symmetric overlap measure (F1-like)
    # precision = overlap / |in_text| ; recall = overlap / |ref|
    overlap = len(in_text_set & ref_set)
    precision = overlap / len(in_text_set) if in_text_set else 1.0
    recall = overlap / len(ref_set) if ref_set else 1.0
    if precision + recall == 0:
        coverage = 0.0
    else:
        coverage = (2 * precision * recall) / (precision + recall)

    # Correctness penalizes missing links more harshly than simple coverage.
    # This makes the metric test-friendly for catching missing/incomplete citations.
    # - Missing in ref: severe (citations without bibliography)
    # - Missing in text: moderate (uncited references)
    miss_ref_rate = len(missing_in_ref) / len(in_text_set) if in_text_set else 0.0
    miss_text_rate = len(missing_in_text) / len(ref_set) if ref_set else 0.0
    correctness = 1.0 - _clip(0.85 * miss_ref_rate + 0.45 * miss_text_rate, 0.0, 1.0)

    # Completeness: average completeness of reference entries,
    # with extra penalty if incomplete refs are cited (more harmful).
    comp_scores: List[float] = []
    incomplete_refs: List[Dict[str, Any]] = []
    cited_incomplete_count = 0

    ref_by_key = {r.key: r for r in reference_entries}
    for r in reference_entries:
        s, flags = _ref_completeness_score(r.raw)
        comp_scores.append(s)
        is_incomplete = s < 0.55
        if is_incomplete:
            incomplete_refs.append(
                {
                    "ref_key": r.key,
                    "ref_index": r.index,
                    "complete_score": round(s, 3),
                    "flags": flags,
                    "raw": r.raw,
                }
            )
            if r.key in in_text_set:
                cited_incomplete_count += 1

    avg_comp = sum(comp_scores) / len(comp_scores) if comp_scores else 1.0
    cited_incomplete_rate = cited_incomplete_count / len(in_text_set) if in_text_set else 0.0
    completeness = _clip(avg_comp - 0.35 * cited_incomplete_rate, 0.0, 1.0)

    # Convert component scores to [0..100]
    coverage_score = coverage * 100.0
    correctness_score = correctness * 100.0
    completeness_score = completeness * 100.0

    overall = (
        w["coverage"] * coverage_score
        + w["correctness"] * correctness_score
        + w["completeness"] * completeness_score
    )
    overall = _clip(overall, 0.0, 100.0)

    # Build human-readable penalties for debugging and reporting
    penalties: List[Dict[str, Any]] = []
    if missing_in_ref:
        penalties.append(
            {
                "type": "missing_in_reference_list",
                "severity": "high",
                "count": len(missing_in_ref),
                "items": missing_in_ref[:50],  # cap for safety
            }
        )
    if missing_in_text:
        penalties.append(
            {
                "type": "uncited_reference_entries",
                "severity": "medium",
                "count": len(missing_in_text),
                "items": missing_in_text[:50],
            }
        )
    if incomplete_refs:
        penalties.append(
            {
                "type": "incomplete_reference_entries",
                "severity": "medium",
                "count": len(incomplete_refs),
                "cited_incomplete_count": cited_incomplete_count,
                "examples": incomplete_refs[:20],
            }
        )

    metrics = {
        "in_text_unique": len(in_text_set),
        "reference_unique": len(ref_set),
        "overlap": overlap,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "coverage_f1": round(coverage, 4),
        "missing_in_ref_count": len(missing_in_ref),
        "missing_in_text_count": len(missing_in_text),
        "avg_ref_completeness": round(avg_comp, 4),
        "cited_incomplete_rate": round(cited_incomplete_rate, 4),
    }

    return ScoreBreakdown(
        overall_score=round(overall, 2),
        completeness_score=round(completeness_score, 2),
        correctness_score=round(correctness_score, 2),
        coverage_score=round(coverage_score, 2),
        penalties=penalties,
        metrics=metrics,
    )


# -----------------------------
# Minimal CLI demo
# -----------------------------

if __name__ == "__main__":
    # Example usage (stdin is optional; kept simple).
    # You can replace these with outputs from your extraction modules.
    sample_in_text = ["N:1", "N:2", "N:5", "AY:smith_2020"]
    sample_refs = [
        RefEntry(key="N:1", raw="[1] Smith, J. (2020). A Title. Journal 3(2), 10-20. doi:10.1234/abcd"),
        RefEntry(key="N:2", raw="[2] Doe, A. 2019. Another Title. www.example.com"),
        RefEntry(key="N:3", raw="[3] Incomplete ref"),
        RefEntry(key="AY:smith_2020", raw="Smith, J. (2020). A Title. Journal 3(2), 10-20."),
    ]

    report = score_citation_quality(sample_in_text, sample_refs)
    import json
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
