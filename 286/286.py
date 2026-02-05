from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class DetectionResult:
    start_line: Optional[int]
    end_line: Optional[int]
    tags: List[str]  # "O", "B-REF", "I-REF"


class ReferenceBoundaryDetector:
    # ---- Section header patterns (Persian + English) ----
    _HEADER_PAT = re.compile(
        r"^\s*(?:"
        r"منابع|مراجع|کتابنامه|فهرست\s*منابع|فهرست\s*مراجع|"
        r"references|bibliography|works\s*cited"
        r")\s*[:：]?\s*$",
        re.IGNORECASE,
    )

    # ---- Citation-like line patterns ----
    _BRACKET_NUM = re.compile(r"^\s*\[\s*\d{1,4}\s*\]")                 # [1]
    _LEAD_NUM = re.compile(r"^\s*(?:\d{1,4}[\.\-\)]|\(\d{1,4}\))\s+")   # 1. / 1- / 1) / (1)
    _YEAR = re.compile(r"(?:19|20)\d{2}|1[3-4]\d{2}")                   # 2020 / 1399
    _DOI = re.compile(r"\bdoi\s*:\s*\S+|\b10\.\d{4,9}/\S+", re.IGNORECASE)
    _URL = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)

    # ---- "Next section" stop patterns (very rough) ----
    _STOP_PAT = re.compile(
        r"^\s*(?:"
        r"پیوست|ضمیمه|appendix|"
        r"نتیجه(?:\s*گیری)?|جمع\s*بندی|conclusion|"
        r"چکیده|abstract|"
        r"فصل\s*\d+|chapter\s*\d+"
        r")\s*[:：]?\s*$",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        # "ML-ish" weights for a tiny linear scorer on hand-crafted features.
        # These are arbitrary defaults (no training data).
        self.w = {
            "is_header": 3.5,
            "has_bracket_num": 1.6,
            "has_lead_num": 1.2,
            "has_year": 1.0,
            "has_doi": 1.8,
            "has_url": 1.3,
            "punct_density": 0.8,   # more punctuation often appears in references
            "len_norm": 0.4,        # medium-length lines tend to be reference entries
        }
        self.threshold_start = 3.0  # score threshold to consider entering reference mode
        self.threshold_in = 1.6     # score threshold to keep inside reference mode

    def _features(self, line: str) -> dict:
        s = (line or "").strip()
        if not s:
            return {
                "is_header": 0.0,
                "has_bracket_num": 0.0,
                "has_lead_num": 0.0,
                "has_year": 0.0,
                "has_doi": 0.0,
                "has_url": 0.0,
                "punct_density": 0.0,
                "len_norm": 0.0,
            }

        is_header = 1.0 if self._HEADER_PAT.match(s) else 0.0
        has_bracket_num = 1.0 if self._BRACKET_NUM.search(s) else 0.0
        has_lead_num = 1.0 if self._LEAD_NUM.search(s) else 0.0
        has_year = 1.0 if self._YEAR.search(s) else 0.0
        has_doi = 1.0 if self._DOI.search(s) else 0.0
        has_url = 1.0 if self._URL.search(s) else 0.0

        # Punctuation density (simple proxy)
        punct = sum(1 for c in s if c in ".,;:()[]{}-–—/\\")
        punct_density = min(1.0, punct / max(10, len(s)))

        # Length normalization (peaks around ~80 chars; crude bell-ish shape)
        L = len(s)
        len_norm = max(0.0, 1.0 - abs(L - 80) / 120.0)

        return {
            "is_header": is_header,
            "has_bracket_num": has_bracket_num,
            "has_lead_num": has_lead_num,
            "has_year": has_year,
            "has_doi": has_doi,
            "has_url": has_url,
            "punct_density": punct_density,
            "len_norm": len_norm,
        }

    def _score(self, feats: dict) -> float:
        return sum(self.w[k] * feats.get(k, 0.0) for k in self.w)

    def detect(self, lines: List[str]) -> DetectionResult:
        """
        Returns:
          start_line: index of first line in references section (or None)
          end_line: index of last line in references section (or None)
          tags: BIO-like tags for each line: O / B-REF / I-REF
        """
        n = len(lines)
        tags = ["O"] * n
        if n == 0:
            return DetectionResult(None, None, tags)

        # 1) Strong rule: find an explicit header first.
        header_idx = None
        for i, line in enumerate(lines):
            if self._HEADER_PAT.match((line or "").strip()):
                header_idx = i
                break

        # 2) If no header, fall back to best-scoring region.
        scores = [self._score(self._features(line)) for line in lines]

        start = None
        if header_idx is not None:
            # Start after header if possible; if header is the actual first reference line, allow it.
            start = header_idx + 1 if header_idx + 1 < n else header_idx
            # If next line is empty, skip a few empties.
            while start < n and not (lines[start] or "").strip():
                start += 1
            if start >= n:
                start = header_idx
        else:
            # Find first index where a small window average exceeds threshold.
            # This tries to avoid picking a random citation-like line in the middle.
            win = 4
            best_i = None
            best_avg = 0.0
            for i in range(0, n):
                a = sum(scores[i : min(n, i + win)]) / max(1, min(n, i + win) - i)
                if a > best_avg:
                    best_avg = a
                    best_i = i
            if best_i is not None and best_avg >= self.threshold_start:
                start = best_i

        if start is None or start < 0 or start >= n:
            return DetectionResult(None, None, tags)

        # 3) Expand to find the end: keep consuming lines while they look like references.
        end = start
        empty_run = 0
        for j in range(start, n):
            s = (lines[j] or "").strip()
            if not s:
                empty_run += 1
                # Allow up to 2 empty lines inside references.
                if empty_run > 2:
                    break
                end = j
                continue
            empty_run = 0

            # Stop on a strong next-section header (rough).
            if j > start and self._STOP_PAT.match(s):
                break

            # Keep if score is high enough OR it looks like a typical reference entry.
            keep = (scores[j] >= self.threshold_in) or bool(
                self._BRACKET_NUM.search(s)
                or self._LEAD_NUM.search(s)
                or self._DOI.search(s)
                or self._URL.search(s)
                or (self._YEAR.search(s) and ("," in s or "." in s))
            )

            if not keep and j > start + 2:
                # If we already have a few reference lines, a non-reference line likely ends the section.
                break

            end = j

        # 4) Tagging output.
        tags[start] = "B-REF"
        for k in range(start + 1, end + 1):
            tags[k] = "I-REF"

        return DetectionResult(start, end, tags)


if __name__ == "__main__":
    # Simple manual test: one line per input line from stdin until EOF.
    import sys

    text_lines = [ln.rstrip("\n") for ln in sys.stdin.readlines()]
    det = ReferenceBoundaryDetector()
    res = det.detect(text_lines)

    print(f"start_line={res.start_line} end_line={res.end_line}")
    for i, (t, ln) in enumerate(zip(res.tags, text_lines)):
        mark = ">>" if i == res.start_line else "  "
        print(f"{mark}{i:04d} [{t}] {ln}")
