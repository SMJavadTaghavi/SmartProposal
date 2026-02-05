from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List


# -----------------------------
# Rule schema
# -----------------------------

@dataclass
class ScoringRules:
    # Component weights (must sum roughly to 1; we normalize anyway).
    w_coverage: float = 0.35
    w_correctness: float = 0.40
    w_completeness: float = 0.25

    # Thresholds for final decision based on overall score [0..100].
    accept_threshold: float = 85.0
    revise_threshold: float = 60.0
    # Decision mapping:
    #   score >= accept_threshold => "ACCEPT"
    #   score >= revise_threshold => "REVISE"
    #   else => "REJECT"

    # Optional guardrails: hard-fail if too many severe issues.
    max_missing_in_ref: int = 0          # citations in text but absent in references
    max_incomplete_refs: int = 3         # number of incomplete reference entries allowed


DEFAULT_RULES_PATH = "scoring_rules.json"


# -----------------------------
# Rule store (simple JSON)
# -----------------------------

class RulesStore:
    """Simple JSON rules store for reading/updating scoring rules."""

    def __init__(self, path: str = DEFAULT_RULES_PATH) -> None:
        self.path = path

    def load(self) -> ScoringRules:
        if not os.path.exists(self.path):
            rules = ScoringRules()
            self.save(rules)
            return rules

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Be tolerant to missing keys.
        return ScoringRules(
            w_coverage=float(data.get("w_coverage", 0.35)),
            w_correctness=float(data.get("w_correctness", 0.40)),
            w_completeness=float(data.get("w_completeness", 0.25)),
            accept_threshold=float(data.get("accept_threshold", 85.0)),
            revise_threshold=float(data.get("revise_threshold", 60.0)),
            max_missing_in_ref=int(data.get("max_missing_in_ref", 0)),
            max_incomplete_refs=int(data.get("max_incomplete_refs", 3)),
        )

    def save(self, rules: ScoringRules) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(asdict(rules), f, ensure_ascii=False, indent=2)

    def update(self, **patch: Any) -> ScoringRules:
        rules = self.load()
        for k, v in patch.items():
            if not hasattr(rules, k):
                raise ValueError(f"Unknown rule field: {k}")
            setattr(rules, k, type(getattr(rules, k))(v))
        self.save(rules)
        return rules


# -----------------------------
# Scoring + decision engine
# -----------------------------

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_overall_score(
    coverage_score: float,      # 0..100
    correctness_score: float,   # 0..100
    completeness_score: float,  # 0..100
    rules: ScoringRules,
) -> float:
    """Weighted sum score with automatic normalization of weights."""
    w_sum = rules.w_coverage + rules.w_correctness + rules.w_completeness
    if w_sum <= 0:
        # Fallback to equal weights if user breaks config.
        w_cov = w_cor = w_com = 1.0 / 3.0
    else:
        w_cov = rules.w_coverage / w_sum
        w_cor = rules.w_correctness / w_sum
        w_com = rules.w_completeness / w_sum

    overall = (
        w_cov * coverage_score +
        w_cor * correctness_score +
        w_com * completeness_score
    )
    return _clip(overall, 0.0, 100.0)


def decide(
    overall_score: float,
    *,
    missing_in_ref_count: int,
    incomplete_refs_count: int,
    rules: ScoringRules,
) -> Tuple[str, str]:
    """
    Final decision with thresholds + guardrails.
    Returns (decision, reason).
    """
    # Guardrails first (hard failures).
    if missing_in_ref_count > rules.max_missing_in_ref:
        return ("REJECT", f"Too many missing citations in reference list: {missing_in_ref_count} > {rules.max_missing_in_ref}")
    if incomplete_refs_count > rules.max_incomplete_refs:
        return ("REVISE", f"Too many incomplete reference entries: {incomplete_refs_count} > {rules.max_incomplete_refs}")

    # Threshold-based decision.
    if overall_score >= rules.accept_threshold:
        return ("ACCEPT", f"Score {overall_score:.2f} >= accept_threshold {rules.accept_threshold:.2f}")
    if overall_score >= rules.revise_threshold:
        return ("REVISE", f"Score {overall_score:.2f} >= revise_threshold {rules.revise_threshold:.2f}")
    return ("REJECT", f"Score {overall_score:.2f} < revise_threshold {rules.revise_threshold:.2f}")


# -----------------------------
# Public interface (what your app calls)
# -----------------------------

def set_rules(path: str = DEFAULT_RULES_PATH, **patch: Any) -> Dict[str, Any]:
    """Update rules in storage."""
    store = RulesStore(path)
    rules = store.update(**patch)
    return {"ok": True, "rules": asdict(rules)}


def get_rules(path: str = DEFAULT_RULES_PATH) -> Dict[str, Any]:
    """Read current rules."""
    store = RulesStore(path)
    rules = store.load()
    return {"ok": True, "rules": asdict(rules)}


def evaluate_with_rules(
    *,
    coverage_score: float,
    correctness_score: float,
    completeness_score: float,
    missing_in_ref_count: int,
    incomplete_refs_count: int,
    rules_path: str = DEFAULT_RULES_PATH,
) -> Dict[str, Any]:
    """Compute overall score and final decision using stored rules."""
    store = RulesStore(rules_path)
    rules = store.load()
    overall = compute_overall_score(coverage_score, correctness_score, completeness_score, rules)
    decision, reason = decide(
        overall,
        missing_in_ref_count=missing_in_ref_count,
        incomplete_refs_count=incomplete_refs_count,
        rules=rules,
    )
    return {
        "overall_score": round(overall, 2),
        "decision": decision,
        "reason": reason,
        "rules": asdict(rules),
    }


# -----------------------------
# CLI demo to satisfy the test:
# "change value -> decision changes"
# -----------------------------

if __name__ == "__main__":
    # Demo scenario: fixed component scores and counts.
    # Then change revise_threshold to force a different decision.
    demo_scores = {
        "coverage_score": 70.0,
        "correctness_score": 70.0,
        "completeness_score": 70.0,
        "missing_in_ref_count": 0,
        "incomplete_refs_count": 0,
    }

    print("== Initial evaluation ==")
    out1 = evaluate_with_rules(**demo_scores)
    print(json.dumps(out1, ensure_ascii=False, indent=2))

    # Force decision change by raising revise_threshold above the computed overall score (~70).
    print("\n== Update revise_threshold to 75 (should change decision) ==")
    set_rules(revise_threshold=75.0)

    out2 = evaluate_with_rules(**demo_scores)
    print(json.dumps(out2, ensure_ascii=False, indent=2))

    # Print explicit test assertion style info.
    changed = out1["decision"] != out2["decision"]
    print(f"\nTEST: decision_changed={changed} (before={out1['decision']} after={out2['decision']})")
    raise SystemExit(0 if changed else 1)
