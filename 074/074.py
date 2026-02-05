
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class SectionBayesConfig:
    # Prior quality belief per section: Beta(alpha0, beta0)
    # Example: alpha0=8,beta0=2 means prior mean 0.8 (optimistic); alpha0=2,beta0=8 means 0.2 (strict).
    # You can tune these to reflect your policy.
    alpha0: float = 4.0
    beta0: float = 2.0

    # Evidence strength per section (acts like a weight).
    # Higher means that section score influences posterior more.
    n_grammar: float = 8.0
    n_structure: float = 8.0
    n_content: float = 10.0
    n_plagiarism: float = 14.0  # usually the strictest / most important

    # Aggregation weights for final combination (log-odds weighted)
    w_grammar: float = 0.22
    w_structure: float = 0.22
    w_content: float = 0.26
    w_plagiarism: float = 0.30

    # Decision thresholds on final probability/score
    accept_threshold: float = 0.82
    revision_threshold: float = 0.62

    # Hard guardrail for plagiarism: if plagiarism score is too low, reject regardless.
    plagiarism_hard_reject_below: float = 40.0  # in 0..100


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _score_to_prob(score_0_100: float) -> float:
    """
    Convert section score [0..100] to probability (0,1).
    We avoid exactly 0 or 1 to prevent infinite log-odds.
    """
    s = _clip(float(score_0_100), 0.0, 100.0) / 100.0
    eps = 1e-6
    return _clip(s, eps, 1.0 - eps)


def _beta_posterior_mean(alpha0: float, beta0: float, p: float, n: float) -> float:
    """
    Treat p as the observed success rate with pseudo-count n.
    k = p*n successes, (n-k) failures.
    Posterior = Beta(alpha0 + k, beta0 + n - k)
    Mean = (alpha0 + k) / (alpha0 + beta0 + n)
    """
    n = max(0.0, float(n))
    k = _clip(float(p), 0.0, 1.0) * n
    a = alpha0 + k
    b = beta0 + (n - k)
    return a / (a + b) if (a + b) > 0 else 0.5


def _logit(p: float) -> float:
    p = _clip(p, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    # Numerically stable-ish sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def bayesian_final_decision(
    grammar_score: float,
    structure_score: float,
    content_score: float,
    plagiarism_score: float,
    cfg: Optional[SectionBayesConfig] = None,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - final_probability (0..1)
      - final_score_0_100
      - decision category
      - per-section posterior means
      - explanation / reason
    """
    cfg = cfg or SectionBayesConfig()

    # Hard guardrail for plagiarism
    if float(plagiarism_score) < cfg.plagiarism_hard_reject_below:
        return {
            "final_probability": 0.0,
            "final_score_0_100": 0.0,
            "decision": "REJECT",
            "reason": f"Plagiarism score {plagiarism_score:.1f} < hard reject threshold {cfg.plagiarism_hard_reject_below:.1f}",
            "posteriors": {},
            "config": asdict(cfg),
        }

    # Convert raw scores to probabilities
    p_g = _score_to_prob(grammar_score)
    p_s = _score_to_prob(structure_score)
    p_c = _score_to_prob(content_score)
    p_p = _score_to_prob(plagiarism_score)

    # Posterior mean quality for each section using Beta prior + evidence strength
    post_g = _beta_posterior_mean(cfg.alpha0, cfg.beta0, p_g, cfg.n_grammar)
    post_s = _beta_posterior_mean(cfg.alpha0, cfg.beta0, p_s, cfg.n_structure)
    post_c = _beta_posterior_mean(cfg.alpha0, cfg.beta0, p_c, cfg.n_content)
    post_p = _beta_posterior_mean(cfg.alpha0, cfg.beta0, p_p, cfg.n_plagiarism)

    # Bayesian-inspired aggregation: weighted sum of log-odds of posterior means
    # This tends to penalize very low section probabilities more sharply than averaging.
    wsum = cfg.w_grammar + cfg.w_structure + cfg.w_content + cfg.w_plagiarism
    if wsum <= 0:
        # Fallback to equal weights
        wg = ws = wc = wp = 0.25
    else:
        wg = cfg.w_grammar / wsum
        ws = cfg.w_structure / wsum
        wc = cfg.w_content / wsum
        wp = cfg.w_plagiarism / wsum

    log_odds = (
        wg * _logit(post_g) +
        ws * _logit(post_s) +
        wc * _logit(post_c) +
        wp * _logit(post_p)
    )

    final_p = _sigmoid(log_odds)
    final_score = round(final_p * 100.0, 2)

    # Decision thresholds
    if final_p >= cfg.accept_threshold:
        decision = "ACCEPT"
        reason = f"final_probability {final_p:.4f} >= accept_threshold {cfg.accept_threshold:.2f}"
    elif final_p >= cfg.revision_threshold:
        decision = "NEED_REVISION"
        reason = f"final_probability {final_p:.4f} >= revision_threshold {cfg.revision_threshold:.2f}"
    else:
        decision = "REJECT"
        reason = f"final_probability {final_p:.4f} < revision_threshold {cfg.revision_threshold:.2f}"

    return {
        "final_probability": round(final_p, 6),
        "final_score_0_100": final_score,
        "decision": decision,
        "reason": reason,
        "posteriors": {
            "grammar": round(post_g, 6),
            "structure": round(post_s, 6),
            "content": round(post_c, 6),
            "plagiarism": round(post_p, 6),
        },
        "inputs": {
            "grammar_score": grammar_score,
            "structure_score": structure_score,
            "content_score": content_score,
            "plagiarism_score": plagiarism_score,
        },
        "weights_normalized": {
            "grammar": round(wg, 6),
            "structure": round(ws, 6),
            "content": round(wc, 6),
            "plagiarism": round(wp, 6),
        },
        "evidence_strength": {
            "grammar": cfg.n_grammar,
            "structure": cfg.n_structure,
            "content": cfg.n_content,
            "plagiarism": cfg.n_plagiarism,
        },
        "config": asdict(cfg),
    }


if __name__ == "__main__":
    # Minimal demo:
    # The plagiarism section typically dominates due to higher weight/strength.
    demo = bayesian_final_decision(
        grammar_score=78,
        structure_score=72,
        content_score=70,
        plagiarism_score=88,
    )
    print(json.dumps(demo, ensure_ascii=False, indent=2))
