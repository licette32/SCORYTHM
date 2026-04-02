"""
agent/uncertainty.py
====================
Utility functions for computing and interpreting prediction uncertainty
from a calibrated probabilistic classifier.

Mathematical Background
-----------------------
Given a calibrated probability p = P(fraud | transaction), we define:

  1. Uncertainty (u):
     u = 1 - |2p - 1|

     Derivation:
       - The "confidence margin" m = |2p - 1| measures how far p is from 0.5.
         - m = 0  when p = 0.5  → maximum uncertainty (model has no preference)
         - m = 1  when p = 0 or p = 1 → maximum certainty
       - Uncertainty is the complement: u = 1 - m
       - Range: u ∈ [0, 1]

     Examples:
       p = 0.50 → u = 1.00  (completely uncertain)
       p = 0.40 → u = 0.80  (quite uncertain)
       p = 0.35 → u = 0.70  (uncertain — triggers signal purchase)
       p = 0.30 → u = 0.60  (borderline)
       p = 0.20 → u = 0.40  (fairly confident it's legitimate)
       p = 0.05 → u = 0.10  (very confident it's legitimate)
       p = 0.90 → u = 0.20  (very confident it's fraud)

  2. Uncertainty threshold:
     The agent enters "signal purchase mode" when u > threshold.
     Default threshold = 0.30, which corresponds to p ∈ (0.35, 0.65).

     This is the "uncertain zone" where the model's confidence is insufficient
     to make a reliable autonomous decision.

  3. Expected utility of a signal:
     EU(signal) = utility_score / cost_usd

     Where utility_score is a prior estimate of how much the signal reduces
     uncertainty. The agent selects the signal with the highest EU ratio.
"""

from __future__ import annotations


def calculate_uncertainty(prob: float) -> float:
    """
    Computes the uncertainty score from a calibrated fraud probability.

    Formula: u = 1 - |2p - 1|

    This is equivalent to:
      u = 1 - |p - (1-p)|   (difference between class probabilities)
      u = 2 * min(p, 1-p)   (twice the minority class probability)

    Parameters
    ----------
    prob : float
        Calibrated probability of fraud, in [0, 1].

    Returns
    -------
    float
        Uncertainty score in [0, 1].
        - 1.0 → maximum uncertainty (prob ≈ 0.5)
        - 0.0 → maximum certainty (prob ≈ 0.0 or prob ≈ 1.0)
    """
    prob = float(prob)
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"prob must be in [0, 1], got {prob}")

    # Equivalent formulations (all give the same result):
    #   1 - abs(2*prob - 1)
    #   2 * min(prob, 1 - prob)
    uncertainty = 1.0 - abs(2.0 * prob - 1.0)
    return round(uncertainty, 6)


def is_uncertain(prob: float, threshold: float = 0.30) -> bool:
    """
    Returns True if the model's prediction is uncertain enough to warrant
    purchasing additional external signals.

    The agent enters signal-purchase mode when:
        uncertainty > threshold
    which is equivalent to:
        threshold_low < prob < threshold_high
    where:
        threshold_low  = (1 - threshold) / 2
        threshold_high = (1 + threshold) / 2

    With the default threshold=0.30:
        uncertain zone: prob ∈ (0.35, 0.65)

    Parameters
    ----------
    prob : float
        Calibrated probability of fraud, in [0, 1].
    threshold : float
        Uncertainty threshold. Default 0.30.
        - Higher threshold → agent buys signals more aggressively
        - Lower threshold  → agent is more conservative (buys less)

    Returns
    -------
    bool
        True if the model is uncertain and should purchase signals.
    """
    uncertainty = calculate_uncertainty(prob)
    return uncertainty > threshold


def uncertainty_to_zone(prob: float, threshold: float = 0.30) -> str:
    """
    Classifies the prediction into a human-readable confidence zone.

    Parameters
    ----------
    prob : float
        Calibrated probability of fraud.
    threshold : float
        Uncertainty threshold for the "uncertain" zone.

    Returns
    -------
    str
        One of: "CONFIDENT_FRAUD", "UNCERTAIN", "CONFIDENT_LEGIT"
    """
    uncertainty = calculate_uncertainty(prob)

    if uncertainty > threshold:
        return "UNCERTAIN"
    elif prob >= 0.5:
        return "CONFIDENT_FRAUD"
    else:
        return "CONFIDENT_LEGIT"


def expected_utility(utility_score: float, cost_usd: float) -> float:
    """
    Computes the expected utility ratio for a signal purchase decision.

    EU = utility_score / cost_usd

    A higher ratio means the signal provides more information per dollar spent.
    The agent selects the signal with the highest EU ratio when in uncertain mode.

    Parameters
    ----------
    utility_score : float
        Prior estimate of the signal's ability to reduce uncertainty (0-1).
    cost_usd : float
        Cost of the signal in USD.

    Returns
    -------
    float
        Expected utility ratio (utility per dollar).
    """
    if cost_usd <= 0:
        raise ValueError(f"cost_usd must be positive, got {cost_usd}")
    return round(utility_score / cost_usd, 4)


# ─── Module self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Uncertainty Module — Self Test")
    print("=" * 55)

    test_probs = [0.01, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 0.99]

    print(f"\n{'prob':>6}  {'uncertainty':>12}  {'uncertain?':>10}  {'zone':>18}")
    print("─" * 55)
    for p in test_probs:
        u = calculate_uncertainty(p)
        unc = is_uncertain(p)
        zone = uncertainty_to_zone(p)
        flag = "⚠️ BUY SIGNAL" if unc else ""
        print(f"  {p:.2f}  →  u={u:.4f}    {str(unc):>5}    {zone:>18}  {flag}")

    print("\n" + "─" * 55)
    print("Expected utility examples:")
    signals = [
        ("ip-reputation",  0.72, 0.001),
        ("device-history", 0.88, 0.003),
        ("tx-velocity",    0.81, 0.002),
    ]
    for name, util, cost in signals:
        eu = expected_utility(util, cost)
        print(f"  {name:20s}  utility={util}  cost=${cost:.3f}  EU={eu:.1f}")
