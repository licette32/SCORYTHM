"""
agent/agent.py
==============
FraudSignal Agent — Core decision-making logic.

The agent evaluates a financial transaction by:
  1. Running the calibrated XGBoost model to get prob_fraud + uncertainty
  2. If uncertainty > threshold → enters "signal purchase mode"
  3. Selects the best signal by expected utility (utility_score / cost_usd)
  4. Purchases up to MAX_SIGNALS signals sequentially via x402 micropayments
  5. After each purchase, re-evaluates with the new information
  6. Returns a comprehensive result with full reasoning trace

Decision logic:
  - prob_fraud >= 0.65  → FRAUD
  - prob_fraud <= 0.35  → LEGITIMATE
  - 0.35 < prob_fraud < 0.65 → UNCERTAIN (triggers signal purchase)
  - After purchasing signals, thresholds tighten to 0.55/0.45
"""

from __future__ import annotations

import sys
import os
import time
from typing import Any

# ─── Path setup for running as script or imported from api/ ──────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.predict import predict as model_predict
from agent.uncertainty import (
    calculate_uncertainty,
    is_uncertain,
    expected_utility,
    uncertainty_to_zone,
)
from agent.x402_client import fetch_signal, SIGNAL_CATALOG

# ─── Agent configuration ─────────────────────────────────────────────────────
UNCERTAINTY_THRESHOLD = 0.30   # triggers signal purchase mode
MAX_SIGNALS = 2                # maximum signals to purchase per evaluation
FRAUD_THRESHOLD_INITIAL = 0.65 # prob_fraud >= this → FRAUD (before signals)
LEGIT_THRESHOLD_INITIAL = 0.35 # prob_fraud <= this → LEGITIMATE (before signals)
FRAUD_THRESHOLD_FINAL = 0.55   # tighter threshold after purchasing signals
LEGIT_THRESHOLD_FINAL = 0.45   # tighter threshold after purchasing signals


def decide_signal(
    available_signals: list[str],
    already_purchased: list[str],
) -> str | None:
    """
    Selects the best signal to purchase next based on expected utility ratio.

    Strategy: maximize EU = utility_score / cost_usd
    This ensures the agent gets the most information per dollar spent.

    Parameters
    ----------
    available_signals : list[str]
        All signal names in the catalog.
    already_purchased : list[str]
        Signals already purchased in this evaluation (skip these).

    Returns
    -------
    str | None
        The name of the best signal to purchase, or None if all purchased.
    """
    candidates = [
        s for s in available_signals
        if s not in already_purchased
    ]

    if not candidates:
        return None

    # Rank by expected utility ratio (descending)
    ranked = sorted(
        candidates,
        key=lambda s: expected_utility(
            SIGNAL_CATALOG[s]["utility_score"],
            SIGNAL_CATALOG[s]["cost_usd"],
        ),
        reverse=True,
    )

    return ranked[0]


def _apply_signal_adjustment(
    base_prob: float,
    signal_result: dict,
) -> float:
    """
    Adjusts the fraud probability based on signal data.

    Uses the fraud_probability_adjustment field from the signal response,
    which encodes the signal's directional impact on fraud probability.

    The adjustment is applied as a weighted blend:
      new_prob = clip(base_prob + adjustment * weight, 0, 1)

    Parameters
    ----------
    base_prob : float
        Current fraud probability from the model.
    signal_result : dict
        Result from fetch_signal(), containing signal data.

    Returns
    -------
    float
        Adjusted fraud probability, clipped to [0, 1].
    """
    data = signal_result.get("data") or {}
    adjustment = float(data.get("fraud_probability_adjustment", 0.0))

    # Weight the adjustment by the signal's utility score
    signal_name = signal_result.get("signal_name", "")
    utility = SIGNAL_CATALOG.get(signal_name, {}).get("utility_score", 0.5)

    # Blend: new_prob = base_prob + adjustment * utility_weight
    # The adjustment is already directional (positive = more fraud risk)
    new_prob = base_prob + (adjustment * utility * 0.5)
    return float(max(0.0, min(1.0, new_prob)))


def _make_decision(
    prob_fraud: float,
    signals_purchased: list[dict],
) -> str:
    """
    Makes the final FRAUD/LEGITIMATE/UNCERTAIN decision.

    Uses tighter thresholds after purchasing signals (we've spent money,
    so we should be more decisive).
    """
    if signals_purchased:
        fraud_threshold = FRAUD_THRESHOLD_FINAL
        legit_threshold = LEGIT_THRESHOLD_FINAL
    else:
        fraud_threshold = FRAUD_THRESHOLD_INITIAL
        legit_threshold = LEGIT_THRESHOLD_INITIAL

    if prob_fraud >= fraud_threshold:
        return "FRAUD"
    elif prob_fraud <= legit_threshold:
        return "LEGITIMATE"
    else:
        return "UNCERTAIN"


def _build_reasoning(
    initial_prob: float,
    initial_uncertainty: float,
    signals_purchased: list[dict],
    final_prob: float,
    final_decision: str,
    elapsed_ms: float,
) -> str:
    """
    Builds a human-readable reasoning trace of the agent's decision process.
    """
    lines = []
    lines.append(
        f"Initial model prediction: prob_fraud={initial_prob:.4f}, "
        f"uncertainty={initial_uncertainty:.4f}"
    )

    if not signals_purchased:
        zone = uncertainty_to_zone(initial_prob)
        lines.append(
            f"Uncertainty ({initial_uncertainty:.4f}) ≤ threshold (0.30) — "
            f"model is confident. Zone: {zone}. No signals purchased."
        )
    else:
        lines.append(
            f"Uncertainty ({initial_uncertainty:.4f}) > threshold (0.30) — "
            f"entering signal purchase mode."
        )
        for i, sig in enumerate(signals_purchased, 1):
            adj = sig.get("prob_adjustment", 0.0)
            direction = "↑" if adj > 0 else ("↓" if adj < 0 else "→")
            lines.append(
                f"  Signal {i}: {sig['signal_name']} "
                f"(cost=${sig['cost_usd']:.3f}, "
                f"tx={sig['simulated_tx_hash'][:12]}...) "
                f"→ prob adjustment: {direction}{abs(adj):.4f}"
            )
        lines.append(
            f"Final probability after signals: {final_prob:.4f} "
            f"(Δ={final_prob - initial_prob:+.4f})"
        )

    lines.append(
        f"Decision: {final_decision} "
        f"(thresholds: fraud≥{FRAUD_THRESHOLD_FINAL if signals_purchased else FRAUD_THRESHOLD_INITIAL}, "
        f"legit≤{LEGIT_THRESHOLD_FINAL if signals_purchased else LEGIT_THRESHOLD_INITIAL})"
    )
    lines.append(f"Total evaluation time: {elapsed_ms:.1f}ms")

    return " | ".join(lines)


def evaluate_transaction(transaction: dict) -> dict:
    """
    Main agent entry point. Evaluates a financial transaction for fraud.

    Parameters
    ----------
    transaction : dict
        Must contain the 6 base features:
          - amount              (float)
          - hour                (int, 0-23)
          - country_mismatch    (int, 0 or 1)
          - new_account         (int, 0 or 1)
          - device_age_days     (float)
          - transactions_last_24h (int)

    Returns
    -------
    dict with:
        prob_fraud        : float — final fraud probability
        uncertainty       : float — final uncertainty score
        decision          : str   — "FRAUD", "LEGITIMATE", or "UNCERTAIN"
        signals_purchased : list  — list of purchased signal dicts
        total_cost        : float — total USD spent on signals
        reasoning         : str   — human-readable decision trace
        initial_prob_fraud: float — model's initial prediction (before signals)
        initial_uncertainty: float
        elapsed_ms        : float — total evaluation time
    """
    start_time = time.monotonic()

    # ── Step 1: Initial model prediction ─────────────────────────────────────
    prediction = model_predict(transaction)
    initial_prob = prediction["prob_fraud"]
    initial_uncertainty = prediction["uncertainty"]

    current_prob = initial_prob
    signals_purchased: list[dict] = []
    total_cost = 0.0
    available_signals = list(SIGNAL_CATALOG.keys())

    # ── Step 2: Check if uncertain → enter signal purchase loop ──────────────
    if is_uncertain(current_prob, threshold=UNCERTAINTY_THRESHOLD):
        purchased_names: list[str] = []

        for _round in range(MAX_SIGNALS):
            # Select the best signal by EU ratio
            best_signal = decide_signal(available_signals, purchased_names)
            if best_signal is None:
                break

            # Purchase the signal via x402 micropayment
            signal_result = fetch_signal(best_signal)
            cost = signal_result["cost_usd"]
            total_cost += cost

            # Apply the signal's probability adjustment
            prev_prob = current_prob
            current_prob = _apply_signal_adjustment(current_prob, signal_result)
            prob_adjustment = current_prob - prev_prob

            # Record the purchase
            purchased_names.append(best_signal)
            signals_purchased.append(
                {
                    "signal_name": best_signal,
                    "cost_usd": cost,
                    "data": signal_result.get("data"),
                    "simulated_tx_hash": signal_result["simulated_tx_hash"],
                    "latency_ms": signal_result.get("latency_ms", 0.0),
                    "prob_adjustment": round(prob_adjustment, 6),
                    "error": signal_result.get("error"),
                    "offline_simulation": signal_result.get("offline_simulation", False),
                }
            )

            # Re-evaluate uncertainty after this signal
            new_uncertainty = calculate_uncertainty(current_prob)

            # Early exit if we're now confident enough
            if not is_uncertain(current_prob, threshold=UNCERTAINTY_THRESHOLD):
                break

    # ── Step 3: Final decision ────────────────────────────────────────────────
    final_uncertainty = calculate_uncertainty(current_prob)
    decision = _make_decision(current_prob, signals_purchased)

    elapsed_ms = (time.monotonic() - start_time) * 1000

    reasoning = _build_reasoning(
        initial_prob=initial_prob,
        initial_uncertainty=initial_uncertainty,
        signals_purchased=signals_purchased,
        final_prob=current_prob,
        final_decision=decision,
        elapsed_ms=elapsed_ms,
    )

    return {
        "prob_fraud": round(current_prob, 6),
        "uncertainty": round(final_uncertainty, 6),
        "decision": decision,
        "signals_purchased": signals_purchased,
        "total_cost": round(total_cost, 6),
        "reasoning": reasoning,
        "initial_prob_fraud": round(initial_prob, 6),
        "initial_uncertainty": round(initial_uncertainty, 6),
        "elapsed_ms": round(elapsed_ms, 2),
    }


# ─── CLI demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    demo_transactions = [
        {
            "name": "🚨 High-risk fraud attempt",
            "tx": {
                "amount": 2500.0,
                "hour": 3,
                "country_mismatch": 1,
                "new_account": 1,
                "device_age_days": 1.0,
                "transactions_last_24h": 28,
            },
        },
        {
            "name": "✅ Normal purchase",
            "tx": {
                "amount": 35.0,
                "hour": 14,
                "country_mismatch": 0,
                "new_account": 0,
                "device_age_days": 400.0,
                "transactions_last_24h": 2,
            },
        },
        {
            "name": "⚠️  Ambiguous — triggers signal purchase",
            "tx": {
                "amount": 450.0,
                "hour": 21,
                "country_mismatch": 1,
                "new_account": 0,
                "device_age_days": 30.0,
                "transactions_last_24h": 9,
            },
        },
    ]

    print("\n" + "=" * 70)
    print("FraudSignal Agent — Demo Evaluation")
    print("=" * 70)

    for case in demo_transactions:
        print(f"\n{'─' * 70}")
        print(f"Transaction: {case['name']}")
        print(f"Input: {json.dumps(case['tx'])}")
        print()

        result = evaluate_transaction(case["tx"])

        print(f"  Decision          : {result['decision']}")
        print(f"  prob_fraud        : {result['prob_fraud']:.4f}  (initial: {result['initial_prob_fraud']:.4f})")
        print(f"  uncertainty       : {result['uncertainty']:.4f}  (initial: {result['initial_uncertainty']:.4f})")
        print(f"  Signals purchased : {len(result['signals_purchased'])}")
        print(f"  Total cost        : ${result['total_cost']:.4f}")
        print(f"  Elapsed           : {result['elapsed_ms']:.1f}ms")

        if result["signals_purchased"]:
            print(f"\n  Signals:")
            for sig in result["signals_purchased"]:
                print(f"    • {sig['signal_name']:20s} cost=${sig['cost_usd']:.3f}  "
                      f"tx={sig['simulated_tx_hash'][:16]}...  "
                      f"adj={sig['prob_adjustment']:+.4f}")

        print(f"\n  Reasoning: {result['reasoning']}")
