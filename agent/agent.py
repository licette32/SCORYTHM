"""
agent/agent.py
=============
FraudSignal Agent — Core decision-making logic.

The agent evaluates a financial transaction by:
  1. Running the calibrated XGBoost model to get prob_fraud + uncertainty + CI
  2. If in ambiguous zone (CI contains 0.5) → enters VOI mode
  3. Selects signals using VOI * Thompson Sampling priority
  4. Purchases up to MAX_SIGNALS signals sequentially via x402 micropayments
  5. Updates bandit with rewards (escaped uncertain zone = 1, else = 0)
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
from agent.bandit import ThompsonBandit

# ─── Agent configuration ─────────────────────────────────────────────────────
UNCERTAINTY_THRESHOLD = 0.30   # triggers signal purchase mode
MAX_SIGNALS = 2                # maximum signals to purchase per evaluation
FRAUD_THRESHOLD_INITIAL = 0.65 # prob_fraud >= this → FRAUD (before signals)
LEGIT_THRESHOLD_INITIAL = 0.35 # prob_fraud <= this → LEGITIMATE (before signals)
FRAUD_THRESHOLD_FINAL = 0.55   # tighter threshold after purchasing signals
LEGIT_THRESHOLD_FINAL = 0.45   # tighter threshold after purchasing signals

# ─── Bandit state path ──────────────────────────────────────────────────────
_BANDIT_STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "bandit_state.json",
)

# ─── Global bandit instance ─────────────────────────────────────────────────
_bandit: ThompsonBandit | None = None


def _get_bandit() -> ThompsonBandit:
    """Returns the global bandit instance, loading state if available."""
    global _bandit
    if _bandit is None:
        _bandit = ThompsonBandit()
        _bandit.load_state(_BANDIT_STATE_PATH)
    return _bandit


def _save_bandit() -> None:
    """Saves the bandit state to disk."""
    bandit = _get_bandit()
    bandit.save_state(_BANDIT_STATE_PATH)

# ─── Cost parameters for VOI calculation ─────────────────────────────────────
C_FN = 1.0   # Cost of False Negative: letting fraud through (high cost)
C_FP = 0.1   # Cost of False Positive: blocking legitimate (low cost)


def compute_voi(
    prob_fraud: float,
    utility_score: float,
    signal_cost: float,
) -> float:
    """
    Computes the Value of Information (VOI) for a potential signal purchase.

    VOI = Expected loss reduction from buying the signal - cost of signal

    Mathematical formulation:
    ─────────────────────────
    1. Expected loss with current probability (asymmetric costs):
       L_before = p * C_FN + (1 - p) * C_FP

       donde:
       - p = prob_fraud
       - C_FN = cost of False Negative (fraud goes through)
       - C_FP = cost of False Positive (legitimate blocked)

    2. Uncertainty reduction via signal:
       El utility_score indica cuánto reduce la incertidumbre (0-1).
       Asumimos reducción de varianza proporcional a utility_score.

       Loss simetrico: L = p * (1 - p)  (asume C_FN = C_FP = 1)
       Este representa la "varianza" de la decisión.

       Expected loss after signal:
       L_after = L_before - improvement + correction

       donde:
       - improvement = (C_FN - C_FP) * uncertainty * utility_score
         (reducción proporcional a incertidumbre actual y utilidad)
       - correction = 0.5 * (C_FN - C_FP) * uncertainty^2 * utility_score^2
         (penalización por incertidumbre residual)

    3. VOI = L_before - L_after - signal_cost

    Esta fórmula da VOI > 0 cuando:
    - La incertidumbre es moderada (no muy alta ni muy baja)
    - El signal tiene alta utilidad
    - El costo es bajo

    Parameters
    ----------
    prob_fraud : float
        Current probability that the transaction is fraud.
    utility_score : float
        Expected reduction in uncertainty from the signal (0-1).
    signal_cost : float
        Cost of purchasing this signal in USD.

    Returns
    -------
    float
        VOI value. Positive = signal is worth buying.
    """
    p = float(prob_fraud)

    # Expected loss with asymmetric costs
    loss_before = p * C_FN + (1 - p) * C_FP

    # Uncertainty: u = 1 - |2p - 1|, range [0, 1]
    # u = 0  → model is certain (p ≈ 0 or 1)
    # u = 1  → model is maximally uncertain (p = 0.5)
    uncertainty = 1.0 - abs(2.0 * p - 1.0)

    # Improvement from signal:
    # La señal reduce la pérdida esperada proporcional a:
    # 1. Cuán inciertos estamos (uncertainty)
    # 2. Qué tan informativa es la señal (utility_score)
    # 3. La asimetría de costos (C_FN - C_FP)
    improvement = (C_FN - C_FP) * uncertainty * utility_score

    # Correction for remaining uncertainty:
    # incluso con señal, queda incertidumbre residual que puede
    # mover la probabilidad en dirección no deseada
    correction = 0.5 * (C_FN - C_FP) * (uncertainty ** 2) * (utility_score ** 2)

    # Expected loss after signal
    loss_after = loss_before - improvement + correction

    # VOI = loss reduction - signal cost
    voi = loss_before - loss_after - signal_cost

    return round(voi, 6)


def compute_all_voi_scores(
    prob_fraud: float,
    available_signals: list[str],
) -> dict[str, float]:
    """
    Computes VOI for all available signals.

    Parameters
    ----------
    prob_fraud : float
        Current fraud probability.
    available_signals : list[str]
        List of signal names to evaluate.

    Returns
    -------
    dict[str, float]
        Mapping of signal_name -> VOI value.
    """
    voi_scores = {}
    for signal_name in available_signals:
        catalog_entry = SIGNAL_CATALOG[signal_name]
        voi_scores[signal_name] = compute_voi(
            prob_fraud,
            catalog_entry["utility_score"],
            catalog_entry["cost_usd"],
        )
    return voi_scores


def decide_signal(
    prob_fraud: float,
    available_signals: list[str],
    already_purchased: list[str],
    bandit: ThompsonBandit | None = None,
) -> str | None:
    """
    Selects the best signal to purchase next based on VOI * bandit priority.

    Strategy:
    1. Compute VOI for all candidates
    2. Multiply each VOI by bandit sample (Thompson Sampling)
    3. Select the signal with highest adjusted VOI
    4. Only buy if adjusted VOI > 0

    The bandit learns which signals are most useful over time.

    Parameters
    ----------
    prob_fraud : float
        Current fraud probability.
    available_signals : list[str]
        All signal names in the catalog.
    already_purchased : list[str]
        Signals already purchased in this evaluation (skip these).
    bandit : ThompsonBandit | None
        Thompson Sampling bandit for adaptive signal selection.

    Returns
    -------
    str | None
        The name of the best signal to purchase, or None if no signal has positive adjusted VOI.
    """
    candidates = [
        s for s in available_signals
        if s not in already_purchased
    ]

    if not candidates:
        return None

    voi_scores = compute_all_voi_scores(prob_fraud, candidates)

    if bandit is None:
        adjusted_scores = voi_scores
    else:
        adjusted_scores = {}
        for name, voi in voi_scores.items():
            priority = bandit.get_priority(name)
            adjusted_scores[name] = voi * priority

    positive_adjusted = {k: v for k, v in adjusted_scores.items() if v > 0}

    if not positive_adjusted:
        return None

    ranked = sorted(positive_adjusted.items(), key=lambda x: x[1], reverse=True)

    return ranked[0][0]


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
    initial_conf_low: float,
    initial_conf_high: float,
    risk_zone: str,
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
        f"uncertainty={initial_uncertainty:.4f}, "
        f"CI=[{initial_conf_low:.4f}, {initial_conf_high:.4f}]"
    )

    lines.append(f"Risk zone: {risk_zone}")

    if not signals_purchased:
        zone = uncertainty_to_zone(initial_prob)
        lines.append(
            f"Zone ({zone}) — no VOI signals purchased."
        )
    else:
        lines.append("VOI-based signal purchase:")
        for i, sig in enumerate(signals_purchased, 1):
            adj = sig.get("prob_adjustment", 0.0)
            voi = sig.get("voi", 0.0)
            direction = "↑" if adj > 0 else ("↓" if adj < 0 else "→")
            lines.append(
                f"  Signal {i}: {sig['signal_name']} "
                f"(cost=${sig['cost_usd']:.3f}, VOI={voi:.4f}, "
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
    initial_conf_low = prediction["conf_low"]
    initial_conf_high = prediction["conf_high"]

    current_prob = initial_prob
    signals_purchased: list[dict] = []
    total_cost = 0.0
    available_signals = list(SIGNAL_CATALOG.keys())

    # ── Step 2: Risk assessment via confidence interval ───────────────────────
    # conf_low > 0.5  → fraud even in worst case
    # conf_high < 0.2 → legit even in best case
    # Ambiguous interval → VOI mode
    risk_zone = "AMBIGUOUS"
    if initial_conf_low > 0.5:
        risk_zone = "RISKY"
    elif initial_conf_high < 0.2:
        risk_zone = "SAFE"

    # ── Step 3: Load Thompson Sampling bandit ──────────────────────────────────
    bandit = _get_bandit()

    # ── Step 4: Check if in ambiguous zone → VOI + bandit mode ───────────────
    # Solo compramos señales si el intervalo de confianza contiene 0.5.
    # Esto significa que ni podemos confirmar fraude ni legimitidad.
    in_ambiguous_zone = initial_conf_low <= 0.5 <= initial_conf_high

    if in_ambiguous_zone:
        purchased_names: list[str] = []
        bandit_priorities: dict[str, float] = {}

        for _round in range(MAX_SIGNALS):
            # Select best signal by VOI * bandit priority (Thompson Sampling)
            best_signal = decide_signal(
                current_prob,
                available_signals,
                purchased_names,
                bandit=bandit,
            )
            if best_signal is None:
                break

            # Store bandit priority for this signal
            bandit_priorities[best_signal] = bandit.get_priority(best_signal)

            # Compute VOI scores for logging
            voi_scores = compute_all_voi_scores(
                current_prob,
                [s for s in available_signals if s not in purchased_names]
            )

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
                    "voi": voi_scores.get(best_signal, 0.0),
                    "bandit_priority": bandit_priorities[best_signal],
                }
            )

            # Re-evaluate after this signal
            if not is_uncertain(current_prob, threshold=UNCERTAINTY_THRESHOLD):
                break

        # ── Step 5: Update bandit with rewards ────────────────────────────────
        # Reward = 1 if we escaped the uncertain zone, 0 otherwise
        final_in_ambiguous = initial_conf_low <= 0.5 <= initial_conf_high
        exited_ambiguous = final_in_ambiguous and not (initial_conf_low <= 0.5 <= initial_conf_high)

        for sig in signals_purchased:
            sig_name = sig["signal_name"]
            reward = 1 if not is_uncertain(current_prob, threshold=UNCERTAINTY_THRESHOLD) else 0
            bandit.update(sig_name, reward)

        # Save bandit state
        _save_bandit()

    # ── Step 6: Final decision ────────────────────────────────────────────────
    final_uncertainty = calculate_uncertainty(current_prob)
    decision = _make_decision(current_prob, signals_purchased)

    elapsed_ms = (time.monotonic() - start_time) * 1000

    reasoning = _build_reasoning(
        initial_prob=initial_prob,
        initial_uncertainty=initial_uncertainty,
        initial_conf_low=initial_conf_low,
        initial_conf_high=initial_conf_high,
        risk_zone=risk_zone,
        signals_purchased=signals_purchased,
        final_prob=current_prob,
        final_decision=decision,
        elapsed_ms=elapsed_ms,
    )

    return {
        "prob_fraud": round(current_prob, 6),
        "uncertainty": round(final_uncertainty, 6),
        "conf_low": round(initial_conf_low, 6),
        "conf_high": round(initial_conf_high, 6),
        "risk_zone": risk_zone,
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
            "name": "❓ Ambiguous — triggers VOI mode",
            "tx": {
                "amount": 50.0,
                "hour": 0,
                "country_mismatch": 1,
                "new_account": 0,
                "device_age_days": 30.0,
                "transactions_last_24h": 5,
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
        print(f"  confidence interval: [{result['conf_low']:.4f}, {result['conf_high']:.4f}]")
        print(f"  risk_zone         : {result['risk_zone']}")
        print(f"  Signals purchased : {len(result['signals_purchased'])}")
        print(f"  Total cost        : ${result['total_cost']:.4f}")
        print(f"  Elapsed           : {result['elapsed_ms']:.1f}ms")

        if result["signals_purchased"]:
            print(f"\n  Signals (VOI + Thompson Sampling):")
            for sig in result["signals_purchased"]:
                print(f"    • {sig['signal_name']:20s} cost=${sig['cost_usd']:.4f}  "
                      f"VOI={sig['voi']:+.4f}  "
                      f"bandit_prio={sig.get('bandit_priority', 0.5):.4f}  "
                      f"adj={sig['prob_adjustment']:+.4f}")

        print(f"\n  Reasoning: {result['reasoning']}")
