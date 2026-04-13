"""
model/predict.py
================
Loads the trained calibrated XGBoost model and returns fraud probability
plus an uncertainty score for a given transaction.

If model.pkl does not exist, automatically runs train.py first.

Usage:
    from model.predict import predict

    result = predict({
        "amount": 250.0,
        "hour": 2,
        "country_mismatch": 1,
        "new_account": 1,
        "device_age_days": 3.0,
        "transactions_last_24h": 12,
        "device_risk_score": 0.7,   # optional, default 0.5
        "email_domain_risk": 0.6,   # optional, default 0.3
    })
    # → {"prob_fraud": 0.823, "uncertainty": 0.298, "conf_low": 0.778, "conf_high": 0.868}
"""

import os
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_MODEL_DIR, "model.pkl")
_TRAIN_SCRIPT = os.path.join(_MODEL_DIR, "train.py")

# ─── Cached model artifact ────────────────────────────────────────────────────
_artifact: dict | None = None


def _load_model() -> dict:
    """
    Loads model.pkl from disk. If the file does not exist, triggers training
    automatically by running train.py as a subprocess.
    """
    global _artifact

    if _artifact is not None:
        return _artifact

    if not os.path.exists(_MODEL_PATH):
        print("[predict.py] model.pkl not found — running train.py automatically...")
        result = subprocess.run(
            [sys.executable, _TRAIN_SCRIPT],
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"train.py failed with return code {result.returncode}. "
                "Please run `python model/train.py` manually and check for errors."
            )

    with open(_MODEL_PATH, "rb") as f:
        _artifact = pickle.load(f)

    return _artifact


def _build_feature_row(transaction: dict, feature_list: list) -> pd.DataFrame:
    """
    Constructs a single-row DataFrame with all required features,
    including engineered features derived from the base inputs.

    Parameters
    ----------
    transaction : dict
        Must contain the base features. New optional features:
        device_risk_score, email_domain_risk (default to neutral values if absent)
    feature_list : list
        Full list of features expected by the model (base + engineered).

    Returns
    -------
    pd.DataFrame with shape (1, len(feature_list))
    """
    amount = float(transaction.get("amount", 100.0))
    hour = int(transaction.get("hour", 12))
    country_mismatch = int(transaction.get("country_mismatch", 0))
    new_account = int(transaction.get("new_account", 0))
    device_age_days = float(transaction.get("device_age_days", 180.0))
    transactions_last_24h = int(transaction.get("transactions_last_24h", 3))

    # New features — default to neutral/mid values if not provided
    device_risk_score = float(transaction.get("device_risk_score", 0.3))
    email_domain_risk = float(transaction.get("email_domain_risk", 0.2))

    row = {
        "amount": amount,
        "hour": hour,
        "country_mismatch": country_mismatch,
        "new_account": new_account,
        "device_age_days": device_age_days,
        "transactions_last_24h": transactions_last_24h,
        "device_risk_score": device_risk_score,
        "email_domain_risk": email_domain_risk,
        # Engineered features (must match train.py exactly)
        "log_amount": np.log1p(amount),
        "is_night": int(0 <= hour <= 5),
        "high_velocity": int(transactions_last_24h > 12),
        "risk_combo": country_mismatch * new_account,
        "amount_velocity_ratio": amount / (transactions_last_24h + 1),
        "device_risk": int(device_age_days < 7),
        "device_email_risk": device_risk_score * email_domain_risk,
    }

    # Build DataFrame with columns in the exact order the model expects
    # Only include columns that exist in the feature_list (backward compat)
    filtered_row = {k: v for k, v in row.items() if k in feature_list}
    df = pd.DataFrame([filtered_row])[feature_list]
    return df


def predict(transaction: dict) -> dict:
    """
    Predicts fraud probability and uncertainty for a single transaction.

    Parameters
    ----------
    transaction : dict
        Keys: amount, hour, country_mismatch, new_account,
              device_age_days, transactions_last_24h
              device_risk_score (optional, 0-1)
              email_domain_risk (optional, 0-1)

    Returns
    -------
    dict with:
        prob_fraud  : float [0, 1] — calibrated probability of fraud
        uncertainty : float [0, 1] — 1.0 means maximum uncertainty (prob ≈ 0.5)
                      Computed as 1 - |2 * prob_fraud - 1|
        conf_low    : float [0, 1] — lower bound of the confidence interval
        conf_high   : float [0, 1] — upper bound of the confidence interval
    """
    artifact = _load_model()
    model = artifact["model"]
    feature_list = artifact["features"]

    X = _build_feature_row(transaction, feature_list)
    proba = model.predict_proba(X)[0]

    prob_fraud = float(np.clip(proba[1], 0.0, 1.0))

    uncertainty = float(1.0 - abs(2.0 * prob_fraud - 1.0))

    # Conformal Prediction: margin proportional to uncertainty.
    # uncertainty=1 (p≈0.5) → wide interval
    # uncertainty=0 (p≈0 or 1) → narrow interval
    margin = uncertainty * 0.5
    conf_low = max(0.0, prob_fraud - margin)
    conf_high = min(1.0, prob_fraud + margin)

    return {
        "prob_fraud": round(prob_fraud, 6),
        "uncertainty": round(uncertainty, 6),
        "conf_low": round(conf_low, 6),
        "conf_high": round(conf_high, 6),
    }


def get_model_metrics() -> dict:
    """Returns the validation metrics stored at training time."""
    artifact = _load_model()
    return artifact.get("metrics", {})


# ─── CLI usage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    test_cases = [
        {
            "name": "CASO A — Legítima clara (no debe comprar señales)",
            "tx": {
                "amount": 45.0, "hour": 14, "country_mismatch": 0, "new_account": 0,
                "device_age_days": 365.0, "transactions_last_24h": 2,
                "device_risk_score": 0.10, "email_domain_risk": 0.05,
            },
        },
        {
            "name": "CASO B — Ambigua que se resuelve con 1 señal",
            "tx": {
                "amount": 320.0, "hour": 22, "country_mismatch": 1, "new_account": 0,
                "device_age_days": 45.0, "transactions_last_24h": 7,
                "device_risk_score": 0.40, "email_domain_risk": 0.35,
            },
        },
        {
            "name": "CASO C — Muy ambigua que necesita 2 señales",
            "tx": {
                "amount": 890.0, "hour": 3, "country_mismatch": 1, "new_account": 1,
                "device_age_days": 8.0, "transactions_last_24h": 12,
                "device_risk_score": 0.60, "email_domain_risk": 0.50,
            },
        },
        {
            "name": "CASO D — Fraude claro (no debe comprar señales)",
            "tx": {
                "amount": 4500.0, "hour": 2, "country_mismatch": 1, "new_account": 1,
                "device_age_days": 1.0, "transactions_last_24h": 28,
                "device_risk_score": 0.90, "email_domain_risk": 0.85,
            },
        },
    ]

    print("\n" + "=" * 60)
    print("Scorythm Agent — Prediction Test (4 Demo Cases)")
    print("=" * 60)

    metrics = get_model_metrics()
    if metrics:
        print(f"\nModel metrics (from training):")
        print(f"  ROC-AUC : {metrics.get('roc_auc', 'N/A'):.4f}  (target 0.82–0.89)")
        print(f"  Avg Prec: {metrics.get('average_precision', 'N/A'):.4f}")
        print(f"  Brier   : {metrics.get('brier_score', 'N/A'):.4f}")
        if "ambiguous_pct" in metrics:
            print(f"  Ambig % : {metrics.get('ambiguous_pct', 0):.1f}%  (target ~15%)")

    print()
    for case in test_cases:
        result = predict(case["tx"])
        p = result["prob_fraud"]
        u = result["uncertainty"]

        if u > 0.30:
            status = "⚠️  AMBIGUOUS — agent will buy signals"
        elif p >= 0.65:
            status = "🚨 FRAUD (confident)"
        elif p <= 0.35:
            status = "✅ LEGITIMATE (confident)"
        else:
            status = "❓ BORDERLINE"

        print(f"  [{case['name']}]")
        print(f"    prob_fraud : {p:.4f}")
        print(f"    uncertainty: {u:.4f}")
        print(f"    CI         : [{result['conf_low']:.4f}, {result['conf_high']:.4f}]")
        print(f"    Status     : {status}\n")
