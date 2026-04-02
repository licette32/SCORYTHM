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
    })
    # → {"prob_fraud": 0.823, "uncertainty": 0.298}
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
    including engineered features derived from the base 6 inputs.

    Parameters
    ----------
    transaction : dict
        Must contain the 6 base features:
        amount, hour, country_mismatch, new_account,
        device_age_days, transactions_last_24h
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

    row = {
        "amount": amount,
        "hour": hour,
        "country_mismatch": country_mismatch,
        "new_account": new_account,
        "device_age_days": device_age_days,
        "transactions_last_24h": transactions_last_24h,
        # Engineered features (must match train.py exactly)
        "log_amount": np.log1p(amount),
        "is_night": int(0 <= hour <= 5),
        "high_velocity": int(transactions_last_24h > 15),
        "risk_combo": country_mismatch * new_account,
        "amount_velocity_ratio": amount / (transactions_last_24h + 1),
        "device_risk": int(device_age_days < 7),
    }

    # Build DataFrame with columns in the exact order the model expects
    df = pd.DataFrame([row])[feature_list]
    return df


def predict(transaction: dict) -> dict:
    """
    Predicts fraud probability and uncertainty for a single transaction.

    Parameters
    ----------
    transaction : dict
        Keys: amount, hour, country_mismatch, new_account,
              device_age_days, transactions_last_24h

    Returns
    -------
    dict with:
        prob_fraud  : float [0, 1] — calibrated probability of fraud
        uncertainty : float [0, 1] — 1.0 means maximum uncertainty (prob ≈ 0.5)
                      Computed as 1 - |2 * prob_fraud - 1|
                      This is the complement of the "margin" from the decision boundary.
    """
    artifact = _load_model()
    model = artifact["model"]
    feature_list = artifact["features"]

    X = _build_feature_row(transaction, feature_list)
    proba = model.predict_proba(X)[0]  # shape: (2,) → [P(legit), P(fraud)]

    prob_fraud = float(np.clip(proba[1], 0.0, 1.0))

    # Uncertainty formula:
    #   |2p - 1| gives the "confidence margin" (0 = max uncertainty, 1 = certain)
    #   1 - |2p - 1| gives uncertainty (1 = max uncertainty at p=0.5, 0 = certain)
    uncertainty = float(1.0 - abs(2.0 * prob_fraud - 1.0))

    return {
        "prob_fraud": round(prob_fraud, 6),
        "uncertainty": round(uncertainty, 6),
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
            "name": "Suspicious transaction (high risk)",
            "tx": {
                "amount": 1850.0,
                "hour": 3,
                "country_mismatch": 1,
                "new_account": 1,
                "device_age_days": 2.0,
                "transactions_last_24h": 22,
            },
        },
        {
            "name": "Normal transaction (low risk)",
            "tx": {
                "amount": 45.0,
                "hour": 14,
                "country_mismatch": 0,
                "new_account": 0,
                "device_age_days": 365.0,
                "transactions_last_24h": 2,
            },
        },
        {
            "name": "Ambiguous transaction (uncertain)",
            "tx": {
                "amount": 300.0,
                "hour": 22,
                "country_mismatch": 1,
                "new_account": 0,
                "device_age_days": 45.0,
                "transactions_last_24h": 8,
            },
        },
    ]

    print("\n" + "=" * 55)
    print("FraudSignal Agent — Prediction Test")
    print("=" * 55)

    metrics = get_model_metrics()
    if metrics:
        print(f"\nModel metrics (from training):")
        print(f"  ROC-AUC : {metrics.get('roc_auc', 'N/A'):.4f}")
        print(f"  Avg Prec: {metrics.get('average_precision', 'N/A'):.4f}")
        print(f"  Brier   : {metrics.get('brier_score', 'N/A'):.4f}")

    print()
    for case in test_cases:
        result = predict(case["tx"])
        print(f"  [{case['name']}]")
        print(f"    Input     : {json.dumps(case['tx'])}")
        print(f"    prob_fraud: {result['prob_fraud']:.4f}")
        print(f"    uncertainty: {result['uncertainty']:.4f}")
        status = "⚠️  UNCERTAIN" if result["uncertainty"] > 0.3 else (
            "🚨 FRAUD" if result["prob_fraud"] >= 0.5 else "✅ LEGIT"
        )
        print(f"    Status    : {status}\n")
