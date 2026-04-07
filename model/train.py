"""
model/train.py
==============
Generates synthetic IEEE-CIS-like fraud data and trains a calibrated XGBoost
classifier. Saves the model to model/model.pkl.

Features used:
  - amount              : transaction amount in USD
  - hour                : hour of day (0-23)
  - country_mismatch    : 1 if billing country != IP country
  - new_account         : 1 if account age < 30 days
  - device_age_days     : days since device was first seen
  - transactions_last_24h: number of transactions in the last 24 hours

Run:
    python model/train.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    brier_score_loss,
)
from xgboost import XGBClassifier

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ─── Synthetic data generation ───────────────────────────────────────────────
N_SAMPLES = 50_000
FRAUD_RATE = 0.035  # ~3.5% fraud, realistic for card-not-present


def generate_synthetic_data(n_samples: int = N_SAMPLES, fraud_rate: float = FRAUD_RATE) -> pd.DataFrame:
    """
    Generates a synthetic dataset that mimics the statistical properties of
    the IEEE-CIS Fraud Detection dataset.

    Fraud transactions are engineered to have:
      - Higher amounts (but not always — card testing uses small amounts)
      - More transactions in the last 24h (velocity fraud)
      - Higher country mismatch rate
      - Newer accounts
      - Newer devices
      - Unusual hours (late night / early morning)
    """
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # ── Legitimate transactions ──────────────────────────────────────────────
    # Generate amount as a mix: 85% lognormal (normal purchases) + 15% uniform (small)
    # Use exact sizes that sum to n_legit to avoid length mismatches.
    n_legit_normal = int(n_legit * 0.85)
    n_legit_small  = n_legit - n_legit_normal   # remainder guarantees exact total

    legit_amounts = np.concatenate([
        np.random.lognormal(mean=4.5, sigma=1.2, size=n_legit_normal),
        np.random.uniform(1, 20, size=n_legit_small),
    ])
    # Shuffle so normal and small purchases are interleaved
    np.random.shuffle(legit_amounts)

    legit = pd.DataFrame(
        {
            "amount": legit_amounts,
            "hour": np.random.choice(
                range(24),
                size=n_legit,
                p=_hour_distribution(fraud=False),
            ),
            "country_mismatch": np.random.binomial(1, 0.05, size=n_legit),
            "new_account": np.random.binomial(1, 0.08, size=n_legit),
            "device_age_days": np.random.exponential(scale=200, size=n_legit).clip(0, 730),
            "transactions_last_24h": np.random.poisson(lam=3, size=n_legit).clip(0, 50),
            "label": 0,
        }
    )

    # ── Fraudulent transactions ──────────────────────────────────────────────
    # Mix of fraud patterns:
    #   60% high-amount fraud
    #   25% card-testing (small amounts, high velocity)
    #   15% account-takeover (new device, country mismatch)
    n_high = int(n_fraud * 0.60)
    n_card = int(n_fraud * 0.25)
    n_ato  = n_fraud - n_high - n_card   # remainder guarantees n_high+n_card+n_ato == n_fraud

    fraud_high = pd.DataFrame(
        {
            "amount": np.random.lognormal(mean=6.0, sigma=1.0, size=n_high).clip(100, 15000),
            "hour": np.random.choice(range(24), size=n_high, p=_hour_distribution(fraud=True)),
            "country_mismatch": np.random.binomial(1, 0.55, size=n_high),
            "new_account": np.random.binomial(1, 0.35, size=n_high),
            "device_age_days": np.random.exponential(scale=30, size=n_high).clip(0, 200),
            "transactions_last_24h": np.random.poisson(lam=8, size=n_high).clip(1, 60),
            "label": 1,
        }
    )

    fraud_card = pd.DataFrame(
        {
            "amount": np.random.uniform(0.5, 15, size=n_card),
            "hour": np.random.choice(range(24), size=n_card, p=_hour_distribution(fraud=True)),
            "country_mismatch": np.random.binomial(1, 0.30, size=n_card),
            "new_account": np.random.binomial(1, 0.60, size=n_card),
            "device_age_days": np.random.exponential(scale=10, size=n_card).clip(0, 60),
            "transactions_last_24h": np.random.poisson(lam=25, size=n_card).clip(5, 80),
            "label": 1,
        }
    )

    fraud_ato = pd.DataFrame(
        {
            "amount": np.random.lognormal(mean=5.5, sigma=0.8, size=n_ato).clip(50, 8000),
            "hour": np.random.choice(range(24), size=n_ato, p=_hour_distribution(fraud=True)),
            "country_mismatch": np.random.binomial(1, 0.80, size=n_ato),
            "new_account": np.random.binomial(1, 0.20, size=n_ato),
            "device_age_days": np.random.exponential(scale=5, size=n_ato).clip(0, 30),
            "transactions_last_24h": np.random.poisson(lam=6, size=n_ato).clip(1, 40),
            "label": 1,
        }
    )

    df = pd.concat([legit, fraud_high, fraud_card, fraud_ato], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Ensure correct dtypes
    df["amount"] = df["amount"].astype(float).round(2)
    df["hour"] = df["hour"].astype(int)
    df["country_mismatch"] = df["country_mismatch"].astype(int)
    df["new_account"] = df["new_account"].astype(int)
    df["device_age_days"] = df["device_age_days"].astype(float).round(1)
    df["transactions_last_24h"] = df["transactions_last_24h"].astype(int)
    df["label"] = df["label"].astype(int)

    return df


def _hour_distribution(fraud: bool) -> list:
    """
    Returns a probability distribution over 24 hours.
    Legitimate: peaks at business hours (9-18).
    Fraud: peaks at night (0-5) and late evening (22-23).
    """
    if not fraud:
        weights = np.array(
            [0.5, 0.4, 0.3, 0.3, 0.4, 0.6, 1.0, 2.0, 3.5, 4.5, 5.0, 5.5,
             5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.7]
        )
    else:
        weights = np.array(
            [3.5, 4.0, 4.5, 4.0, 3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0,
             1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 2.5, 3.0, 3.5, 4.0, 4.0]
        )
    return (weights / weights.sum()).tolist()


# ─── Feature engineering ─────────────────────────────────────────────────────
FEATURE_COLS = [
    "amount",
    "hour",
    "country_mismatch",
    "new_account",
    "device_age_days",
    "transactions_last_24h",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived features that improve model discrimination."""
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)
    df["high_velocity"] = (df["transactions_last_24h"] > 15).astype(int)
    df["risk_combo"] = df["country_mismatch"] * df["new_account"]
    df["amount_velocity_ratio"] = df["amount"] / (df["transactions_last_24h"] + 1)
    df["device_risk"] = (df["device_age_days"] < 7).astype(int)
    return df


ALL_FEATURES = FEATURE_COLS + [
    "log_amount",
    "is_night",
    "high_velocity",
    "risk_combo",
    "amount_velocity_ratio",
    "device_risk",
]


# ─── Training ─────────────────────────────────────────────────────────────────
def train_model():
    print("=" * 60)
    print("FraudSignal Agent — Model Training")
    print("=" * 60)

    print(f"\n[1/5] Generating {N_SAMPLES:,} synthetic transactions...")
    df = generate_synthetic_data()
    df = add_engineered_features(df)

    fraud_count = df["label"].sum()
    print(f"      Fraud: {fraud_count:,} ({fraud_count/len(df)*100:.1f}%)")
    print(f"      Legit: {len(df)-fraud_count:,} ({(len(df)-fraud_count)/len(df)*100:.1f}%)")

    X = df[ALL_FEATURES]
    y = df["label"]

    print("\n[2/5] Splitting train/test (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # Class weight to handle imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print(f"\n[3/5] Training XGBoost (scale_pos_weight={scale_pos_weight:.1f})...")
    base_model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
    )

    print("\n[4/5] Calibrating probabilities with CalibratedClassifierCV (isotonic)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=cv,
    )
    calibrated_model.fit(X_train, y_train)

    print("\n[5/5] Evaluating on held-out test set...")
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    print("\n" + "-" * 60)
    print(f"  ROC-AUC Score        : {auc:.4f}  (target > 0.80)")
    print(f"  Average Precision    : {ap:.4f}")
    print(f"  Brier Score (lower)  : {brier:.4f}")
    print("-" * 60)
    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    if auc < 0.80:
        print("WARNING: AUC below 0.80 - consider increasing N_SAMPLES or tuning hyperparameters.")
    else:
        print(f"AUC = {auc:.4f} - model meets quality threshold.")

    # ── Save model ────────────────────────────────────────────────────────────
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "model.pkl")

    model_artifact = {
        "model": calibrated_model,
        "features": ALL_FEATURES,
        "base_features": FEATURE_COLS,
        "metrics": {
            "roc_auc": float(auc),
            "average_precision": float(ap),
            "brier_score": float(brier),
        },
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_artifact, f)

    print(f"\nSaved model to: {model_path}")
    print("=" * 60)

    return model_artifact


if __name__ == "__main__":
    train_model()
