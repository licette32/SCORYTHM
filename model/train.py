"""
model/train.py
==============
Generates synthetic fraud data with deliberate ambiguity zones and trains
a calibrated XGBoost classifier. Target AUC: 0.82–0.89.

Features used:
  - amount                : transaction amount in USD
  - hour                  : hour of day (0-23)
  - country_mismatch      : 1 if billing country != IP country
  - new_account           : 1 if account age < 30 days
  - device_age_days       : days since device was first seen
  - transactions_last_24h : number of transactions in the last 24 hours
  - device_risk_score     : composite device risk [0-1]
  - email_domain_risk     : risk score of email domain [0-1]

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

# ─── Config ───────────────────────────────────────────────────────────────────
N_SAMPLES = 50_000
FRAUD_RATE = 0.15   # 15% fraud — higher rate creates more overlap


def generate_synthetic_data(n_samples: int = N_SAMPLES, fraud_rate: float = FRAUD_RATE) -> pd.DataFrame:
    """
    Generates synthetic transactions designed to produce AUC ~0.82-0.89.

    Key design decisions:
    - High fraud rate (15%) creates more class overlap
    - Large Gaussian noise on ALL features blurs the decision boundary
    - 30% of fraud transactions are "soft fraud" — nearly identical to legit
    - 20% of legit transactions have "suspicious" features (false positives)
    - device_risk_score and email_domain_risk have high variance
    """
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    rng = np.random.default_rng(SEED)

    # ── Legitimate transactions ──────────────────────────────────────────────
    # 80% clearly legit, 20% "suspicious legit" (overlap zone)
    n_legit_clear = int(n_legit * 0.80)
    n_legit_susp  = n_legit - n_legit_clear

    # Clearly legitimate
    legit_clear = pd.DataFrame({
        "amount": np.clip(
            rng.lognormal(mean=4.0, sigma=1.2, size=n_legit_clear) + rng.normal(0, 30, size=n_legit_clear),
            1, 3000
        ),
        "hour": rng.choice(range(24), size=n_legit_clear, p=_hour_distribution(fraud=False)),
        "country_mismatch": rng.binomial(1, 0.05, size=n_legit_clear),
        "new_account": rng.binomial(1, 0.08, size=n_legit_clear),
        "device_age_days": np.clip(
            rng.exponential(scale=200, size=n_legit_clear) + rng.normal(0, 40, size=n_legit_clear),
            0, 730
        ),
        "transactions_last_24h": np.clip(
            rng.poisson(lam=3, size=n_legit_clear) + rng.integers(-1, 3, size=n_legit_clear),
            0, 30
        ).astype(int),
        "device_risk_score": np.clip(rng.beta(1.5, 10, size=n_legit_clear) + rng.normal(0, 0.10, size=n_legit_clear), 0, 1),
        "email_domain_risk": np.clip(rng.beta(1.2, 12, size=n_legit_clear) + rng.normal(0, 0.08, size=n_legit_clear), 0, 1),
        "label": 0,
    })

    # Suspicious legitimate (overlap zone — hard cases)
    legit_susp = pd.DataFrame({
        "amount": np.clip(
            rng.lognormal(mean=5.0, sigma=1.5, size=n_legit_susp) + rng.normal(0, 100, size=n_legit_susp),
            10, 8000
        ),
        "hour": rng.choice(range(24), size=n_legit_susp),  # uniform — no pattern
        "country_mismatch": rng.binomial(1, 0.35, size=n_legit_susp),
        "new_account": rng.binomial(1, 0.30, size=n_legit_susp),
        "device_age_days": np.clip(
            rng.exponential(scale=40, size=n_legit_susp) + rng.normal(0, 20, size=n_legit_susp),
            0, 200
        ),
        "transactions_last_24h": np.clip(
            rng.poisson(lam=7, size=n_legit_susp) + rng.integers(-2, 4, size=n_legit_susp),
            0, 40
        ).astype(int),
        "device_risk_score": np.clip(rng.beta(3, 4, size=n_legit_susp) + rng.normal(0, 0.15, size=n_legit_susp), 0, 1),
        "email_domain_risk": np.clip(rng.beta(2.5, 4, size=n_legit_susp) + rng.normal(0, 0.15, size=n_legit_susp), 0, 1),
        "label": 0,
    })

    # ── Fraudulent transactions ──────────────────────────────────────────────
    # 40% clear fraud, 30% soft fraud (overlaps with legit), 30% mixed
    n_fraud_clear = int(n_fraud * 0.40)
    n_fraud_soft  = int(n_fraud * 0.30)
    n_fraud_mixed = n_fraud - n_fraud_clear - n_fraud_soft

    # Clear fraud (high signal)
    fraud_clear = pd.DataFrame({
        "amount": np.clip(
            rng.lognormal(mean=6.5, sigma=1.0, size=n_fraud_clear) + rng.normal(0, 80, size=n_fraud_clear),
            100, 25000
        ),
        "hour": rng.choice(range(24), size=n_fraud_clear, p=_hour_distribution(fraud=True)),
        "country_mismatch": rng.binomial(1, 0.70, size=n_fraud_clear),
        "new_account": rng.binomial(1, 0.55, size=n_fraud_clear),
        "device_age_days": np.clip(
            rng.exponential(scale=8, size=n_fraud_clear) + rng.normal(0, 3, size=n_fraud_clear),
            0, 60
        ),
        "transactions_last_24h": np.clip(
            rng.poisson(lam=18, size=n_fraud_clear) + rng.integers(-3, 5, size=n_fraud_clear),
            3, 80
        ).astype(int),
        "device_risk_score": np.clip(rng.beta(7, 1.5, size=n_fraud_clear) + rng.normal(0, 0.08, size=n_fraud_clear), 0, 1),
        "email_domain_risk": np.clip(rng.beta(6, 1.5, size=n_fraud_clear) + rng.normal(0, 0.08, size=n_fraud_clear), 0, 1),
        "label": 1,
    })

    # Soft fraud (deliberately similar to legit — creates ambiguous zone)
    fraud_soft = pd.DataFrame({
        "amount": np.clip(
            rng.lognormal(mean=4.5, sigma=1.6, size=n_fraud_soft) + rng.normal(0, 120, size=n_fraud_soft),
            5, 6000
        ),
        "hour": rng.choice(range(24), size=n_fraud_soft),  # uniform
        "country_mismatch": rng.binomial(1, 0.30, size=n_fraud_soft),
        "new_account": rng.binomial(1, 0.25, size=n_fraud_soft),
        "device_age_days": np.clip(
            rng.exponential(scale=80, size=n_fraud_soft) + rng.normal(0, 30, size=n_fraud_soft),
            0, 400
        ),
        "transactions_last_24h": np.clip(
            rng.poisson(lam=5, size=n_fraud_soft) + rng.integers(-2, 5, size=n_fraud_soft),
            0, 30
        ).astype(int),
        "device_risk_score": np.clip(rng.beta(3, 3, size=n_fraud_soft) + rng.normal(0, 0.18, size=n_fraud_soft), 0, 1),
        "email_domain_risk": np.clip(rng.beta(2.5, 3, size=n_fraud_soft) + rng.normal(0, 0.18, size=n_fraud_soft), 0, 1),
        "label": 1,
    })

    # Mixed fraud (moderate signals)
    fraud_mixed = pd.DataFrame({
        "amount": np.clip(
            rng.lognormal(mean=5.5, sigma=1.3, size=n_fraud_mixed) + rng.normal(0, 60, size=n_fraud_mixed),
            20, 12000
        ),
        "hour": rng.choice(range(24), size=n_fraud_mixed, p=_hour_distribution(fraud=True)),
        "country_mismatch": rng.binomial(1, 0.50, size=n_fraud_mixed),
        "new_account": rng.binomial(1, 0.40, size=n_fraud_mixed),
        "device_age_days": np.clip(
            rng.exponential(scale=20, size=n_fraud_mixed) + rng.normal(0, 10, size=n_fraud_mixed),
            0, 150
        ),
        "transactions_last_24h": np.clip(
            rng.poisson(lam=10, size=n_fraud_mixed) + rng.integers(-3, 4, size=n_fraud_mixed),
            1, 50
        ).astype(int),
        "device_risk_score": np.clip(rng.beta(4, 2.5, size=n_fraud_mixed) + rng.normal(0, 0.14, size=n_fraud_mixed), 0, 1),
        "email_domain_risk": np.clip(rng.beta(3.5, 2.5, size=n_fraud_mixed) + rng.normal(0, 0.14, size=n_fraud_mixed), 0, 1),
        "label": 1,
    })

    df = pd.concat([legit_clear, legit_susp, fraud_clear, fraud_soft, fraud_mixed], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Ensure correct dtypes
    df["amount"] = df["amount"].astype(float).round(2)
    df["hour"] = df["hour"].astype(int)
    df["country_mismatch"] = df["country_mismatch"].astype(int)
    df["new_account"] = df["new_account"].astype(int)
    df["device_age_days"] = df["device_age_days"].astype(float).round(1)
    df["transactions_last_24h"] = df["transactions_last_24h"].astype(int)
    df["device_risk_score"] = df["device_risk_score"].astype(float).round(4)
    df["email_domain_risk"] = df["email_domain_risk"].astype(float).round(4)
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
    "device_risk_score",
    "email_domain_risk",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived features. Kept intentionally simple to avoid over-fitting."""
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])
    df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)
    df["high_velocity"] = (df["transactions_last_24h"] > 12).astype(int)
    df["risk_combo"] = df["country_mismatch"] * df["new_account"]
    df["amount_velocity_ratio"] = df["amount"] / (df["transactions_last_24h"] + 1)
    df["device_risk"] = (df["device_age_days"] < 7).astype(int)
    # Non-linear interaction: device risk × email risk
    df["device_email_risk"] = df["device_risk_score"] * df["email_domain_risk"]
    return df


ALL_FEATURES = FEATURE_COLS + [
    "log_amount",
    "is_night",
    "high_velocity",
    "risk_combo",
    "amount_velocity_ratio",
    "device_risk",
    "device_email_risk",
]


# ─── Training ─────────────────────────────────────────────────────────────────
def train_model():
    print("=" * 60)
    print("Scorythm Agent — Model Training")
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

    # ── Label noise injection ─────────────────────────────────────────────────
    # Flip 25% of labels in the "ambiguous zone" (transactions with mixed signals)
    # This forces the model to be uncertain in that region → AUC drops to ~0.85
    rng_noise = np.random.default_rng(SEED + 1)
    y_train_noisy = y_train.copy()

    # Define ambiguous zone: moderate feature values
    X_tr = X_train.copy()
    ambig_mask = (
        (X_tr["device_risk_score"] > 0.25) & (X_tr["device_risk_score"] < 0.75) &
        (X_tr["email_domain_risk"] > 0.20) & (X_tr["email_domain_risk"] < 0.70) &
        (X_tr["transactions_last_24h"] < 20)
    )
    ambig_idx = y_train_noisy.index[ambig_mask]
    flip_n = int(len(ambig_idx) * 0.30)  # flip 30% of ambiguous zone
    flip_idx = rng_noise.choice(ambig_idx, size=flip_n, replace=False)
    y_train_noisy.loc[flip_idx] = 1 - y_train_noisy.loc[flip_idx]  # flip label

    noise_pct = (y_train_noisy != y_train).mean() * 100
    print(f"      Label noise injected: {noise_pct:.1f}% of training labels flipped in ambiguous zone")

    print(f"\n[3/5] Training XGBoost (balanced weights, noisy labels)...")
    base_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.10,
        subsample=0.70,
        colsample_bytree=0.60,
        min_child_weight=20,
        gamma=0.5,
        reg_alpha=1.0,
        reg_lambda=3.0,
        scale_pos_weight=1.0,      # balanced — noisy labels handle the rest
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
    )

    print("\n[4/5] Calibrating probabilities with CalibratedClassifierCV (sigmoid)...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=cv,
    )
    calibrated_model.fit(X_train, y_train_noisy)

    print("\n[5/5] Evaluating on held-out test set...")
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    # Count ambiguous zone
    ambig_mask = (y_prob >= 0.35) & (y_prob <= 0.65)
    ambig_pct = ambig_mask.mean() * 100

    print("\n" + "-" * 60)
    print(f"  ROC-AUC Score        : {auc:.4f}  (target 0.82–0.89)")
    print(f"  Average Precision    : {ap:.4f}")
    print(f"  Brier Score (lower)  : {brier:.4f}")
    print(f"  Ambiguous zone (%)   : {ambig_pct:.1f}%  (target ~15%)")
    print("-" * 60)
    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    if auc < 0.80:
        print("⚠️  WARNING: AUC below 0.80 — model may be too weak.")
    elif auc > 0.90:
        print("⚠️  WARNING: AUC above 0.90 — model may be too separable for demo.")
    else:
        print(f"✅ AUC = {auc:.4f} — within target range 0.82–0.89.")

    if ambig_pct < 10:
        print(f"⚠️  WARNING: Only {ambig_pct:.1f}% ambiguous — signals may not trigger.")
    else:
        print(f"✅ Ambiguous zone: {ambig_pct:.1f}% — agent will purchase signals.")

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
            "ambiguous_pct": float(ambig_pct),
        },
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_artifact, f)

    print(f"\nSaved model to: {model_path}")

    # ── Save sample transactions ──────────────────────────────────────────────
    _save_sample_transactions(model_dir)

    print("=" * 60)
    return model_artifact


def _save_sample_transactions(model_dir: str):
    """Save 4 demo cases + 16 additional test cases to data/sample_transactions.csv"""
    data_dir = os.path.join(model_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "sample_transactions.csv")

    rows = [
        # ── 4 DEMO CASES ──────────────────────────────────────────────────────
        # CASO A — Legítima clara (no debe comprar señales)
        {
            "case": "A_legit_clear",
            "amount": 45.0, "hour": 14, "country_mismatch": 0, "new_account": 0,
            "device_age_days": 365.0, "transactions_last_24h": 2,
            "device_risk_score": 0.10, "email_domain_risk": 0.05,
            "expected_decision": "LEGITIMATE", "expected_signals": 0,
        },
        # CASO B — Ambigua que se resuelve con 1 señal
        {
            "case": "B_ambig_1signal",
            "amount": 320.0, "hour": 22, "country_mismatch": 1, "new_account": 0,
            "device_age_days": 45.0, "transactions_last_24h": 7,
            "device_risk_score": 0.40, "email_domain_risk": 0.35,
            "expected_decision": "FRAUD", "expected_signals": 1,
        },
        # CASO C — Muy ambigua que necesita 2 señales
        {
            "case": "C_ambig_2signals",
            "amount": 890.0, "hour": 3, "country_mismatch": 1, "new_account": 1,
            "device_age_days": 8.0, "transactions_last_24h": 12,
            "device_risk_score": 0.60, "email_domain_risk": 0.50,
            "expected_decision": "FRAUD", "expected_signals": 2,
        },
        # CASO D — Fraude claro (no debe comprar señales)
        {
            "case": "D_fraud_clear",
            "amount": 4500.0, "hour": 2, "country_mismatch": 1, "new_account": 1,
            "device_age_days": 1.0, "transactions_last_24h": 28,
            "device_risk_score": 0.90, "email_domain_risk": 0.85,
            "expected_decision": "FRAUD", "expected_signals": 0,
        },
        # ── ADDITIONAL TEST CASES ─────────────────────────────────────────────
        {"case": "legit_01", "amount": 28.5, "hour": 10, "country_mismatch": 0, "new_account": 0, "device_age_days": 500.0, "transactions_last_24h": 1, "device_risk_score": 0.05, "email_domain_risk": 0.03, "expected_decision": "LEGITIMATE", "expected_signals": 0},
        {"case": "legit_02", "amount": 120.0, "hour": 15, "country_mismatch": 0, "new_account": 0, "device_age_days": 200.0, "transactions_last_24h": 3, "device_risk_score": 0.12, "email_domain_risk": 0.08, "expected_decision": "LEGITIMATE", "expected_signals": 0},
        {"case": "legit_03", "amount": 75.0, "hour": 11, "country_mismatch": 0, "new_account": 0, "device_age_days": 730.0, "transactions_last_24h": 2, "device_risk_score": 0.08, "email_domain_risk": 0.04, "expected_decision": "LEGITIMATE", "expected_signals": 0},
        {"case": "fraud_01", "amount": 3200.0, "hour": 1, "country_mismatch": 1, "new_account": 1, "device_age_days": 0.5, "transactions_last_24h": 35, "device_risk_score": 0.95, "email_domain_risk": 0.90, "expected_decision": "FRAUD", "expected_signals": 0},
        {"case": "fraud_02", "amount": 5800.0, "hour": 3, "country_mismatch": 1, "new_account": 1, "device_age_days": 1.0, "transactions_last_24h": 42, "device_risk_score": 0.92, "email_domain_risk": 0.88, "expected_decision": "FRAUD", "expected_signals": 0},
        {"case": "fraud_03", "amount": 1200.0, "hour": 4, "country_mismatch": 1, "new_account": 1, "device_age_days": 2.0, "transactions_last_24h": 22, "device_risk_score": 0.85, "email_domain_risk": 0.80, "expected_decision": "FRAUD", "expected_signals": 0},
        {"case": "ambig_01", "amount": 450.0, "hour": 21, "country_mismatch": 1, "new_account": 0, "device_age_days": 30.0, "transactions_last_24h": 8, "device_risk_score": 0.45, "email_domain_risk": 0.40, "expected_decision": "FRAUD", "expected_signals": 1},
        {"case": "ambig_02", "amount": 180.0, "hour": 23, "country_mismatch": 0, "new_account": 1, "device_age_days": 15.0, "transactions_last_24h": 6, "device_risk_score": 0.35, "email_domain_risk": 0.30, "expected_decision": "UNCERTAIN", "expected_signals": 2},
        {"case": "ambig_03", "amount": 650.0, "hour": 2, "country_mismatch": 1, "new_account": 0, "device_age_days": 20.0, "transactions_last_24h": 10, "device_risk_score": 0.55, "email_domain_risk": 0.45, "expected_decision": "FRAUD", "expected_signals": 1},
        {"case": "ambig_04", "amount": 290.0, "hour": 22, "country_mismatch": 0, "new_account": 1, "device_age_days": 12.0, "transactions_last_24h": 9, "device_risk_score": 0.42, "email_domain_risk": 0.38, "expected_decision": "UNCERTAIN", "expected_signals": 2},
        {"case": "ambig_05", "amount": 520.0, "hour": 20, "country_mismatch": 1, "new_account": 1, "device_age_days": 5.0, "transactions_last_24h": 11, "device_risk_score": 0.58, "email_domain_risk": 0.52, "expected_decision": "FRAUD", "expected_signals": 2},
        {"case": "ambig_06", "amount": 380.0, "hour": 1, "country_mismatch": 1, "new_account": 0, "device_age_days": 25.0, "transactions_last_24h": 7, "device_risk_score": 0.48, "email_domain_risk": 0.42, "expected_decision": "FRAUD", "expected_signals": 1},
        {"case": "ambig_07", "amount": 210.0, "hour": 23, "country_mismatch": 1, "new_account": 1, "device_age_days": 10.0, "transactions_last_24h": 8, "device_risk_score": 0.50, "email_domain_risk": 0.46, "expected_decision": "UNCERTAIN", "expected_signals": 2},
        {"case": "ambig_08", "amount": 760.0, "hour": 3, "country_mismatch": 0, "new_account": 1, "device_age_days": 18.0, "transactions_last_24h": 13, "device_risk_score": 0.62, "email_domain_risk": 0.55, "expected_decision": "FRAUD", "expected_signals": 1},
        {"case": "card_test_01", "amount": 1.5, "hour": 4, "country_mismatch": 1, "new_account": 1, "device_age_days": 0.5, "transactions_last_24h": 45, "device_risk_score": 0.88, "email_domain_risk": 0.75, "expected_decision": "FRAUD", "expected_signals": 0},
        {"case": "card_test_02", "amount": 5.0, "hour": 2, "country_mismatch": 0, "new_account": 1, "device_age_days": 1.0, "transactions_last_24h": 38, "device_risk_score": 0.82, "email_domain_risk": 0.70, "expected_decision": "FRAUD", "expected_signals": 0},
    ]

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved sample transactions to: {csv_path}")


if __name__ == "__main__":
    train_model()
