"""
api/schemas.py
==============
Pydantic v2 schemas for the FraudSignal Agent API.

Defines the request and response models for the /evaluate endpoint.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Request Schema ───────────────────────────────────────────────────────────

class Transaction(BaseModel):
    """
    A financial transaction to be evaluated for fraud.

    All fields have sensible defaults so the frontend can submit
    partial data and still get a valid response.
    """

    amount: float = Field(
        default=150.0,
        ge=0.0,
        le=1_000_000.0,
        description="Transaction amount in USD",
        examples=[150.0, 2500.0, 35.0],
    )

    hour: int = Field(
        default=14,
        ge=0,
        le=23,
        description="Hour of day when the transaction occurred (0-23, UTC)",
        examples=[14, 3, 22],
    )

    country_mismatch: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 if billing country does not match IP geolocation country, 0 otherwise",
        examples=[0, 1],
    )

    new_account: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 if the account was created less than 30 days ago, 0 otherwise",
        examples=[0, 1],
    )

    device_age_days: float = Field(
        default=180.0,
        ge=0.0,
        le=3650.0,
        description="Number of days since the device was first seen by the system",
        examples=[180.0, 2.0, 365.0],
    )

    transactions_last_24h: int = Field(
        default=3,
        ge=0,
        le=500,
        description="Number of transactions made by this account in the last 24 hours",
        examples=[3, 28, 1],
    )

    @field_validator("country_mismatch", "new_account", mode="before")
    @classmethod
    def coerce_to_binary(cls, v: Any) -> int:
        """Accepts True/False/1/0 and coerces to int."""
        if isinstance(v, bool):
            return int(v)
        v = int(v)
        if v not in (0, 1):
            raise ValueError("Must be 0 or 1")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "amount": 2500.0,
                    "hour": 3,
                    "country_mismatch": 1,
                    "new_account": 1,
                    "device_age_days": 2.0,
                    "transactions_last_24h": 28,
                }
            ]
        }
    }


# ─── Signal Purchase Schema ───────────────────────────────────────────────────

class SignalPurchase(BaseModel):
    """Details of a single signal purchased during evaluation."""

    signal_name: str = Field(description="Name of the purchased signal")
    cost_usd: float = Field(description="Cost of the signal in USD")
    data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Signal payload returned by the server",
    )
    simulated_tx_hash: str = Field(
        description="Simulated Stellar transaction hash (64-char hex string)"
    )
    latency_ms: float = Field(
        default=0.0,
        description="HTTP request latency in milliseconds",
    )
    prob_adjustment: float = Field(
        default=0.0,
        description="How much this signal shifted the fraud probability",
    )
    voi: float = Field(
        default=0.0,
        description="Value of Information (VOI) of this signal at purchase time",
    )
    bandit_priority: float = Field(
        default=0.5,
        description="Thompson Sampling priority score for this signal",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the signal fetch failed",
    )
    offline_simulation: bool = Field(
        default=False,
        description="True if the signal data was simulated (server offline)",
    )


# ─── Response Schema ──────────────────────────────────────────────────────────

class EvaluationResult(BaseModel):
    """
    Complete result of a transaction fraud evaluation by the FraudSignal Agent.
    """

    prob_fraud: float = Field(
        ge=0.0,
        le=1.0,
        description="Final calibrated probability of fraud after all signals",
    )

    uncertainty: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Final uncertainty score. "
            "1.0 = maximum uncertainty (prob ≈ 0.5), "
            "0.0 = maximum certainty"
        ),
    )

    conf_low: float = Field(
        ge=0.0,
        le=1.0,
        description="Lower bound of the 95% confidence interval for prob_fraud",
    )

    conf_high: float = Field(
        ge=0.0,
        le=1.0,
        description="Upper bound of the 95% confidence interval for prob_fraud",
    )

    risk_zone: str = Field(
        description="Risk classification: RISKY (conf_low>0.5), SAFE (conf_high<0.2), AMBIGUOUS (otherwise)",
        pattern="^(RISKY|SAFE|AMBIGUOUS)$",
    )

    decision: str = Field(
        description="Final decision: FRAUD, LEGITIMATE, or UNCERTAIN",
        pattern="^(FRAUD|LEGITIMATE|UNCERTAIN)$",
    )

    signals_purchased: list[SignalPurchase] = Field(
        default_factory=list,
        description="List of external signals purchased via x402 micropayments",
    )

    total_cost: float = Field(
        ge=0.0,
        description="Total USD spent on signal purchases",
    )

    reasoning: str = Field(
        description="Human-readable explanation of the agent's decision process",
    )

    initial_prob_fraud: float = Field(
        ge=0.0,
        le=1.0,
        description="Model's initial fraud probability before purchasing signals",
    )

    initial_uncertainty: float = Field(
        ge=0.0,
        le=1.0,
        description="Model's initial uncertainty before purchasing signals",
    )

    elapsed_ms: float = Field(
        ge=0.0,
        description="Total evaluation time in milliseconds",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prob_fraud": 0.823,
                    "uncertainty": 0.354,
                    "conf_low": 0.778,
                    "conf_high": 0.868,
                    "risk_zone": "RISKY",
                    "decision": "FRAUD",
                    "signals_purchased": [
                        {
                            "signal_name": "device-history",
                            "cost_usd": 0.003,
                            "data": {
                                "device_age_days": 2,
                                "previous_fraud_flags": 3,
                                "risk_tier": "HIGH",
                            },
                            "simulated_tx_hash": "a3f8c2d1e4b7a9f0" + "0" * 48,
                            "latency_ms": 12.4,
                    "prob_adjustment": 0.0823,
                    "voi": 0.0345,
                    "error": None,
                            "offline_simulation": False,
                        }
                    ],
                    "total_cost": 0.003,
                    "reasoning": "Initial model prediction: prob_fraud=0.7407 | ...",
                    "initial_prob_fraud": 0.7407,
                    "initial_uncertainty": 0.5186,
                    "elapsed_ms": 45.2,
                }
            ]
        }
    }


# ─── Health check schema ──────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str = Field(default="ok")
    service: str = Field(default="FraudSignal Agent API")
    model_loaded: bool = Field(description="Whether the ML model is loaded")
    model_metrics: Optional[dict[str, float]] = Field(
        default=None,
        description="Validation metrics from training (roc_auc, etc.)",
    )
    bandit_stats: Optional[dict[str, dict]] = Field(
        default=None,
        description="Thompson Sampling bandit statistics per signal (alpha, beta, success_rate, n_trials)",
    )
    version: str = Field(default="1.0.0")
