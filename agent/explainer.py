"""
agent/explainer.py
==================
Calls Claude API (Anthropic) to generate natural-language explanations
of the fraud agent's decisions.

The agent already made the decision using XGBoost + VOI + Thompson Sampling.
Claude doesn't decide — it explains the mathematical reasoning in plain language.
"""

from __future__ import annotations

import os
import sys


def format_signals(signals: list[dict]) -> str:
    """
    Formats the purchased signals into a readable bullet-list string.
    """
    if not signals:
        return "None — model was confident without additional data."

    lines = []
    for sig in signals:
        name = sig.get("signal_name", "unknown")
        cost = sig.get("cost_usd", 0)
        data = sig.get("data") or {}

        if name == "ip-reputation":
            risk = data.get("risk_score", "?")
            vpn = data.get("is_vpn", False)
            tor = data.get("is_tor", False)
            flags = []
            if vpn: flags.append("vpn")
            if tor: flags.append("tor")
            flag_str = f"risk={risk}" + (f", {', '.join(flags)}" if flags else "")

        elif name == "device-history":
            fraud_flag = data.get("fraud_flag", "?")
            linked = data.get("linked_accounts", "?")
            flag_str = f"fraud_flag={fraud_flag}, linked={linked} accounts"

        elif name == "tx-velocity":
            vel_flag = data.get("velocity_flag", "?")
            anomaly = data.get("anomaly_detected", False)
            tx_1h = data.get("transactions_1h", "?")
            flag_str = f"{tx_1h}/1h, vel_flag={vel_flag}" + (", anomaly" if anomaly else "")

        else:
            items = [f"{k}={v}" for k, v in list(data.items())[:3]]
            flag_str = ", ".join(items) if items else "no data"

        adj = sig.get("prob_adjustment", 0)
        adj_dir = "+" if adj >= 0 else ""
        lines.append(
            f"- {name} (${cost:.3f}): {flag_str} → probability {adj_dir}{adj * 100:.1f}%"
        )

    return "\n".join(lines)


async def explain_decision(result: dict) -> str | None:
    """
    Calls Claude API to explain the fraud decision in natural language.

    Parameters
    ----------
    result : dict
        The evaluation result from evaluate_transaction().

    Returns
    -------
    str | None
        Explanation text from Claude, or None if API key is missing or call fails.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    system = (
        "You are a fraud analyst assistant for SCORYTHM. "
        "Your job is to explain, in 2-3 clear sentences, why an AI agent "
        "made a specific fraud decision. Be concise and factual. "
        "Never use jargon. Speak as if explaining to a bank officer. "
        "Always mention: the final decision, the confidence level, "
        "and if signals were purchased, what they revealed."
    )

    conf_low = result.get("conf_low", result.get("initial_prob_fraud", 0))
    conf_high = result.get("conf_high", result.get("initial_prob_fraud", 0))
    user = f"""The agent evaluated a transaction and reached this conclusion:

Decision: {result['decision']}
Initial fraud probability: {result['initial_prob_fraud']:.1%}
Final fraud probability: {result['prob_fraud']:.1%}
Confidence interval: [{conf_low:.1%}, {conf_high:.1%}]
Uncertainty level: {result['uncertainty']:.1%}

Signals purchased: {len(result['signals_purchased'])}
{format_signals(result['signals_purchased'])}

Total cost: ${result['total_cost']:.4f}

Explain why the agent made this decision in 2-3 sentences."""

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=150,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text

    except Exception:
        return None
