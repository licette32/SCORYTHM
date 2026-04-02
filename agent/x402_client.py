"""
agent/x402_client.py
====================
HTTP client that calls the x402-protected signal endpoints on the local
Express server (localhost:3000).

For the hackathon demo, payments are simulated: we include a hardcoded
Authorization header that the server accepts in testnet mode, and we
generate a fake Stellar transaction hash to represent the micropayment.

In production, the real x402 flow would be:
  1. Client sends request → server responds 402 with payment details
  2. Client signs a Stellar payment transaction
  3. Client retries with X-PAYMENT header containing the signed tx
  4. Server verifies via facilitator and returns the data

TODO (v2 — Real Stellar Integration):
  See the TODO blocks below for the actual Stellar SDK implementation.
"""

from __future__ import annotations

import os
import random
import string
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:3000")
STELLAR_ADDRESS = os.getenv("STELLAR_ADDRESS", "GBXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
STELLAR_SECRET = os.getenv("STELLAR_SECRET", "SXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

# Hardcoded testnet authorization token for demo purposes.
# In production this would be a signed Stellar transaction envelope.
_DEMO_AUTH_TOKEN = "Bearer x402-testnet-demo-token-fraudsignal-hackathon-2026"

# Signal catalog: maps signal name → endpoint path, cost, and utility score
SIGNAL_CATALOG: dict[str, dict] = {
    "ip-reputation": {
        "endpoint": "/signal/ip-reputation",
        "cost_usd": 0.001,
        "utility_score": 0.72,
        "description": "IP address reputation, VPN/Tor detection, abuse reports",
    },
    "device-history": {
        "endpoint": "/signal/device-history",
        "cost_usd": 0.003,
        "utility_score": 0.88,
        "description": "Device fingerprint history, fraud flags, linked accounts",
    },
    "tx-velocity": {
        "endpoint": "/signal/tx-velocity",
        "cost_usd": 0.002,
        "utility_score": 0.81,
        "description": "Transaction velocity metrics for the last 24 hours",
    },
}


def _generate_simulated_tx_hash() -> str:
    """
    Generates a fake Stellar transaction hash for demo purposes.
    Real Stellar tx hashes are 64-character hex strings.

    TODO (v2): Replace with actual Stellar SDK transaction hash.
    """
    return "".join(random.choices(string.hexdigits.lower(), k=64))


def _simulate_x402_payment(signal_name: str, cost_usd: float) -> dict:
    """
    Simulates the x402 micropayment flow for testnet demo.

    In the real implementation this would:
      1. Receive a 402 response with payment details (amount, asset, destination)
      2. Build a Stellar payment transaction using stellar-sdk
      3. Sign it with STELLAR_SECRET
      4. Submit to Stellar testnet
      5. Return the transaction hash

    TODO (v2 — Real Stellar SDK integration):
    ─────────────────────────────────────────
    from stellar_sdk import Server, Keypair, TransactionBuilder, Network, Asset

    horizon = Server("https://horizon-testnet.stellar.org")
    keypair = Keypair.from_secret(STELLAR_SECRET)
    account = horizon.load_account(keypair.public_key)

    # USDC on Stellar testnet (Circle testnet issuer)
    usdc_asset = Asset(
        "USDC",
        "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
    )

    # Convert USD amount to stroops (1 XLM = 10_000_000 stroops)
    # For USDC: amount is in units (e.g., "0.001" for $0.001)
    amount_str = f"{cost_usd:.7f}"

    tx = (
        TransactionBuilder(
            source_account=account,
            network_passphrase=Network.TESTNET_NETWORK_PASSPHRASE,
            base_fee=100,
        )
        .append_payment_op(
            destination=STELLAR_ADDRESS,  # signal server's address
            asset=usdc_asset,
            amount=amount_str,
        )
        .set_timeout(30)
        .build()
    )
    tx.sign(keypair)
    response = horizon.submit_transaction(tx)
    tx_hash = response["hash"]
    ─────────────────────────────────────────
    """
    tx_hash = _generate_simulated_tx_hash()
    return {
        "tx_hash": tx_hash,
        "amount_usd": cost_usd,
        "from_address": STELLAR_ADDRESS,
        "to_address": STELLAR_ADDRESS,  # demo: same address
        "network": "stellar-testnet",
        "simulated": True,
    }


def fetch_signal(signal_name: str, timeout: int = 10) -> dict:
    """
    Fetches a signal from the x402-protected server.

    Simulates the payment by including the demo Authorization header.
    In production, this would perform the full x402 handshake.

    Parameters
    ----------
    signal_name : str
        One of: "ip-reputation", "device-history", "tx-velocity"
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    dict with:
        signal_name     : str
        cost_usd        : float
        data            : dict  (the signal payload from the server)
        simulated_tx_hash: str  (fake Stellar tx hash for demo)
        payment_info    : dict  (payment simulation details)
        latency_ms      : float (request latency)
        error           : str | None

    Raises
    ------
    ValueError if signal_name is not in SIGNAL_CATALOG.
    """
    if signal_name not in SIGNAL_CATALOG:
        raise ValueError(
            f"Unknown signal '{signal_name}'. "
            f"Available: {list(SIGNAL_CATALOG.keys())}"
        )

    catalog_entry = SIGNAL_CATALOG[signal_name]
    url = f"{SERVER_URL}{catalog_entry['endpoint']}"
    cost_usd = catalog_entry["cost_usd"]

    # Simulate the micropayment before calling the endpoint
    payment_info = _simulate_x402_payment(signal_name, cost_usd)

    headers = {
        "Authorization": _DEMO_AUTH_TOKEN,
        # TODO (v2): Replace with real x402 payment header:
        # "X-PAYMENT": base64_encoded_signed_stellar_tx,
        "Content-Type": "application/json",
        "X-Signal-Client": "FraudSignal-Agent/1.0",
    }

    start_time = time.monotonic()
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        latency_ms = (time.monotonic() - start_time) * 1000

        # The x402 server may return 402 if payment is invalid.
        # In demo mode with the hardcoded token, it should return 200.
        if response.status_code == 402:
            # TODO (v2): Handle real 402 response — parse payment requirements
            # and perform actual Stellar payment before retrying.
            return {
                "signal_name": signal_name,
                "cost_usd": cost_usd,
                "data": None,
                "simulated_tx_hash": payment_info["tx_hash"],
                "payment_info": payment_info,
                "latency_ms": round(latency_ms, 2),
                "error": "402 Payment Required — real x402 payment needed (demo mode limitation)",
            }

        response.raise_for_status()
        signal_data = response.json()

        return {
            "signal_name": signal_name,
            "cost_usd": cost_usd,
            "data": signal_data.get("data", signal_data),
            "raw_response": signal_data,
            "simulated_tx_hash": payment_info["tx_hash"],
            "payment_info": payment_info,
            "latency_ms": round(latency_ms, 2),
            "error": None,
        }

    except requests.exceptions.ConnectionError:
        latency_ms = (time.monotonic() - start_time) * 1000
        # Server not running — return simulated data for offline demo
        return _fallback_signal(signal_name, cost_usd, payment_info, latency_ms,
                                error="Connection refused — server not running, using simulated data")

    except requests.exceptions.Timeout:
        latency_ms = (time.monotonic() - start_time) * 1000
        return _fallback_signal(signal_name, cost_usd, payment_info, latency_ms,
                                error=f"Request timed out after {timeout}s")

    except requests.exceptions.RequestException as exc:
        latency_ms = (time.monotonic() - start_time) * 1000
        return _fallback_signal(signal_name, cost_usd, payment_info, latency_ms,
                                error=str(exc))


def _fallback_signal(
    signal_name: str,
    cost_usd: float,
    payment_info: dict,
    latency_ms: float,
    error: str,
) -> dict:
    """
    Returns simulated signal data when the server is unreachable.
    This allows the agent to function in offline/demo mode.
    """
    simulated_data = _generate_simulated_signal_data(signal_name)
    return {
        "signal_name": signal_name,
        "cost_usd": cost_usd,
        "data": simulated_data,
        "raw_response": None,
        "simulated_tx_hash": payment_info["tx_hash"],
        "payment_info": payment_info,
        "latency_ms": round(latency_ms, 2),
        "error": error,
        "offline_simulation": True,
    }


def _generate_simulated_signal_data(signal_name: str) -> dict:
    """Generates realistic simulated signal data for offline demo mode."""
    rng = random.Random()

    if signal_name == "ip-reputation":
        risk = round(rng.uniform(0.1, 0.9), 3)
        return {
            "risk_score": risk,
            "is_vpn": rng.random() > 0.75,
            "is_tor": rng.random() > 0.92,
            "is_datacenter": rng.random() > 0.80,
            "country_code": rng.choice(["US", "BR", "RU", "CN", "NG", "DE"]),
            "abuse_reports_30d": rng.randint(0, 40),
            "fraud_probability_adjustment": round(risk * 0.25, 4),
        }
    elif signal_name == "device-history":
        flags = rng.randint(0, 4)
        return {
            "device_age_days": rng.randint(0, 730),
            "previous_fraud_flags": flags,
            "unique_accounts_linked": rng.randint(1, 12),
            "avg_transaction_amount_usd": round(rng.uniform(10, 1500), 2),
            "risk_tier": "HIGH" if flags > 2 else ("MEDIUM" if flags > 0 else "LOW"),
            "last_seen_country": rng.choice(["US", "BR", "RU", "CN", "NG"]),
            "fraud_probability_adjustment": round(flags * 0.08, 4),
        }
    elif signal_name == "tx-velocity":
        count = rng.randint(1, 60)
        vel = round(min(1.0, count / 50), 3)
        return {
            "transactions_last_24h": count,
            "total_amount_usd_24h": round(rng.uniform(50, 12000), 2),
            "unique_merchants_24h": rng.randint(1, 15),
            "declined_transactions_24h": rng.randint(0, 8),
            "velocity_score": vel,
            "anomaly_detected": count > 40,
            "fraud_probability_adjustment": round(vel * 0.30, 4),
        }
    return {}


def fetch_all_signals() -> dict[str, dict]:
    """
    Fetches all available signals. Useful for testing.

    Returns
    -------
    dict mapping signal_name → fetch_signal() result
    """
    results = {}
    for signal_name in SIGNAL_CATALOG:
        results[signal_name] = fetch_signal(signal_name)
    return results


# ─── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("=" * 60)
    print("FraudSignal x402 Client — Test")
    print(f"Server: {SERVER_URL}")
    print("=" * 60)

    for name in SIGNAL_CATALOG:
        print(f"\n→ Fetching signal: {name}")
        result = fetch_signal(name)
        print(f"  Cost       : ${result['cost_usd']:.3f}")
        print(f"  TX Hash    : {result['simulated_tx_hash'][:16]}...")
        print(f"  Latency    : {result['latency_ms']:.1f}ms")
        if result.get("error"):
            print(f"  ⚠️  Error   : {result['error']}")
        if result.get("offline_simulation"):
            print(f"  📡 Mode    : OFFLINE SIMULATION")
        print(f"  Data       : {json.dumps(result['data'], indent=4)}")
