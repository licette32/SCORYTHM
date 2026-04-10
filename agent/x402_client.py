"""
agent/x402_client.py
===================
HTTP client that calls x402-protected signal endpoints and performs
real Stellar testnet micropayments.

Happy Path:
  1. Client sends request → server responds 402 with payment details
  2. Client parses payment requirements (scheme, amount, destination)
  3. Client builds a Stellar payment transaction (USDC on testnet)
  4. Client signs with STELLAR_SECRET
  5. Client submits to Horizon testnet
  6. Client retries request with X-402-Authorization header
  7. Server verifies payment and returns signal data
"""

from __future__ import annotations

import os
import base64
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:3000")
STELLAR_ADDRESS = os.getenv("STELLAR_ADDRESS", "")
STELLAR_SECRET = os.getenv("STELLAR_SECRET", "")

# USDC issuer on Stellar testnet
USDC_ISSUER_TESTNET = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
USDC_ASSET_TYPE = "credit_alphanum4"

HORIZON_TESTNET = "https://horizon-testnet.stellar.org"
NETWORK_PASSPHRASE = "Test SDF Network ; September 2015"

# Signal catalog
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


def _parse_402_response(response: requests.Response) -> dict | None:
    """
    Parse the PAYMENT-REQUIRED header from a 402 response.
    
    The header contains base64-encoded JSON with x402 payment details.
    Returns parsed payment info or None if header is missing.
    """
    import base64
    import json
    
    payment_header = response.headers.get("PAYMENT-REQUIRED")
    if not payment_header:
        return None
    
    try:
        decoded = base64.b64decode(payment_header).decode()
        data = json.loads(decoded)
        
        accepts = data.get("accepts", [{}])
        if not accepts:
            return None
        
        payment_spec = accepts[0]
        
        return {
            "scheme": payment_spec.get("scheme"),
            "network": payment_spec.get("network"),
            "amount": payment_spec.get("amount"),
            "asset": payment_spec.get("asset"),
            "payto": payment_spec.get("payTo"),
            "max_timeout_seconds": payment_spec.get("maxTimeoutSeconds"),
            "raw": data,
        }
    except Exception:
        return None


def _ensure_asset_trustline(source_secret: str, asset_code: str, asset_issuer: str) -> bool:
    """
    Ensure the account has a trustline for the specified asset.
    Returns True if trustline exists or was created.
    """
    from stellar_sdk import (
        Keypair, Server, TransactionBuilder, Network, Asset
    )
    import time
    
    server = Server(horizon_url=HORIZON_TESTNET)
    source_keypair = Keypair.from_secret(source_secret)
    source_public = source_keypair.public_key
    asset = Asset(asset_code, asset_issuer)
    
    # Check if trustline exists via Horizon API
    try:
        account_resp = server.accounts().account_id(source_public).call()
        balances = account_resp.get("balances", [])
        for bal in balances:
            if bal.get("asset_code") == asset_code and bal.get("asset_issuer") == asset_issuer:
                return True  # Trustline exists
    except Exception:
        pass
    
    # Create trustline
    try:
        account = server.load_account(source_public)
        base_fee = server.fetch_base_fee()
        trustline_tx = (
            TransactionBuilder(
                source_account=account,
                network_passphrase=NETWORK_PASSPHRASE,
                base_fee=base_fee,
            )
            .append_change_trust_op(asset=asset, limit="1000000")
            .set_timeout(30)
            .build()
        )
        trustline_tx.sign(source_keypair)
        server.submit_transaction(trustline_tx)
        time.sleep(2)  # Wait for propagation
        return True
    except Exception as e:
        if "already exists" in str(e).lower():
            return True
        return False


def _submit_stellar_payment(
    destination_address: str,
    amount: str,
    asset_issuer: str,
    source_secret: str,
) -> str:
    """
    Submit a real payment on Stellar testnet.
    - If asset is 'native' or 'XLM': send native XLM (stroops)
    - Otherwise: send the specified asset (requires trustline)
    """
    from stellar_sdk import (
        Keypair, Server, TransactionBuilder, Network, Asset
    )
    
    server = Server(horizon_url=HORIZON_TESTNET)
    source_keypair = Keypair.from_secret(source_secret)
    source_public = source_keypair.public_key
    
    # Load source account
    account = server.load_account(source_public)
    base_fee = server.fetch_base_fee()
    
    # Determine if native XLM or asset
    is_native = (asset_issuer in ('native', '', None) or 
                 amount.isdigit() and int(amount) > 1000)
    
    if is_native:
        # Send native XLM (amount is in stroops)
        transaction = (
            TransactionBuilder(
                source_account=account,
                network_passphrase=NETWORK_PASSPHRASE,
                base_fee=base_fee,
            )
            .append_payment_op(
                destination=destination_address,
                asset=Asset.native(),
                amount=amount,
            )
            .set_timeout(30)
            .build()
        )
    else:
        # Send asset (requires trustline)
        _ensure_asset_trustline(source_secret, "USDC", asset_issuer)
        account = server.load_account(source_public)  # reload after trustline
        asset = Asset("USDC", asset_issuer)
        transaction = (
            TransactionBuilder(
                source_account=account,
                network_passphrase=NETWORK_PASSPHRASE,
                base_fee=base_fee,
            )
            .append_payment_op(
                destination=destination_address,
                asset=asset,
                amount=amount,
            )
            .set_timeout(30)
            .build()
        )
    
    transaction.sign(source_keypair)
    response = server.submit_transaction(transaction)
    
    return response["hash"]


def _build_402_authorization_header(tx_hash: str) -> str:
    """
    Build the Payment-Signature header for x402 retry.
    
    Format: <raw_base64_encoded_json>
    No prefix needed - the server decodes base64 directly.
    """
    import json
    payload = {
        "hash": tx_hash,
        "scheme": "exact",
        "network": "stellar:testnet",
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()


def fetch_signal(signal_name: str, timeout: int = 15) -> dict:
    """
    Fetches a signal with real x402 payment on Stellar testnet.
    
    Parameters
    ----------
    signal_name : str
        One of: "ip-reputation", "device-history", "tx-velocity"
    timeout : int
        HTTP request timeout in seconds.
    
    Returns
    -------
    dict with:
        signal_name       : str
        cost_usd          : float
        data              : dict (the signal payload)
        tx_hash           : str (real Stellar testnet tx hash)
        payment_info      : dict (payment details)
        latency_ms        : float
        error             : str | None
        offline_simulation: bool
    """
    if signal_name not in SIGNAL_CATALOG:
        raise ValueError(
            f"Unknown signal '{signal_name}'. "
            f"Available: {list(SIGNAL_CATALOG.keys())}"
        )
    
    catalog_entry = SIGNAL_CATALOG[signal_name]
    url = f"{SERVER_URL}{catalog_entry['endpoint']}"
    cost_usd = catalog_entry["cost_usd"]
    
    headers = {
        "Content-Type": "application/json",
        "X-Signal-Client": "FraudSignal-Agent/1.0",
    }
    
    start_time = time.monotonic()
    tx_hash = None
    payment_info = {
        "amount_usd": cost_usd,
        "network": "stellar:testnet",
        "simulated": False,
    }
    
    try:
        # ── Step 1: Initial request ──────────────────────────────────────
        response = requests.get(url, headers=headers, timeout=timeout)
        latency_ms = (time.monotonic() - start_time) * 1000
        
        # ── Step 2: Handle 402 — Make real payment ─────────────────────
        if response.status_code == 402:
            print(f"[x402] Received 402 for {signal_name}, processing payment...")
            
            # Parse payment requirements
            payment = _parse_402_response(response)
            if not payment:
                return {
                    "signal_name": signal_name,
                    "cost_usd": cost_usd,
                    "data": None,
                    "tx_hash": None,
                    "payment_info": {**payment_info, "error": "No payment header in 402"},
                    "latency_ms": round(latency_ms, 2),
                    "error": "402 but no payment details",
                    "offline_simulation": False,
                }
            
            destination = payment.get("payto", STELLAR_ADDRESS)
            raw_amount = payment.get("amount", "10000")
            raw_asset = payment.get("asset", "")
            
            # Determine if native XLM or asset
            # Native is indicated by asset being null/empty/native
            is_native = (raw_asset in (None, '', 'native') or 
                         len(raw_asset) < 10)  # Short or empty means native
            
            if is_native:
                # Amount is in stroops, convert to XLM for display
                stroops = int(raw_amount)
                xlm_amount = stroops / 10_000_000
                print(f"[x402] Paying {xlm_amount:.7f} XLM (native)")
                amount_value = raw_amount  # Send in stroops
                asset_issuer = 'native'
            else:
                # Asset payment (USDC etc)
                amount_value = raw_amount
                asset_issuer = raw_asset
            
            payment_info["destination"] = destination
            payment_info["amount"] = amount_value
            payment_info["payto"] = destination
            payment_info["asset"] = asset_issuer
            
            # ── Step 3: Submit real Stellar payment ───────────────────
            if not STELLAR_SECRET:
                return {
                    "signal_name": signal_name,
                    "cost_usd": cost_usd,
                    "data": None,
                    "tx_hash": None,
                    "payment_info": {**payment_info, "error": "No STELLAR_SECRET configured"},
                    "latency_ms": round(latency_ms, 2),
                    "error": "Stellar secret not configured",
                    "offline_simulation": False,
                }
            
            try:
                # For non-native assets, ensure trustline exists
                if not is_native:
                    if not _ensure_asset_trustline(STELLAR_SECRET, "USDC", asset_issuer):
                        return {
                            "signal_name": signal_name,
                            "cost_usd": cost_usd,
                            "data": None,
                            "tx_hash": None,
                            "payment_info": {**payment_info, "error": "Could not create asset trustline"},
                            "latency_ms": round(latency_ms, 2),
                            "error": "Asset trustline not available",
                            "offline_simulation": False,
                        }
                
                tx_hash = _submit_stellar_payment(
                    destination_address=destination,
                    amount=amount_value,
                    asset_issuer=asset_issuer,
                    source_secret=STELLAR_SECRET,
                )
                print(f"[x402] Payment submitted: {tx_hash}")
                payment_info["tx_hash"] = tx_hash
            except Exception as pay_error:
                return {
                    "signal_name": signal_name,
                    "cost_usd": cost_usd,
                    "data": None,
                    "tx_hash": None,
                    "payment_info": {**payment_info, "error": str(pay_error)},
                    "latency_ms": round(latency_ms, 2),
                    "error": f"Payment failed: {pay_error}",
                    "offline_simulation": False,
                }
            
            # ── Step 4: Retry with payment header ────────────────────────
            pay_time = time.monotonic()
            auth_header = _build_402_authorization_header(tx_hash)
            retry_headers = {
                **headers,
                "payment-signature": auth_header,
            }
            
            response = requests.get(url, headers=retry_headers, timeout=timeout)
            latency_ms = (time.monotonic() - pay_time) * 1000
            
            if response.status_code != 200:
                return {
                    "signal_name": signal_name,
                    "cost_usd": cost_usd,
                    "data": None,
                    "tx_hash": tx_hash,
                    "payment_info": payment_info,
                    "latency_ms": round(latency_ms, 2),
                    "error": f"Retry failed with status {response.status_code}: {response.text}",
                    "offline_simulation": False,
                }
        
        # ── Step 5: Parse response ────────────────────────────────────
        signal_data = response.json()
        
        return {
            "signal_name": signal_name,
            "cost_usd": cost_usd,
            "data": signal_data.get("data", signal_data),
            "raw_response": signal_data,
            "tx_hash": tx_hash,
            "payment_info": payment_info,
            "latency_ms": round(latency_ms, 2),
            "error": None,
            "offline_simulation": False,
        }
    
    except requests.exceptions.ConnectionError:
        latency_ms = (time.monotonic() - start_time) * 1000
        return {
            "signal_name": signal_name,
            "cost_usd": cost_usd,
            "data": None,
            "tx_hash": None,
            "payment_info": {**payment_info, "error": "Connection refused"},
            "latency_ms": round(latency_ms, 2),
            "error": "Connection refused — server not running",
            "offline_simulation": True,
        }
    
    except requests.exceptions.Timeout:
        latency_ms = (time.monotonic() - start_time) * 1000
        return {
            "signal_name": signal_name,
            "cost_usd": cost_usd,
            "data": None,
            "tx_hash": tx_hash,
            "payment_info": payment_info,
            "latency_ms": round(latency_ms, 2),
            "error": f"Request timed out after {timeout}s",
            "offline_simulation": False,
        }
    
    except Exception as exc:
        latency_ms = (time.monotonic() - start_time) * 1000
        return {
            "signal_name": signal_name,
            "cost_usd": cost_usd,
            "data": None,
            "tx_hash": tx_hash,
            "payment_info": payment_info,
            "latency_ms": round(latency_ms, 2),
            "error": str(exc),
            "offline_simulation": False,
        }


def fetch_all_signals() -> dict[str, dict]:
    """Fetch all available signals."""
    results = {}
    for signal_name in SIGNAL_CATALOG:
        results[signal_name] = fetch_signal(signal_name)
    return results


# ─── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("FraudSignal x402 Client — Real Stellar Payments")
    print(f"Server: {SERVER_URL}")
    print(f"Stellar Address: {STELLAR_ADDRESS[:10]}...")
    print("=" * 60)
    
    for name in SIGNAL_CATALOG:
        print(f"\n→ Fetching signal: {name}")
        result = fetch_signal(name)
        print(f"  Cost       : ${result['cost_usd']:.3f}")
        print(f"  TX Hash    : {result.get('tx_hash', 'N/A') or 'N/A'}")
        print(f"  Latency    : {result['latency_ms']:.1f}ms")
        if result.get("error"):
            print(f"  ⚠️  Error   : {result['error']}")
        if result.get("offline_simulation"):
            print(f"  📡 Mode    : OFFLINE SIMULATION")
        if result.get("data"):
            print(f"  Data       : {json.dumps(result['data'], indent=4)}")
