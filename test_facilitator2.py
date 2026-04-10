import requests
import json

tx_hash = "76c7d37a1db6466ded82f3a2aba4461ad674ab52b07a6843a925a40e929b5492"

# Test the facilitator verify endpoint
facilitator_url = "https://www.x402.org/facilitator"

# Try different formats
payloads = [
    {"hash": tx_hash, "network": "stellar:testnet"},
    {"hash": tx_hash, "network": "stellar:testnet", "scheme": "exact"},
    {"paymentPayload": {"hash": tx_hash}, "paymentRequirements": {"network": "stellar:testnet"}},
]

for i, payload in enumerate(payloads):
    print(f"\n=== Test {i+1} ===")
    print(f"Payload: {payload}")
    try:
        resp = requests.post(f"{facilitator_url}/verify", json=payload, timeout=10)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")

# Also check what /supported returns
print("\n=== Supported ===")
try:
    resp = requests.get(f"{facilitator_url}/supported", timeout=10)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text[:1000]}")
except Exception as e:
    print(f"Error: {e}")
