import base64
import json
import requests

tx_hash = "76c7d37a1db6466ded82f3a2aba4461ad674ab52b07a6843a925a40e929b5492"

# Build the payload
payload = {"hash": tx_hash, "scheme": "exact", "network": "stellar:testnet"}
auth_value = base64.b64encode(json.dumps(payload).encode()).decode()

print(f"Auth value: {auth_value}")
print(f"Hash in payload: {tx_hash}")
print()

# Test with different header names
headers_to_test = [
    {"payment-signature": auth_value},
    {"Payment-Signature": auth_value},
    {"PAYMENT-SIGNATURE": auth_value},
    {"X-Payment": auth_value},
]

for headers in headers_to_test:
    header_name = list(headers.keys())[0]
    resp = requests.get("http://localhost:3000/signal/ip-reputation", headers=headers)
    print(f"{header_name}: status={resp.status_code}")
