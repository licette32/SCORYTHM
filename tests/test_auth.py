import base64
import json
import requests

tx_hash = "34045b0b3076ecfab9be4e8daba98e2d5327518520efedc88a1c877ee974356e"
payload = {"hash": tx_hash, "scheme": "exact", "network": "stellar:testnet"}
auth = f"x402 {base64.b64encode(json.dumps(payload).encode()).decode()}"
print("Auth header:", auth)
print()

url = "http://localhost:3000/signal/ip-reputation"
headers = {"Authorization": auth}
resp = requests.get(url, headers=headers)
print(f"Status: {resp.status_code}")
print(f"Body: {resp.text[:500]}")
