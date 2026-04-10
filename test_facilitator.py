import requests
import json

tx_hash = "c0f2faa7af072b1dd8cdbb8adcf8931b522220534676572af61c41493fcec64e"
url = "https://www.x402.org/facilitator/verify"

data = {"hash": tx_hash, "network": "stellar:testnet"}
print(f"POST to {url}")
print(f"Body: {json.dumps(data)}")

try:
    resp = requests.post(url, json=data, timeout=10)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"Error: {e}")
