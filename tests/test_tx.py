import requests
import json

tx_hash = "8ddb2316c8b4a40ffa09d4b7426c7360500961839ecbc91d73a966aba2947d25"
url = f"https://horizon-testnet.stellar.org/transactions/{tx_hash}"

resp = requests.get(url)
data = resp.json()

print("Transaction Status:", data.get("status"))
print("Fee Paid:", data.get("fee_paid"))
print("Source Account:", data.get("source_account"))
print("Paging Token:", data.get("paging_token", "N/A")[:50] if data.get("paging_token") else "N/A")

if "type" in data and "error" in data["type"]:
    print("ERROR:", data)
