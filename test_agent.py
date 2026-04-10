from agent.agent import evaluate_transaction
import json

result = evaluate_transaction({
    'amount': 50.0,
    'hour': 0,
    'country_mismatch': 1,
    'new_account': 0,
    'device_age_days': 30.0,
    'transactions_last_24h': 5
})

print('Decision:', result['decision'])
print('Signals:', len(result['signals_purchased']))
print('Total cost:', result['total_cost'])
for s in result['signals_purchased']:
    tx = s.get('tx_hash', 'N/A')
    print(f"  - {s.get('signal_name')}: tx={tx[:20] if tx != 'N/A' else 'N/A'}...")
    if s.get('data'):
        print(f"    Data: {json.dumps(s.get('data'), indent=4)[:200]}")
