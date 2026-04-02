# FraudSignal Agent

> **An AI agent that knows when it doesn't know — and pays for the answer.**

FraudSignal Agent detects financial fraud using a calibrated XGBoost model. When the model is uncertain, the agent autonomously purchases external data signals via **x402 micropayments on Stellar testnet**, resolving ambiguity before making a final decision. If the model is confident, it decides alone. If not, it buys exactly the signal that reduces uncertainty the most per dollar spent.

Built for **[Stellar Hacks: Agents 2026](https://stellarhacks.devpost.com)** — deadline April 13, 2026.

---

## What it does

> *"Most fraud detection models give you a score. FraudSignal Agent gives you a score — and when it's not sure, it autonomously spends $0.001–$0.003 in Stellar micropayments to buy the external signal that most reduces its uncertainty, then decides."*

1. A transaction arrives with 6 features (amount, hour, country mismatch, new account, device age, velocity).
2. A **calibrated XGBoost** model predicts `prob_fraud` and computes `uncertainty = 1 - |2p - 1|`.
3. If `uncertainty > 0.30` (i.e., `prob_fraud` is between 0.35 and 0.65), the agent enters **signal purchase mode**.
4. It ranks available signals by **expected utility ratio** (`utility_score / cost_usd`) and buys the best one via an **x402 HTTP micropayment**.
5. The signal's data adjusts the fraud probability. The agent re-evaluates and may buy a second signal.
6. Final decision: **FRAUD** / **LEGITIMATE** / **UNCERTAIN** — with full reasoning trace.

---

## Why it's unique

| Feature | Traditional Fraud Detection | FraudSignal Agent |
|---|---|---|
| Decision model | Static threshold on score | Calibrated uncertainty-aware |
| External data | Pre-fetched batch | Purchased on-demand per transaction |
| Payment for data | Monthly API subscription | Per-query x402 micropayment ($0.001–$0.003) |
| Autonomy | Human decides when to enrich | Agent decides autonomously |
| Transparency | Black box score | Full reasoning trace |

The core innovation: **uncertainty-gated autonomous data purchasing**. The agent only spends money when it needs to, and it picks the cheapest signal that gives the most information.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FraudSignal Agent                        │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │   Frontend   │    │           FastAPI (port 8000)        │  │
│  │  index.html  │───▶│  POST /evaluate                      │  │
│  │   app.js     │    │  GET  /health                        │  │
│  └──────────────┘    └──────────────┬───────────────────────┘  │
│                                     │                           │
│                          ┌──────────▼──────────┐               │
│                          │    agent/agent.py    │               │
│                          │  evaluate_transaction│               │
│                          └──────┬──────┬────────┘               │
│                                 │      │                        │
│                    ┌────────────▼─┐  ┌─▼──────────────────┐   │
│                    │ model/predict│  │ agent/x402_client   │   │
│                    │  XGBoost +   │  │  fetch_signal()     │   │
│                    │  Calibrated  │  │  simulated payment  │   │
│                    └──────────────┘  └────────┬────────────┘   │
│                                               │                 │
│                                    ┌──────────▼──────────┐     │
│                                    │  Express (port 3000) │     │
│                                    │  @x402/express       │     │
│                                    │  /signal/ip-rep      │     │
│                                    │  /signal/device-hist │     │
│                                    │  /signal/tx-velocity │     │
│                                    └─────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘

Stellar Testnet (simulated in demo):
  Agent Wallet ──x402 micropayment──▶ Signal Server Wallet
  $0.001 – $0.003 USDC per signal
```

---

## How to run

### Prerequisites

- **Node.js** ≥ 18
- **Python** ≥ 3.12
- **WSL Ubuntu** (recommended) or any Unix-like shell

### Step 1 — Clone and set up environment

```bash
cd fraud-agent
cp .env.example .env
# Edit .env with your Stellar testnet keypair (optional for demo)
```

### Step 2 — Install Node.js dependencies

```bash
cd server
npm install
cd ..
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Train the model

```bash
python model/train.py
```

Expected output:
```
FraudSignal Agent — Model Training
[1/5] Generating 50,000 synthetic transactions...
[2/5] Splitting train/test (80/20 stratified)...
[3/5] Training XGBoost ...
[4/5] Calibrating probabilities with CalibratedClassifierCV (isotonic)...
[5/5] Evaluating on held-out test set...

  ROC-AUC Score        : 0.9XXX  (target > 0.80)
  Average Precision    : 0.XXXX
  Brier Score (lower↓) : 0.XXXX

✅  AUC = 0.9XXX — model meets quality threshold.
💾 Model saved to: model/model.pkl
```

### Step 5 — Start the x402 signal server

```bash
cd server
npm start
# Server running on http://localhost:3000
```

### Step 6 — Start the FastAPI agent API

In a new terminal, from the `fraud-agent/` directory:

```bash
uvicorn api.main:app --reload --port 8000
```

### Step 7 — Open the frontend

Open `frontend/index.html` directly in your browser:

```bash
# From WSL:
explorer.exe frontend/index.html

# Or just double-click the file in your file manager
```

The dashboard will connect to `http://localhost:8000` automatically.

---

## What each component does

| File | Role |
|---|---|
| `server/index.js` | Express server with `@x402/express` middleware. Exposes 3 payment-gated signal endpoints. |
| `model/train.py` | Generates 50k synthetic IEEE-CIS-like transactions, trains XGBoost with isotonic calibration, saves `model.pkl`. |
| `model/predict.py` | Loads `model.pkl` (auto-trains if missing), returns `prob_fraud` + `uncertainty` for a transaction dict. |
| `agent/uncertainty.py` | Math utilities: `calculate_uncertainty(p)`, `is_uncertain(p, threshold)`, `expected_utility(util, cost)`. |
| `agent/x402_client.py` | HTTP client for the signal server. Simulates x402 payments with a demo auth token. Returns signal data + fake Stellar tx hash. |
| `agent/agent.py` | Core agent logic: `evaluate_transaction()` orchestrates model prediction → uncertainty check → signal purchase loop → final decision. |
| `api/schemas.py` | Pydantic v2 models: `Transaction` (input), `EvaluationResult` (output), `SignalPurchase`. |
| `api/main.py` | FastAPI app with CORS. `POST /evaluate` runs the agent. `GET /health` returns model status. |
| `frontend/index.html` | Single-page dashboard with dark theme, animated probability bar, signal cards with tx hashes. |
| `frontend/app.js` | Dashboard logic: form handling, API calls, result rendering, health polling. |

---

## Tech stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost 2.x + scikit-learn CalibratedClassifierCV |
| Agent Logic | Python 3.12 |
| API | FastAPI + Pydantic v2 + Uvicorn |
| Signal Server | Node.js + Express + @x402/express |
| Payments | x402 protocol (HTTP 402) — Stellar testnet (simulated) |
| Frontend | Vanilla HTML/CSS/JS — no build step required |

---

## Stellar integration

The x402 payment flow works as follows:

1. The agent calls a signal endpoint (e.g., `GET /signal/device-history`).
2. In production, the server responds `402 Payment Required` with payment details (amount, asset, destination address).
3. The agent builds a **Stellar payment transaction** (USDC on testnet), signs it with the wallet secret, and submits to Horizon.
4. The agent retries the request with the `X-PAYMENT` header containing the signed transaction.
5. The `@x402/express` middleware verifies the payment via the facilitator and returns the signal data.

**In the current demo**, payments are simulated with a hardcoded auth token and a randomly generated 64-character hex string as the transaction hash. The full Stellar SDK integration code is documented in `agent/x402_client.py` under the `TODO (v2)` comment blocks.

Signal costs:
- `ip-reputation` → **$0.001** (cheapest, good for quick checks)
- `tx-velocity` → **$0.002**
- `device-history` → **$0.003** (most informative, highest utility score)

---

## Roadmap

### v1 (current — hackathon demo)
- ✅ Calibrated XGBoost with uncertainty quantification
- ✅ Autonomous signal purchase decision via EU ratio
- ✅ x402 payment simulation with Stellar tx hash
- ✅ Full reasoning trace
- ✅ Dashboard with animated probability bar

### v2 — Production
- 🔲 **Real Stellar SDK integration**: live USDC micropayments on testnet
- 🔲 **ZK proofs**: prove the model ran correctly without revealing the model weights (using RISC Zero or SP1)
- 🔲 **MCP (Model Context Protocol)**: expose the agent as an MCP tool so other AI agents can call it
- 🔲 **Multi-model ensemble**: combine XGBoost uncertainty with neural network epistemic uncertainty
- 🔲 **On-chain audit log**: record every signal purchase and decision on Stellar as a memo transaction
- 🔲 **Dynamic signal marketplace**: any data provider can register a signal endpoint and set their own x402 price

---

## License

MIT — built for Stellar Hacks: Agents 2026.
