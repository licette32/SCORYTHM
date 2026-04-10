# FraudSignal Agent — Informe Funcional
## Stellar Hacks 2026

---

## 1. Resumen Ejecutivo

**FraudSignal Agent** es un sistema de detección de fraude en transacciones financieras que utiliza aprendizaje automático calibrado, cuantificación de incertidumbre y un agente de decisión que aprende a comprar señales de información de forma autónoma mediante micropagos en la red Stellar.

El proyecto demuestra una arquitectura completa de agente de IA que:
- Evalúa transacciones con confianza estadísticamente garantizada
- Decide dinámicamente cuándo comprar información adicional
- Aprende qué señales son más útiles mediante Thompson Sampling
- Realiza pagos reales en Stellar Testnet via x402

---

## 2. Visión de Negocio

### 2.1 Problema

Los sistemas tradicionales de detección de fraude operan con umbrales fijos, lo que genera:
- **Falsos positivos excesivos**: Transacciones legítimas bloqueadas → mala experiencia de usuario
- **Falsos negativos costosos**: Fraude no detectado → pérdidas financieras directas
- **Sobregeneralización**: Mismo comportamiento para todas las transacciones → ineficiencia

### 2.2 Solución Propuesta

Un agente que:
1. **Cuantifica su propia incertidumbre** — sabe cuándo no sabe
2. **Compra información adicional de forma inteligente** — solo cuando el costo de no saber es mayor
3. **Aprende del mercado de señales** — descubre qué fuentes de datos son más valiosas
4. **Opera de manera auditable** — registra cada decisión y su razonamiento

### 2.3 Modelo de Monetización x402

```
┌─────────────────────────────────────────────────────────────┐
│                     PAGADOR (Cliente)                       │
│              /evaluate → paga señales via x402             │
└─────────────────────┬───────────────────────────────────────┘
                      │ $0.001-0.01 XLM por señal
                      │ HTTP con header payment-signature
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              FRAUDSIGNAL AGENT (Servidor)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │  FastAPI    │→ │    Agente    │→ │  XGBoost + CI   │    │
│  │  (Node.js)  │  │   (Python)   │  │  (Conformal)    │    │
│  └─────────────┘  └──────────────┘  └─────────────────┘    │
│                           │                                 │
│                    ┌──────┴───────┐                         │
│                    │ VOI + Bandit │                         │
│                    │  (Thompson)  │                         │
│                    └──────────────┘                         │
└─────────────────────┬───────────────────────────────────────┘
                      │ $0.001-0.01 XLM por señal
                      │ fetch_signal() via x402
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               SIGNAL PROVIDERS (3 fuentes)                   │
│  ┌──────────────┐ ┌───────────────┐ ┌───────────────────┐   │
│  │ IP Reputation│ │ Device History│ │ TX Velocity       │   │
│  │ ($0.001)     │ │ ($0.002)      │ │ ($0.003)          │   │
│  └──────────────┘ └───────────────┘ └───────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Casos de Uso

| Escenario | Comportamiento del Agente |
|-----------|---------------------------|
| Transacción clara legítima | Aprueba inmediatamente, 0 costo |
| Transacción claramente fraudulenta | Bloquea inmediatamente, 0 costo |
| Transacción ambigua | Compra 1-2 señales, aprende, decide |
| Primera evaluación ambigua | Compra señales, inicializa bandit |
| Evaluación repetida ambigua | Bandit prioriza señales útiles |

---

## 3. Arquitectura Técnica

### 3.1 Stack Tecnológico

```
┌─────────────────────────────────────────────────┐
│                  FRONTEND                        │
│         HTML + CSS + Vanilla JS                  │
│         Dashboard: http://localhost:3000         │
└─────────────────────┬───────────────────────────┘
                      │ HTTP (REST)
┌─────────────────────▼───────────────────────────┐
│              API LAYER (Node.js)                 │
│         server/index.js                          │
│         - Sirve frontend estático               │
│         - Proxy /api/* → :8000                  │
│         - Verificación simplificada x402        │
└─────────────────────┬───────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────┐
│              BACKEND (Python/FastAPI)            │
│         api/main.py                              │
│         - GET  /health                          │
│         - POST /evaluate                        │
└────────┬───────────────────────────┬─────────────┘
         │                           │
┌────────▼────────┐     ┌───────────▼──────────────┐
│  FRAUD AGENT    │     │   SIGNAL PROVIDERS        │
│  agent/agent.py │     │   agent/x402_client.py    │
│                 │     │                           │
│  - predict()   │←────→│  - ip-reputation          │
│  - evaluate()   │     │  - device-history         │
│                 │     │  - tx-velocity            │
└────────┬────────┘     └───────────────────────────┘
         │
┌────────▼──────────────┐
│  MODEL LAYER           │
│  model/predict.py      │
│  model/train.py        │
│                       │
│  - XGBoost calibrado   │
│  - Conformal Predict.  │
│  - Confidence Intervals│
└────────────────────────┘
```

### 3.2 Modelo de Machine Learning

#### 3.2.1 Training Pipeline (`model/train.py`)

```
Dataset: 50,000 transacciones sintéticas
├── Features (6):
│   ├── amount (0 - 1,000,000 USD)
│   ├── hour (0-23 UTC)
│   ├── country_mismatch (0/1)
│   ├── new_account (0/1)
│   ├── device_age_days (0-3650)
│   └── transactions_last_24h (0-500)
│
├── Label: is_fraud (0/1)
│
├── Modelo: XGBoost
│   ├── n_estimators=200
│   ├── max_depth=6
│   ├── learning_rate=0.1
│   └── objective='binary:logistic'
│
└── Calibración: CalibratedClassifierCV (isotonic)
    └── Garantiza probabilidades calibradas [0,1]
```

#### 3.2.2 Predicción con Incertidumbre (`model/predict.py`)

```
┌─────────────────────────────────────────────────┐
│              predict(transaction)                 │
├─────────────────────────────────────────────────┤
│ 1. Load model if not cached                     │
│ 2. Run XGBoost → raw_prob (calibrated)          │
│ 3. Bootstrap (100 iterations)                   │
│    → produce 100 predictions                    │
│ 4. Conformal Prediction:                         │
│    - conf_low  = percentile(predictions, 2.5)   │
│    - conf_high = percentile(predictions, 97.5)  │
│ 5. uncertainty = conf_high - conf_low          │
│ 6. Return:                                      │
│    { prob_fraud, uncertainty, conf_low, conf_high } │
└─────────────────────────────────────────────────┘
```

**Garantía estadística**: El intervalo `[conf_low, conf_high]` contiene el 95% de las veces la probabilidad real del modelo.

### 3.3 Conformal Prediction — Detalle Matemático

```python
# Given:
#   - Calibration set: {X_i, Y_i} con Y_i binario (fraud/legit)
#   - X_new: nueva transacción
#   - T(X_i, Y_i) = |P(Y=1|X_i) - Y_i|  # calibración residual

# Step 1: Calcular non-conformity scores en calibration set
scores = [|p_1 - y_1|, |p_2 - y_2|, ..., |p_n - y_n|]

# Step 2: Para X_new con predicción p_new:
#   - α_low  = quantile(scores, α/2)      # α = 0.05 → 2.5%
#   - α_high = quantile(scores, 1-α/2)    # → 97.5%

# Step 3: Intervalo de confianza:
#   conf_low  = max(0, p_new - α_low)
#   conf_high = min(1, p_new + α_high)
```

**Interpretación**: Si el intervalo `[0.3, 0.7]` → 95% de certeza de que la probabilidad real está en ese rango.

### 3.4 Value of Information (VOI) — Detalle Matemático

```python
# Parámetros de costos asimétricos:
C_FN = 1.0   # Costo de no detectar fraude (alto)
C_FP = 0.1   # Costo de bloquear legítimo (bajo)

def compute_voi(p_fraud, utility_score, signal_cost):
    # Incertidumbre: u = 1 - |2p - 1|
    # u = 0 → p ≈ 0 o 1 (modelo seguro)
    # u = 1 → p = 0.5 (máxima incertidumbre)
    uncertainty = 1.0 - abs(2.0 * p_fraud - 1.0)
    
    # Pérdida esperada antes de señal:
    L_before = p_fraud * C_FN + (1 - p_fraud) * C_FP
    
    # Mejora por señal:
    improvement = (C_FN - C_FP) * uncertainty * utility_score
    
    # Corrección por incertidumbre residual:
    correction = 0.5 * (C_FN - C_FP) * (uncertainty ** 2) * (utility_score ** 2)
    
    # Pérdida después de señal:
    L_after = L_before - improvement + correction
    
    # VOI = reducción de pérdida - costo de señal
    voi = L_before - L_after - signal_cost
    
    return voi
```

**Decisión de compra**: Si `VOI > 0` → comprar señal

### 3.5 Thompson Sampling Bandit — Detalle Matemático

```python
class ThompsonBandit:
    """
    Thompson Sampling con distribución Beta.
    
    Para cada señal s, mantenemos:
    - alpha_s: éxitos observados + 1
    - beta_s: fracasos observados + 1
    
    Selección:
    1. Para cada signal, sample de Beta(alpha_s, beta_s)
    2. Multiplicar VOI por el sample
    3. Seleccionar señal con mayor producto
    4. Actualizar alpha/beta según recompensa
    """
    
    def get_priority(self, signal_name):
        sample = beta(self.alpha, self.beta)
        return sample  # [0, 1], mayor = más probable útil
    
    def update(self, signal_name, reward):
        # reward = 1 si escapó zona incierta, 0 si no
        self.alpha[signal_name] += reward
        self.beta[signal_name] += (1 - reward)
    
    def select(self, voi_scores):
        priorities = {s: self.get_priority(s) for s in voi_scores}
        adjusted = {s: voi_scores[s] * priorities[s] for s in voi_scores}
        return max(adjusted, key=adjusted.get)
```

**Beneficio**: El bandit explora señales infrautilizadas mientras explota las que funcionan.

### 3.6 Sistema de Pagos x402 con Stellar

#### 3.6.1 Por qué Stellar y XLM Nativo

**Problema resuelto**: El issuer de USDC en testnet (`CBIELTK6YBZJU...`) no es válido en Horizon testnet, causando errores en `@x402/stellar`.

**Solución**: Usar XLM nativo directamente, evitando assets que requieren trustlines.

#### 3.6.2 Flujo de Pago

```
┌──────────────┐                              ┌─────────────────┐
│ Fraud Agent  │                              │ Signal Provider │
│ (payer)      │                              │ (receiver)      │
└──────┬───────┘                              └────────┬────────┘
       │                                              │
       │  1. Pre-approve amount via Stellar SDK       │
       │─────────────────────────────────────────────→
       │                                              │
       │  2. Construir TransactionBuilder             │
       │     - source: GBULNFYD... (agent address)   │
       │     - destination: GB2FGBR... (provider)   │
       │     - amount: 0.001 XLM (~0.1 USD)          │
       │     - memo: "sig:ip-reputation"              │
       │                                              │
       │  3. Firmar con secret                        │
       │     - sign() → SHA256                       │
       │                                              │
       │  4. submit_transaction()                      │
       │─────────────────────────────────────────────→
       │                                              │
       │  5. TX Hash returned (64 hex chars)           │
       │◄─────────────────────────────────────────────
       │                                              │
       │  6. Generar PaymentAuthorization             │
       │     - { hash: tx_hash, spender: agent_addr }│
       │     - base64 encode                          │
       │                                              │
       │  7. Request signal with header:             │
       │     payment-signature: <base64>              │
       │─────────────────────────────────────────────→
       │                                              │
       │  8. Signal returned                          │
       │◄─────────────────────────────────────────────
```

#### 3.6.3 Claves de Testnet

```
Address: GBULNFYDNJNDW2HLRJWVWQDYVPP5PDEJ56POOQXACD575GHQXQMGKF4S
Secret:  SCPEQCPK55S57XB7S3S6EOTBHGZMJKSHTYOJA732U7YCWWTYN72FALYH
Network: Stellar Testnet
```

#### 3.6.4 Catálogo de Señales

| Signal | Costo (XLM) | Costo (USD) | Utilidad | Datos Proveídos |
|--------|-------------|--------------|----------|-----------------|
| ip-reputation | 0.001 | ~$0.0002 | 0.70 | VPN, Tor, risk_score, abuse_reports |
| device-history | 0.002 | ~$0.0004 | 0.75 | device_age, fraud_flags, risk_tier |
| tx-velocity | 0.003 | ~$0.0006 | 0.80 | velocity_score, anomaly, declined |

### 3.7 Decisión del Agente

```
┌────────────────────────────────────────────────────────┐
│              evaluate_transaction(tx)                   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Step 1: GET INITIAL PREDICTION                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │ prob_fraud = XGBoost_predict(tx)                 │  │
│  │ conf_low, conf_high = ConformalPrediction()      │  │
│  │ uncertainty = conf_high - conf_low               │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  Step 2: RISK ZONE ASSESSMENT                          │
│  ┌─────────────────────────────────────────────────┐  │
│  │ if conf_low > 0.5:        RISKY (block)         │  │
│  │ elif conf_high < 0.2:     SAFE (approve)        │  │
│  │ else:                     AMBIGUOUS             │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  Step 3: IF AMBIGUOUS → BUY SIGNALS                   │
│  ┌─────────────────────────────────────────────────┐  │
│  │ max 2 signals per evaluation                    │  │
│  │ for each signal:                                │  │
│  │   1. VOI = compute_voi(prob, utility, cost)    │  │
│  │   2. Priority = bandit.sample()                 │  │
│  │   3. adjusted = VOI * Priority                  │  │
│  │   4. Buy signal with highest adjusted VOI       │  │
│  │   5. Apply probability adjustment                │  │
│  │   6. Break if no longer uncertain               │  │
│  │   7. Update bandit with reward                  │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  Step 4: FINAL DECISION                                │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Without signals:                                │  │
│  │   FRAUD      if prob >= 0.65                    │  │
│  │   LEGITIMATE if prob <= 0.35                    │  │
│  │   UNCERTAIN  otherwise                          │  │
│  │                                                 │  │
│  │ With signals (tighter thresholds):              │  │
│  │   FRAUD      if prob >= 0.55                    │  │
│  │   LEGITIMATE if prob <= 0.45                    │  │
│  │   UNCERTAIN  otherwise                          │  │
│  └─────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## 4. Endpoints API

### 4.1 `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_metrics": {
    "roc_auc": 0.987,
    "average_precision": 0.891
  },
  "bandit_state": {
    "ip-reputation": {
      "alpha": 3.0,
      "beta": 2.0,
      "n_trials": 5,
      "expected_value": 0.60
    },
    "device-history": {...},
    "tx-velocity": {...}
  },
  "stellar_address": "GBULNFYDNJNDW2HLRJWVWQDYVPP5PDEJ56POOQXACD575GHQXQMGKF4S"
}
```

### 4.2 `POST /evaluate`

**Request:**
```json
{
  "amount": 2500.00,
  "hour": 3,
  "country_mismatch": 1,
  "new_account": 1,
  "device_age_days": 2.0,
  "transactions_last_24h": 28
}
```

**Response:**
```json
{
  "prob_fraud": 0.782,
  "uncertainty": 0.12,
  "conf_low": 0.65,
  "conf_high": 0.89,
  "risk_zone": "RISKY",
  "decision": "FRAUD",
  "signals_purchased": [],
  "total_cost": 0.0,
  "reasoning": "Initial model prediction: prob_fraud=0.782...",
  "initial_prob_fraud": 0.782,
  "initial_uncertainty": 0.12,
  "elapsed_ms": 45.3
}
```

**Response con señales:**
```json
{
  "prob_fraud": 0.48,
  "uncertainty": 0.08,
  "conf_low": 0.35,
  "conf_high": 0.61,
  "risk_zone": "AMBIGUOUS",
  "decision": "UNCERTAIN",
  "signals_purchased": [
    {
      "signal_name": "ip-reputation",
      "cost_usd": 0.0002,
      "tx_hash": "a1b2c3d4e5f6...",
      "data": {
        "risk_score": 0.85,
        "is_vpn": true,
        "is_tor": false
      },
      "prob_adjustment": 0.045,
      "voi": 0.023,
      "bandit_priority": 0.67
    },
    {
      "signal_name": "tx-velocity",
      "cost_usd": 0.0006,
      "tx_hash": "b2c3d4e5f6a7...",
      "data": {...},
      "prob_adjustment": -0.085,
      "voi": 0.015,
      "bandit_priority": 0.72
    }
  ],
  "total_cost": 0.0008,
  "elapsed_ms": 234.1
}
```

---

## 5. Dashboard de Visualización

### 5.1 Features Implementadas

```
┌────────────────────────────────────────────────────────────────┐
│  FraudSignal Agent          Stellar Hacks 2026  [●]           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────┐    ┌──────────────────────────────────┐ │
│  │ Transaction Input │    │ Result Panel                     │ │
│  │                   │    │                                  │ │
│  │ Presets:          │    │ ┌──────────────────────────────┐  │ │
│  │ [Fraud] [Legit]   │    │ │ 🚨 FRAUD                    │  │ │
│  │ [Uncertain]       │    │ │ Transaction flagged...      │  │ │
│  │                   │    │ │              Cost: $0.000   │  │ │
│  │ Amount: [2500.00] │    │ └──────────────────────────────┘  │ │
│  │ Hour:   [3     ]  │    │                                  │ │
│  │ Country: [Yes ▼]  │    │ Fraud Probability                 │ │
│  │ New Acct: [Yes▼]  │    │ ████████████░░░░░░░ 78.2%        │ │
│  │ Device:  [2 days] │    │                                  │ │
│  │ Tx/24h:  [28    ]  │    │ Confidence Interval              │ │
│  │                   │    │ [====|====] [65% - 89%]          │ │
│  │ [⚡ Evaluate]     │    │                                  │ │
│  └──────────────────┘    │ Risk Zone: RISKY                  │ │
│                          │                                  │ │
│  ┌──────────────────┐    │ ┌──────┬─────────┬──────────┐    │ │
│  │ Model Info       │    │ │ 12%  │    0    │   45ms   │    │ │
│  │ ROC-AUC: 98.7%   │    │ │Uncert│ Bought  │ Time     │    │ │
│  │ PR: 89.1%        │    │ └──────┴─────────┴──────────┘    │ │
│  └──────────────────┘    │                                  │ │
│                          │ Signals Purchased                 │ │
│                          │ ┌──────────────────────────────┐  │ │
│                          │ │ ip-reputation        $0.0002  │  │ │
│                          │ │ TX: a1b2c3...7890  [Link]   │  │ │
│                          │ │ +4.5% fraud ↑     VOI:+2.3%  │  │ │
│                          │ │ Risk Score: 0.85             │  │ │
│                          │ └──────────────────────────────┘  │ │
│                          │                                  │ │
│                          │ Agent Learning                   │ │
│                          │ ┌──────────────────────────────┐  │ │
│                          │ │ ip-reputation                │  │ │
│                          │ │ alpha:3 β:2 trials:5 60%    │  │ │
│                          │ │ [████████████░░░░░░░░░]     │  │ │
│                          │ └──────────────────────────────┘  │ │
│                          │                                  │ │
│                          │ Agent Reasoning                  │ │
│                          │ [1] Initial: prob=0.782...       │ │
│                          │ [2] Zone: RISKY...              │ │
│                          └──────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 Visualizaciones Implementadas

| Visualización | Descripción | Archivo |
|---------------|-------------|---------|
| Probability Bar | Barra animada con gradiente de color según decisión | index.html |
| Confidence Interval | Intervalo con marcador de punto estimado | index.html |
| Risk Zone Badge | Indicador visual RISKY/SAFE/AMBIGUOUS | index.html |
| Signal Cards | Cards con TX hash, badges VOI/Bandit, datos | app.js |
| Stellar Expert Links | Links a transacciones en explorer | app.js |
| Bandit Progress Bars | Barras de E[success] por señal | app.js |
| Decision Reasoning | Trace numerado del proceso de decisión | app.js |

---

## 6. Estructura de Archivos

```
fraud-agent/
├── .env                          # STELLAR_SECRET (secretos)
├── requirements.txt              # Python dependencies
├── README.md                     # Documentación
│
├── model/
│   ├── predict.py               # XGBoost + Conformal Prediction
│   └── train.py                 # Training pipeline
│
├── agent/
│   ├── agent.py                 # Core: VOI + Bandit + Decision
│   ├── bandit.py                # Thompson Sampling implementation
│   ├── x402_client.py           # Stellar payment client
│   ├── uncertainty.py           # Uncertainty calculations
│   └── signals.py               # Signal catalog & fetchers
│
├── api/
│   ├── main.py                  # FastAPI endpoints
│   └── schemas.py               # Pydantic models
│
├── server/
│   └── index.js                 # Node.js proxy + x402 server
│
├── frontend/
│   ├── index.html               # Dashboard UI
│   └── app.js                   # Dashboard logic
│
└── data/
    ├── model.joblib             # Trained model
    ├── calibration_data.json    # Calibration set
    └── bandit_state.json        # Bandit alpha/beta values
```

---

## 7. Comandos de Ejecución

```bash
# 1. Windows: Abrir WSL
wsl

# 2. Crear/entrar al virtual environment
cd /mnt/c/Users/bever/Desktop/fraud-agent
python -m venv venv
source venv/Scripts/activate

# 3. Instalar dependencias Python
pip install -r requirements.txt

# 4. Verificar .env con secrets
cat .env

# 5. Entrar al directorio del proyecto (para imports relativos)
cd /mnt/c/Users/bever/Desktop/fraud-agent

# 6. Terminal 1: Iniciar servidor Node.js (con x402 simplificado)
node server/index.js

# 7. Terminal 2: Iniciar FastAPI
uvicorn api.main:app --reload --port 8000

# 8. Abrir navegador
# http://localhost:3000
```

---

## 8. Logros Técnicos

| Logro | Descripción | Estado |
|-------|-------------|--------|
| XGBoost calibrado | Probabilidades estadísticamente válidas | ✅ |
| Conformal Prediction | Intervalos de confianza 95% | ✅ |
| VOI calculation | Decisión óptima de compra de señales | ✅ |
| Thompson Sampling | Aprendizaje adaptativo de señales | ✅ |
| x402 con Stellar | Micropagos reales en testnet | ✅ |
| XLM nativo | Sin trustlines, sin issuers problemáticos | ✅ |
| Dashboard interactivo | Visualización completa + presets | ✅ |
| Links Stellar Expert | TX hashes clickeables | ✅ |
| Estado persistente | Bandit y modelo se guardan a disco | ✅ |

---

## 9. Limitaciones y Problemas Conocidos

| Problema | Impacto | Solución Actual |
|----------|---------|-----------------|
| Facilitator público no soporta XLM testnet | No se puede usar servidor x402 estándar | Servidor propio simplificado que verifica hash |
| USDC issuer inválido | No se pueden usar stablecoins en testnet | Usar XLM nativo directamente |
| Synthetic dataset | El modelo se entrena con datos simulados | Para producción, entrenar con datos reales |
| Sin persistencia de decisiones | El bandit guarda estado entre requests | Bandit state se guarda en JSON |

---

## 10. Pasos Siguientes

### 10.1 Corto Plazo (para demo del hackathon)

1. **Probar flujo completo**
   - Correr servers
   - Evaluar transacción "Uncertain" preset
   - Verificar que se compran señales
   - Confirmar TX hashes en Stellar Expert testnet

2. **Grabar demo en video**
   - 30-60 segundos mostrando el dashboard
   - Transacción clara → aprobación inmediata
   - Transacción ambigua → compra de señales
   - Mostrar link de Stellar Expert

3. **Limpiar código**
   - Agregar comments donde falten
   - Verificar que no haya hardcoded secrets en código

### 10.2 Medio Plazo (post-hackathon)

1. **Integrar Signal Providers reales**
   - IP Reputation: API de IPQualityScore o similar
   - Device Fingerprint: Servicio como FingerprintJS
   - TX Velocity: Webhook de blockchain analytics

2. **Dashboard mejorado**
   - Gráfico de aprendizaje del bandit en tiempo real
   - Historial de decisiones
   - Costos acumulados

3. **Producción**
   - Mover a Stellar Mainnet
   - Usar servidor x402 con facilitator propio
   - Integrar con wallets reales (Albedo, Freighter)

### 10.3 Largo Plazo

1. **Señales dinámicas**
   - Nuevos signals se agregan automáticamente al bandit
   - Descubrimiento de signals via marketplace

2. **Multi-chain**
   - Extender a Solana, Ethereum via x402
   - Agregar señales on-chain cross-chain

3. **Modelo auto-actualizable**
   - Retraining semanal con nuevo data
   - A/B testing de thresholds

---

## 11. Glosario

| Término | Definición |
|---------|------------|
| **XGBoost** | Algoritmo de gradient boosting para clasificación |
| **CalibratedClassifierCV** | Técnica para calibrar probabilidades de modelos |
| **Conformal Prediction** | Framework para intervalos de predicción con garantía |
| **Thompson Sampling** | Algoritmo de bandit multi-armed para exploración/explotación |
| **Value of Information (VOI)** | Valor esperado de reducir incertidumbre |
| **x402** | Protocolo de micropagos HTTP |
| **Stellar Testnet** | Red de pruebas de Stellar (no dinero real) |
| **Confidence Interval** | Rango donde se espera que esté el valor real |
| **False Positive** | Bloquear transacción legítima |
| **False Negative** | Aprobar transacción fraudulenta |

---

## 12. Agradecimientos y Referencias

- Stellar Development Foundation — por el hackathon y documentación de x402
- XGBoost — por el modelo de ML
- Sklearn — por CalibratedClassifierCV
- Stellar SDK Python — por la integración con Stellar
- Conformal Prediction — por la teoría de intervalos

---

*Documento generado: 8 de Abril 2026*
*Proyecto: FraudSignal Agent — Stellar Hacks 2026*
