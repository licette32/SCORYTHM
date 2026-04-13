# Scorythm Agent — Informe Funcional

## 1. Resumen Ejecutivo

**Scorythm Agent** es un sistema de detección de fraude en transacciones financieras que utiliza aprendizaje automático calibrado, cuantificación de incertidumbre y un agente de decisión que aprende a comprar señales de información de forma autónoma mediante micropagos en la red Stellar.

El proyecto demuestra una arquitectura completa de agente de IA que:
- Evalúa transacciones con confianza estadísticamente garantizada
- Decide dinámicamente cuándo comprar información adicional
- Aprende qué señales son más útiles mediante Thompson Sampling
- Realiza pagos reales en Stellar Testnet via x402
- Explica sus decisiones en lenguaje natural usando Claude API

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
5. **Explica sus decisiones** — en lenguaje natural para humanos no técnicos

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
│              FRAUD AGENT (Servidor)                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │  FastAPI    │→ │    Agente    │→ │  XGBoost + CI   │ │
│  │  (Python)   │  │   (Python)   │  │  (Conformal)    │ │
│  └─────────────┘  └──────────────┘  └─────────────────┘ │
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
│  ┌──────────────┐ ┌───────────────┐ ┌───────────────────┐ │
│  │ IP Reputation│ │ Device History│ │ TX Velocity        │ │
│  │ ($0.001)     │ │ ($0.002)      │ │ ($0.003)          │ │
│  └──────────────┘ └───────────────┘ └───────────────────┘ │
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
                       │ HTTP (REST + SSE Streaming)
┌─────────────────────▼───────────────────────────┐
│              API LAYER (Python/FastAPI)          │
│         api/main.py                              │
│         - POST /evaluate (sync)                  │
│         - POST /evaluate-stream (SSE streaming)   │
│         - GET  /health                          │
└─────────────────────┬───────────────────────────┘
                       │
┌─────────────────────▼───────────────────────────┐
│              FRAUD AGENT (Python)                 │
│         agent/agent.py                           │
│         - evaluate_transaction()                 │
│         - evaluate_transaction_stream()          │
│         agent/explainer.py                       │
│         - explain_decision() → Claude API        │
└────────┬───────────────────────────┬─────────────┘
         │                           │
┌────────▼────────┐     ┌───────────▼──────────────┐
│  FRAUD AGENT    │     │   SIGNAL PROVIDERS        │
│  agent/agent.py │     │   agent/x402_client.py  │
│                 │     │                           │
│  - predict()   │←────→│  - ip-reputation         │
│  - evaluate()  │     │  - device-history         │
│  - stream()     │     │  - tx-velocity            │
└────────┬────────┘     └───────────────────────────┘
         │
┌────────▼──────────────────────────────┐
│  MODEL LAYER                          │
│  model/predict.py                      │
│  model/train.py                        │
│                                       │
│  - XGBoost calibrado                   │
│  - Conformal Prediction                │
│  - Confidence Intervals               │
└───────────────────────────────────────┘
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
│    → produce 100 predictions                   │
│ 4. Conformal Prediction:                        │
│    - conf_low  = percentile(predictions, 2.5)   │
│    - conf_high = percentile(predictions, 97.5)  │
│ 5. uncertainty = conf_high - conf_low           │
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

**Problema resuelto**: El issuer de USDC en testnet no es válido en Horizon testnet, causando errores en `@x402/stellar`.

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
       │  2. Construir TransactionBuilder               │
       │     - source: GBULNFYD... (agent address)   │
       │     - destination: GB2FGBR... (provider)     │
       │     - amount: 0.001 XLM (~0.1 USD)          │
       │     - memo: "sig:ip-reputation"              │
       │                                              │
       │  3. Firmar con secret                        │
       │     - sign() → SHA256                        │
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
       │  7. Request signal with header:              │
       │     payment-signature: <base64>              │
       │─────────────────────────────────────────────→
       │                                              │
       │  8. Signal returned                          │
       │◄─────────────────────────────────────────────
```

#### 3.6.3 Claves de Testnet

```
Address: GBULNFYDNJNDW2HLRJWVWQDYVPP5PDEJ56POOQXACD575GHQXQMGKF4S
Secret:  SCPEQ.....
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
│  Step 2: RISK ZONE ASSESSMENT                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │ if conf_low > 0.5:        RISKY (block)         │  │
│  │ elif conf_high < 0.2:     SAFE (approve)        │  │
│  │ else:                     AMBIGUOUS             │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  Step 3: IF AMBIGUOUS → BUY SIGNALS                   │
│  ┌─────────────────────────────────────────────────┐  │
│  │ max 2 signals per evaluation                     │  │
│  │ for each signal:                                │  │
│  │   1. VOI = compute_voi(prob, utility, cost)     │  │
│  │   2. Priority = bandit.sample()                  │  │
│  │   3. adjusted = VOI * Priority                   │  │
│  │   4. Buy signal with highest adjusted VOI        │  │
│  │   5. Apply probability adjustment               │  │
│  │   6. Break if no longer uncertain               │  │
│  │   7. Update bandit with reward                   │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  Step 4: FINAL DECISION                               │
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

### 3.8 Explicabilidad con Claude API

El agente utiliza Claude (Anthropic) para generar explicaciones en lenguaje natural de sus decisiones.

#### 3.8.1 Archivo: `agent/explainer.py`

```python
async def explain_decision(result: dict) -> str | None:
    """
    Llama a Claude API para explicar la decisión del agente.
    Si ANTHROPIC_API_KEY no está configurado, retorna None sin error.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    
    system = """You are a fraud analyst assistant for SCORYTHM. 
    Your job is to explain, in 2-3 clear sentences, why an AI agent 
    made a specific fraud decision. Be concise and factual."""
    
    user = f"""The agent evaluated a transaction:
    Decision: {result['decision']}
    Initial prob: {result['initial_prob_fraud']:.1%}
    Final prob: {result['prob_fraud']:.1%}
    Signals purchased: {len(result['signals_purchased'])}
    ...
    Explain why the agent made this decision in 2-3 sentences."""
    
    # Llama a Claude y retorna explicación
    ...
```

#### 3.8.2 Características

- **Silencioso**: Si no hay API key, el sistema funciona sin explicación
- **Máximo 150 tokens**: Respuesta rápida
- **Sin errores visibles**: Las excepciones se capturan y no afectan el flujo
- **2-3 oraciones**: Conciso y factual

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
      "alpha": 1,
      "beta": 13,
      "n_trials": 14,
      "expected_value": 0.071
    },
    "device-history": {
      "alpha": 1,
      "beta": 20,
      "n_trials": 21,
      "expected_value": 0.048
    },
    "tx-velocity": {
      "alpha": 1,
      "beta": 18,
      "n_trials": 19,
      "expected_value": 0.053
    }
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
    }
  ],
  "total_cost": 0.0008,
  "elapsed_ms": 234.1,
  "explanation": "The agent flagged this transaction as uncertain..."
}
```

### 4.3 `POST /evaluate-stream`

**Descripción**: Versión streaming del endpoint `/evaluate`. Envía Server-Sent Events (SSE) para proporcionar actualizaciones en tiempo real.

**Eventos transmitidos:**

| Tipo | Descripción |
|------|-------------|
| `step_start` | Indica que un paso comenzó |
| `step_complete` | Un paso se completó con datos |
| `signal_start` | Compra de señal iniciada |
| `signal_complete` | Señal comprada exitosamente |
| `explanation` | Texto de Claude (si está configurado) |
| `done` | Todos los pasos completados |

**Ejemplo de flujo:**
```
data: {"type": "step_start", "step": "model", "message": "Running XGBoost model..."}
data: {"type": "step_complete", "step": "model", "data": {"prob_fraud": 0.48, ...}}
data: {"type": "step_start", "step": "zone", "message": "Assessing risk zone: AMBIGUOUS"}
data: {"type": "step_complete", "step": "zone", "data": {"risk_zone": "AMBIGUOUS", ...}}
data: {"type": "signal_start", "round": 1, "signal_name": "ip-reputation", ...}
data: {"type": "signal_complete", "round": 1, "signal": {...}, "current_prob": 0.52}
data: {"type": "explanation", "text": "The agent flagged this..."}
data: [DONE]
```

---

## 5. Dashboard "Proof of Value"

El dashboard está diseñado para responder en <10 segundos:
1. ¿El agente mejoró la decisión?
2. ¿Valió la pena pagar por las señales?
3. ¿Cuánto dinero se ahorró?

### 5.1 Layout del Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER: SCORYTHM + Tagline                                     │
├─────────────────────────────────────────────────────────────────┤
│  INPUT FORM                    │  Evidence Flow                  │
│  [Fraud] [Legit] [Uncertain] │  Baseline → Signal → Final    │
│                                │  48.5%  → 52.8% → 58.7%      │
│  Amount: $2500               │  AMBIGUOUS    +4.3%   FRAUD ✓ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  DECISION TIMELINE (Step Line Chart)                             │
│  ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●                       │
│  Baseline    signal #1     signal #2     Final                   │
│  48.5%       52.8%         58.7%        58.7%                 │
└─────────────────────────────────────────────────────────────────┘

┌────────────────────────────┐  ┌────────────────────────────────┐
│  ⚖ COUNTERFACTUAL         │  │  $ ECONOMIC IMPACT             │
│                            │  │                                │
│  WITH AGENT  WITHOUT AGENT│  │  Amount at risk:  $2,500.00   │
│  58.7%        48.5%      │  │  Loss Avoided:    $2,499.99  │
│  FRAUD ✓      LEGITIMATE ✗│  │  Agent Cost:           $0.004 │
│                            │  │  NET SAVINGS:     +$2,499.99  │
│  ✓ IMPROVED DECISION      │  │                                │
└────────────────────────────┘  └────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  WHY THIS SIGNAL:                                                │
│  Selected due to highest VOI (0.3055) under uncertainty.        │
│  Signal increased fraud probability by +4.3%.                      │
│  TX: 949aad26209a... [View on Stellar Expert]                    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Panel Counterfactual

Muestra la diferencia entre:
- **Con agente**: Decisión final con señales compradas
- **Sin agente**: Decisión que habría tomado el modelo base

Si las decisiones difieren, muestra "IMPROVED DECISION" con badge verde.

### 5.3 Panel Economic Impact

Calcula el valor real del agente:
- **Amount at risk**: El monto de la transacción
- **Loss Avoided**: Si se bloqueó un fraude, cuánto se evitó perder
- **Agent Cost**: Lo que costaron las señales
- **Net Savings**: Loss Avoided - Agent Cost

### 5.4 Evidence Flow

Flujo visual que muestra:
1. **Baseline**: Probabilidad inicial del modelo
2. **Signal #1**: Impacto de la primera señal comprada
3. **Signal #2**: Impacto de la segunda señal comprada
4. **Final**: Decisión final con badge ✓/✗

### 5.5 Signal Cards con "WHY"

Cada señal comprada muestra:
- Costo y ajuste de probabilidad
- TX hash en Stellar (verificable)
- **WHY THIS SIGNAL**: Explicación de por qué fue seleccionada
  - VOI score
  - Bandit priority
  - Impacto en la probabilidad

### 5.6 Visualizaciones Implementadas

| Visualización | Descripción | Estado |
|---------------|-------------|--------|
| Evidence Flow | Flujo numérico Baseline → Signals → Final | ✅ |
| Step Line Chart | Gráfico SVG con saltos de probabilidad | ✅ |
| Counterfactual | Con/Sin agente lado a lado | ✅ |
| Economic Impact | Savings basados en monto de transacción | ✅ |
| Signal Cards + WHY | Cards con razón de selección | ✅ |
| Bandit Bars | Barras + usos + tendencias | ✅ |
| Audit Trace | Collapsible con TX links | ✅ |
| Explanation Card | Texto de Claude en lenguaje natural | ✅ |

---

## 6. Estructura de Archivos

```
fraud-agent/
├── .env                          # STELLAR_SECRET, ANTHROPIC_API_KEY
├── .env.example                  # Template de variables de entorno
├── requirements.txt              # Python dependencies
├── run_server.py                 # Quick launcher para FastAPI
├── README.md                     # Documentación
│
├── model/
│   ├── predict.py               # XGBoost + Conformal Prediction
│   └── train.py                 # Training pipeline
│
├── agent/
│   ├── agent.py                 # Core: VOI + Bandit + Decision + Stream
│   ├── bandit.py                # Thompson Sampling implementation
│   ├── x402_client.py           # Stellar payment client (XLM native)
│   ├── uncertainty.py           # Uncertainty calculations
│   └── explainer.py             # Claude API integration
│
├── api/
│   ├── main.py                  # FastAPI endpoints (/evaluate, /evaluate-stream)
│   └── schemas.py               # Pydantic models
│
├── server/
│   └── index.js                 # Node.js: x402 server + frontend serving
│
├── frontend/
│   ├── index.html               # Dashboard UI
│   └── app.js                   # Dashboard logic (streaming + fallback)
│
└── data/
    ├── model.joblib             # Trained model
    ├── calibration_data.json    # Calibration set
    └── bandit_state.json        # Bandit alpha/beta values
```

---

## 7. Variables de Entorno

```bash
# ─── Stellar Testnet ─────────────────────────────────────────
STELLAR_ADDRESS=GBXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
STELLAR_SECRET=SXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
SERVER_URL=http://localhost:3000
FACILITATOR_URL=https://x402.org/facilitator

# ─── Claude (Anthropic) — para explicaciones ✨NUEVO ──────────
ANTHROPIC_API_KEY=sk-ant-...  # Opcional
```

---

## 8. Comandos de Ejecución

```bash
# 1. Crear/entrar al virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar .env
cp .env.example .env
# Editar .env con:
#   - STELLAR_SECRET (obligatorio)
#   - ANTHROPIC_API_KEY (opcional, para explicaciones)

# 4. Terminal 1: FastAPI
python run_server.py
# O: uvicorn api.main:app --reload --port 8000

# 5. Terminal 2: Servidor Node.js (sirve frontend en :3000)
node server/index.js

# 6. Abrir navegador
# http://localhost:3000
```

---

## 9. Logros Técnicos

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
| Streaming en vivo | Pipeline visible durante evaluación | ✅ |
| Live Probability Tracker | Gráfico animado en tiempo real | ✅ |
| Explicabilidad con Claude | Decisiones en lenguaje natural | ✅ |
| Fallback automático | Funciona sin streaming si falla | ✅ |
| Bandit mejorado | Barras + usos + tendencias | ✅ |
| run_server.py | Launcher rápido para FastAPI | ✅ |

---

## 10. Limitaciones y Problemas Conocidos

| Problema | Impacto | Solución Actual |
|----------|---------|-----------------|
| USDC issuer no funciona en testnet | No se pueden usar stablecoins | Usar XLM nativo directamente |
| Synthetic dataset | El modelo se entrena con datos simulados | Para producción, entrenar con datos reales |
| ANTHROPIC_API_KEY opcional | Sin API key no hay explicación | Sistema funciona sin explicación (sin error) |
| Facilitator público no soportado | Servidor x402 estándar no funciona | Servidor propio que verifica hash de TX |

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
| **SSE (Server-Sent Events)** | Protocolo para enviar actualizaciones en tiempo real |
| **Claude API** | API de Anthropic para generación de lenguaje natural |
| **Confidence Interval** | Rango donde se espera que esté el valor real |
| **False Positive** | Bloquear transacción legítima |
| **False Negative** | Aprobar transacción fraudulenta |

---

## 12. Agradecimientos y Referencias

- Stellar Development Foundation — por el hackathon y documentación de x402
- XGBoost — por el modelo de ML
- Sklearn — por CalibratedClassifierCV
- Stellar SDK Python — por la integración con Stellar
- Anthropic — por Claude API para explicaciones
- Conformal Prediction — por la teoría de intervalos

---

*Documento actualizado: 12 de Abril 2026*
*Proyecto: Scorythm Agent — Stellar Hacks 2026*
