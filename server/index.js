import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { paymentMiddlewareFromConfig } from "@x402/express";
import { ExactStellarScheme, STELLAR_TESTNET_CAIP2 } from "@x402/stellar";

dotenv.config({ path: "../.env" });

const app = express();
const PORT = process.env.PORT || 3000;
const STELLAR_ADDRESS =
  process.env.STELLAR_ADDRESS ||
  "GBXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";

// Enable CORS so FastAPI and the frontend can call this server
app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: [
      "Content-Type",
      "Authorization",
      "X-PAYMENT",
      "X-PAYMENT-RESPONSE",
      "payment-signature",
    ],
  })
);

app.use(express.json());

// ─── x402 Payment Middleware (v2.8.0 — Stellar testnet) ──────────────────────
// paymentMiddlewareFromConfig(routes, facilitatorClients, schemes, ...)
// - routes: map of "METHOD /path" → { accepts: [...] }
// - facilitatorClients: null → uses default public facilitator
// - schemes: register ExactStellarScheme for stellar:testnet
app.use(
  paymentMiddlewareFromConfig(
    // Route definitions
    {
      "GET /signal/ip-reputation": {
        accepts: [
          {
            scheme: "exact",
            price: "$0.001",
            network: STELLAR_TESTNET_CAIP2,
            payTo: STELLAR_ADDRESS,
          },
        ],
        description: "IP reputation score for fraud detection",
      },
      "GET /signal/device-history": {
        accepts: [
          {
            scheme: "exact",
            price: "$0.003",
            network: STELLAR_TESTNET_CAIP2,
            payTo: STELLAR_ADDRESS,
          },
        ],
        description: "Device fingerprint history and risk score",
      },
      "GET /signal/tx-velocity": {
        accepts: [
          {
            scheme: "exact",
            price: "$0.002",
            network: STELLAR_TESTNET_CAIP2,
            payTo: STELLAR_ADDRESS,
          },
        ],
        description: "Transaction velocity analysis for the last 24h",
      },
    },
    // facilitatorClients — null uses the default public x402 facilitator
    null,
    // schemes — register the Stellar exact-payment scheme
    [
      {
        network: STELLAR_TESTNET_CAIP2,
        server: ExactStellarScheme,
      },
    ]
  )
);

// ─── Signal Endpoints ────────────────────────────────────────────────────────

/**
 * GET /signal/ip-reputation  — $0.001
 * Returns IP-based reputation signals.
 */
app.get("/signal/ip-reputation", (req, res) => {
  const scenarios = [
    {
      blacklisted: false,
      vpn: false,
      country: "AR",
      risk_score: 0.08,
      isp: "Fibertel",
      abuse_reports_30d: 0,
      fraud_probability_adjustment: 0.02,
    },
    {
      blacklisted: false,
      vpn: true,
      country: "BR",
      risk_score: 0.55,
      isp: "Unknown VPN Provider",
      abuse_reports_30d: 12,
      fraud_probability_adjustment: 0.14,
    },
    {
      blacklisted: true,
      vpn: true,
      country: "RU",
      risk_score: 0.95,
      isp: "TOR Exit Node",
      abuse_reports_30d: 47,
      fraud_probability_adjustment: 0.24,
    },
    {
      blacklisted: false,
      vpn: false,
      country: "NG",
      risk_score: 0.72,
      isp: "MTN Nigeria",
      abuse_reports_30d: 28,
      fraud_probability_adjustment: 0.18,
    },
  ];
  const data = scenarios[Math.floor(Math.random() * scenarios.length)];
  res.json({
    signal: "ip-reputation",
    version: "1.0",
    timestamp: new Date().toISOString(),
    data,
    cost_usd: 0.001,
    utility_score: 0.72,
  });
});

/**
 * GET /signal/device-history  — $0.003
 * Returns device fingerprint history and risk.
 */
app.get("/signal/device-history", (req, res) => {
  const scenarios = [
    {
      seen_before: true,
      linked_accounts: 1,
      fraud_flag: false,
      device_age_days: 365,
      risk_tier: "LOW",
      previous_fraud_flags: 0,
      fraud_probability_adjustment: 0.0,
    },
    {
      seen_before: false,
      linked_accounts: 3,
      fraud_flag: false,
      device_age_days: 2,
      risk_tier: "MEDIUM",
      previous_fraud_flags: 1,
      fraud_probability_adjustment: 0.08,
    },
    {
      seen_before: false,
      linked_accounts: 7,
      fraud_flag: true,
      device_age_days: 1,
      risk_tier: "HIGH",
      previous_fraud_flags: 4,
      fraud_probability_adjustment: 0.32,
    },
    {
      seen_before: true,
      linked_accounts: 2,
      fraud_flag: false,
      device_age_days: 90,
      risk_tier: "LOW",
      previous_fraud_flags: 0,
      fraud_probability_adjustment: 0.01,
    },
  ];
  const data = scenarios[Math.floor(Math.random() * scenarios.length)];
  res.json({
    signal: "device-history",
    version: "1.0",
    timestamp: new Date().toISOString(),
    data,
    cost_usd: 0.003,
    utility_score: 0.88,
  });
});

/**
 * GET /signal/tx-velocity  — $0.002
 * Returns transaction velocity metrics for the past 24 hours.
 */
app.get("/signal/tx-velocity", (req, res) => {
  const scenarios = [
    {
      transactions_1h: 1,
      transactions_last_24h: 3,
      avg_amount: 150,
      velocity_score: 0.06,
      anomaly_detected: false,
      declined_transactions_24h: 0,
      fraud_probability_adjustment: 0.02,
    },
    {
      transactions_1h: 5,
      transactions_last_24h: 18,
      avg_amount: 890,
      velocity_score: 0.36,
      anomaly_detected: true,
      declined_transactions_24h: 3,
      fraud_probability_adjustment: 0.11,
    },
    {
      transactions_1h: 15,
      transactions_last_24h: 47,
      avg_amount: 3200,
      velocity_score: 0.94,
      anomaly_detected: true,
      declined_transactions_24h: 9,
      fraud_probability_adjustment: 0.28,
    },
    {
      transactions_1h: 2,
      transactions_last_24h: 8,
      avg_amount: 420,
      velocity_score: 0.16,
      anomaly_detected: false,
      declined_transactions_24h: 1,
      fraud_probability_adjustment: 0.05,
    },
  ];
  const data = scenarios[Math.floor(Math.random() * scenarios.length)];
  res.json({
    signal: "tx-velocity",
    version: "1.0",
    timestamp: new Date().toISOString(),
    data,
    cost_usd: 0.002,
    utility_score: 0.81,
  });
});

// ─── Health check (no payment required) ─────────────────────────────────────
app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "FraudSignal x402 Server",
    stellar_address: STELLAR_ADDRESS,
    signals_available: ["ip-reputation", "device-history", "tx-velocity"],
    timestamp: new Date().toISOString(),
  });
});

// ─── Start server ─────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🚀 FraudSignal x402 Server running on http://localhost:${PORT}`);
  console.log(`💳 Stellar address: ${STELLAR_ADDRESS}`);
  console.log(`\nAvailable signals (x402 protected):`);
  console.log(`  GET /signal/ip-reputation   → $0.001`);
  console.log(`  GET /signal/device-history  → $0.003`);
  console.log(`  GET /signal/tx-velocity     → $0.002`);
  console.log(`\nHealth check: http://localhost:${PORT}/health\n`);
});
