import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config({ path: '../.env' });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

// ── Serve frontend static files ───────────────────────────────────────────────
const FRONTEND_DIR = path.join(__dirname, '..', 'frontend');
app.use(express.static(FRONTEND_DIR));

app.get('/', (req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, 'index.html'));
});

// ── Configuration ─────────────────────────────────────────────────────────────
const PORT = parseInt(process.env.PORT || '3000', 10);
const STELLAR_ADDRESS = process.env.STELLAR_ADDRESS || 'GBULNFYDNJNDW2HLRJWVWQDYVPP5PDEJ56POOQXACD575GHQXQMGKF4S';
// Server wallet — receives payments (different from agent wallet that pays)
const SERVER_STELLAR_ADDRESS = process.env.SERVER_STELLAR_ADDRESS || 'GDNIWY6TITEJEUC7O7TTGQME45C2WOILUIVDJZICPC47R43NU5UFJA75';
const IPQS_API_KEY = process.env.IPQS_API_KEY || '';

// USDC issuer on Stellar testnet
const USDC_ISSUER_TESTNET = 'GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5';

// Prices
const PRICE_XLM_STROOPS = 10000;       // 0.001 XLM in stroops
const PRICE_USDC = '0.001';            // 0.001 USDC (6 decimal places on Stellar)

// ── Payment tracking ──────────────────────────────────────────────────────────
// Map: txHash -> { timestamp, asset }
const paidTransactions = new Map();

// Cleanup old paid transactions every 10 minutes (prevent memory leak)
setInterval(() => {
  const cutoff = Date.now() - 10 * 60 * 1000;
  for (const [hash, entry] of paidTransactions.entries()) {
    if (entry.timestamp < cutoff) paidTransactions.delete(hash);
  }
}, 10 * 60 * 1000);

// ── Payment middleware ────────────────────────────────────────────────────────
function requirePayment(req, res, next) {
  const paymentHeader = req.headers['payment-signature'];

  if (!paymentHeader) {
    // Return 402 with both XLM and USDC payment options
    const paymentReq = {
      x402Version: 2,
      error: 'Payment required',
      resource: {
        url: `http://localhost:${PORT}${req.path}`,
        description: 'Fraud signal data — Scorythm Agent',
      },
      accepts: [
        // Option 1: USDC (preferred for hackathon demo)
        {
          scheme: 'exact',
          network: 'stellar:testnet',
          amount: PRICE_USDC,
          asset: USDC_ISSUER_TESTNET,
          asset_code: 'USDC',
          payTo: SERVER_STELLAR_ADDRESS,
          maxTimeoutSeconds: 300,
          description: '0.001 USDC on Stellar testnet',
        },
        // Option 2: Native XLM (fallback)
        {
          scheme: 'exact',
          network: 'stellar:testnet',
          amount: PRICE_XLM_STROOPS.toString(),
          asset: 'native',
          asset_code: 'XLM',
          payTo: SERVER_STELLAR_ADDRESS,
          maxTimeoutSeconds: 300,
          description: '0.001 XLM (native) on Stellar testnet',
        },
      ],
    };

    const encoded = Buffer.from(JSON.stringify(paymentReq)).toString('base64');
    res.set('PAYMENT-REQUIRED', encoded);
    res.set('X-Payment-Options', 'USDC,XLM');
    return res.status(402).json({
      error: 'Payment required',
      accepts: paymentReq.accepts,
    });
  }

  // Verify the payment header
  try {
    const payload = JSON.parse(Buffer.from(paymentHeader, 'base64').toString());
    const txHash = payload.hash;

    if (!txHash || typeof txHash !== 'string' || txHash.length !== 64) {
      return res.status(402).json({ error: 'Invalid payment signature: bad tx hash' });
    }

    // Prevent replay attacks: each tx hash can only be used once
    if (paidTransactions.has(txHash)) {
      return res.status(402).json({ error: 'Payment already used (replay attack prevention)' });
    }

    const asset = payload.asset || 'native';
    paidTransactions.set(txHash, { timestamp: Date.now(), asset });

    // Attach payment info to request for downstream use
    req.paymentInfo = {
      txHash,
      asset,
      network: payload.network || 'stellar:testnet',
    };

    next();
  } catch (e) {
    return res.status(402).json({ error: 'Invalid payment signature format' });
  }
}

// ── Signal data pools ─────────────────────────────────────────────────────────
// Each signal has 3 pools: high-risk, low-risk, neutral
// risk_hint query param selects the pool: "high" | "low" | "neutral" (default)
const SIGNAL_DATA = {
  'ip-reputation': {
    high: [
      { is_vpn: true,  is_tor: true,  country: 'RU', risk_score: 0.95, blacklisted: true  },
      { is_vpn: true,  is_tor: false, country: 'CN', risk_score: 0.82, blacklisted: true  },
      { is_vpn: true,  is_tor: true,  country: 'KP', risk_score: 0.98, blacklisted: true  },
    ],
    low: [
      { is_vpn: false, is_tor: false, country: 'AR', risk_score: 0.08, blacklisted: false },
      { is_vpn: false, is_tor: false, country: 'DE', risk_score: 0.05, blacklisted: false },
      { is_vpn: false, is_tor: false, country: 'US', risk_score: 0.12, blacklisted: false },
    ],
    neutral: [
      { is_vpn: false, is_tor: false, country: 'US', risk_score: 0.30, blacklisted: false },
      { is_vpn: true,  is_tor: false, country: 'BR', risk_score: 0.60, blacklisted: false },
      { is_vpn: false, is_tor: false, country: 'MX', risk_score: 0.42, blacklisted: false },
      { is_vpn: true,  is_tor: false, country: 'IN', risk_score: 0.55, blacklisted: false },
    ],
  },
  'device-history': {
    high: [
      { seen_before: false, fraud_flag: true,  linked_accounts: 7, device_age_days:   1 },
      { seen_before: false, fraud_flag: true,  linked_accounts: 9, device_age_days:   0 },
      { seen_before: false, fraud_flag: false, linked_accounts: 6, device_age_days:   2 },
    ],
    low: [
      { seen_before: true,  fraud_flag: false, linked_accounts: 1, device_age_days: 365 },
      { seen_before: true,  fraud_flag: false, linked_accounts: 1, device_age_days: 180 },
      { seen_before: true,  fraud_flag: false, linked_accounts: 1, device_age_days: 500 },
    ],
    neutral: [
      { seen_before: false, fraud_flag: false, linked_accounts: 2, device_age_days:  45 },
      { seen_before: false, fraud_flag: false, linked_accounts: 3, device_age_days:   2 },
      { seen_before: true,  fraud_flag: false, linked_accounts: 2, device_age_days:  30 },
      { seen_before: false, fraud_flag: false, linked_accounts: 4, device_age_days:  10 },
    ],
  },
  'tx-velocity': {
    high: [
      { transactions_1h: 15, transactions_24h: 47, velocity_flag: true,  anomaly_detected: true  },
      { transactions_1h: 12, transactions_24h: 38, velocity_flag: true,  anomaly_detected: true  },
      { transactions_1h: 18, transactions_24h: 55, velocity_flag: true,  anomaly_detected: true  },
    ],
    low: [
      { transactions_1h:  1, transactions_24h:  2, velocity_flag: false, anomaly_detected: false },
      { transactions_1h:  1, transactions_24h:  3, velocity_flag: false, anomaly_detected: false },
      { transactions_1h:  2, transactions_24h:  4, velocity_flag: false, anomaly_detected: false },
    ],
    neutral: [
      { transactions_1h:  3, transactions_24h:  8, velocity_flag: false, anomaly_detected: false },
      { transactions_1h:  5, transactions_24h: 12, velocity_flag: true,  anomaly_detected: false },
      { transactions_1h:  7, transactions_24h: 18, velocity_flag: true,  anomaly_detected: false },
      { transactions_1h:  4, transactions_24h: 10, velocity_flag: false, anomaly_detected: false },
    ],
  },
};

function pickSample(signalName, riskHint) {
  const pools = SIGNAL_DATA[signalName];
  if (!pools) return {};
  const hint = ['high', 'low', 'neutral'].includes(riskHint) ? riskHint : 'neutral';
  const pool = pools[hint];
  return pool[Math.floor(Math.random() * pool.length)];
}

// ── Signal endpoints ──────────────────────────────────────────────────────────
app.get('/signal/ip-reputation', requirePayment, async (req, res) => {
  const riskHint = req.query.risk_hint || 'neutral';
  const ip = req.query.ip || '8.8.8.8';

  // Try real IPQualityScore API first
  if (IPQS_API_KEY) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      const response = await fetch(
        `https://ipqualityscore.com/api/json/ip/${IPQS_API_KEY}/${ip}`,
        { signal: controller.signal }
      );
      clearTimeout(timeout);
      const ipData = await response.json();

      if (ipData && ipData.success !== false) {
        return res.json({
          blacklisted: (ipData.fraud_score || 0) > 75,
          vpn: ipData.vpn || false,
          tor: ipData.tor || false,
          country: ipData.country_code || 'Unknown',
          risk_score: (ipData.fraud_score || 0) / 100,
          proxy: ipData.proxy || false,
          fraud_score: ipData.fraud_score || 0,
          risk_hint: (ipData.fraud_score || 0) > 75 ? 'high'
                   : (ipData.fraud_score || 0) > 40 ? 'neutral' : 'low',
          _source: 'ipqualityscore',
          _ip_queried: ip,
          _payment: req.paymentInfo,
          _asset: req.paymentInfo?.asset === 'native' ? 'XLM' : 'USDC',
        });
      }
    } catch (err) {
      console.warn(`[ip-reputation] IPQualityScore failed (${err.message}), using fallback`);
    }
  }

  // Fallback: use simulated data pool
  const data = pickSample('ip-reputation', riskHint);
  res.json({
    ...data,
    _source: 'simulated',
    _payment: req.paymentInfo,
    _asset: req.paymentInfo?.asset === 'native' ? 'XLM' : 'USDC',
    _risk_hint: riskHint,
  });
});

app.get('/signal/device-history', requirePayment, (req, res) => {
  const riskHint = req.query.risk_hint || 'neutral';
  const data = pickSample('device-history', riskHint);
  res.json({
    ...data,
    _payment: req.paymentInfo,
    _asset: req.paymentInfo?.asset === 'native' ? 'XLM' : 'USDC',
    _risk_hint: riskHint,
  });
});

app.get('/signal/tx-velocity', requirePayment, (req, res) => {
  const riskHint = req.query.risk_hint || 'neutral';
  const data = pickSample('tx-velocity', riskHint);
  res.json({
    ...data,
    _payment: req.paymentInfo,
    _asset: req.paymentInfo?.asset === 'native' ? 'XLM' : 'USDC',
    _risk_hint: riskHint,
  });
});

// ── Health endpoint ───────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    agent_wallet: STELLAR_ADDRESS,
    server_wallet: SERVER_STELLAR_ADDRESS,
    network: 'stellar:testnet',
    ipqs_enabled: !!IPQS_API_KEY,
    payment_options: [
      { asset: 'USDC', amount: PRICE_USDC, issuer: USDC_ISSUER_TESTNET, payTo: SERVER_STELLAR_ADDRESS },
      { asset: 'XLM',  amount: `${PRICE_XLM_STROOPS} stroops (0.001 XLM)`, payTo: SERVER_STELLAR_ADDRESS },
    ],
    paid_count: paidTransactions.size,
  });
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`🚀 Scorythm x402 Server on http://localhost:${PORT}`);
  console.log(`🤖 Agent wallet (pays):   ${STELLAR_ADDRESS}`);
  console.log(`🏦 Server wallet (receives): ${SERVER_STELLAR_ADDRESS}`);
  console.log(`🔍 IPQualityScore: ${IPQS_API_KEY ? 'ENABLED' : 'disabled (fallback mode)'}`);
  console.log(`💵 Accepts: USDC (${PRICE_USDC}) or XLM (${PRICE_XLM_STROOPS} stroops)`);
  console.log(`📊 Dashboard: http://localhost:${PORT}`);
});
