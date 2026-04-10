import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import crypto from 'crypto';

dotenv.config({ path: '../.env' });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

// Serve frontend static files
const FRONTEND_DIR = path.join(__dirname, '..', 'frontend');
app.use(express.static(FRONTEND_DIR));

const STELLAR_ADDRESS = process.env.STELLAR_ADDRESS || 'GBULNFYDNJNDW2HLRJWVWQDYVPP5PDEJ56POOQXACD575GHQXQMGKF4S';
const PORT = 3000;

// Simple payment verification (no facilitator needed)
const paidTransactions = new Map(); // txHash -> timestamp

// Amount in stroops (0.001 XLM)
const PRICE_STROOPS = 10000;

// Middleware to check for payment
function requirePayment(req, res, next) {
  const paymentHeader = req.headers['payment-signature'];
  
  if (!paymentHeader) {
    // Return 402 with payment info
    const paymentReq = {
      x402Version: 2,
      error: "Payment required",
      resource: { url: `http://localhost:${PORT}${req.path}`, description: "Signal data" },
      accepts: [{
        scheme: "exact",
        network: "stellar:testnet",
        amount: PRICE_STROOPS.toString(),
        asset: "native",
        payTo: STELLAR_ADDRESS,
        maxTimeoutSeconds: 300,
      }]
    };
    const encoded = Buffer.from(JSON.stringify(paymentReq)).toString('base64');
    res.set('PAYMENT-REQUIRED', encoded);
    return res.status(402).json({});
  }
  
  // Verify the payment
  try {
    const payload = JSON.parse(Buffer.from(paymentHeader, 'base64').toString());
    const txHash = payload.hash;
    
    if (!txHash || typeof txHash !== 'string' || txHash.length !== 64) {
      return res.status(402).json({ error: 'Invalid payment signature' });
    }
    
    // Mark as paid
    paidTransactions.set(txHash, Date.now());
    next();
  } catch (e) {
    return res.status(402).json({ error: 'Invalid payment signature format' });
  }
}

app.get('/signal/ip-reputation', requirePayment, (req, res) => {
  const data = [
    { blacklisted: false, vpn: false, country: "AR", risk_score: 0.1 },
    { blacklisted: false, vpn: true,  country: "BR", risk_score: 0.6 },
    { blacklisted: true,  vpn: true,  country: "RU", risk_score: 0.95 },
  ];
  res.json(data[Math.floor(Math.random() * data.length)]);
});

app.get('/signal/device-history', requirePayment, (req, res) => {
  const data = [
    { seen_before: true,  linked_accounts: 1, fraud_flag: false, device_age_days: 180 },
    { seen_before: false, linked_accounts: 3, fraud_flag: false, device_age_days: 2 },
    { seen_before: false, linked_accounts: 7, fraud_flag: true,  device_age_days: 1 },
  ];
  res.json(data[Math.floor(Math.random() * data.length)]);
});

app.get('/signal/tx-velocity', requirePayment, (req, res) => {
  const data = [
    { transactions_1h: 1,  transactions_24h: 3,  velocity_flag: false },
    { transactions_1h: 5,  transactions_24h: 12, velocity_flag: true  },
    { transactions_1h: 15, transactions_24h: 47, velocity_flag: true  },
  ];
  res.json(data[Math.floor(Math.random() * data.length)]);
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', address: STELLAR_ADDRESS, network: 'stellar:testnet' });
});

// Serve index.html for root
app.get('/', (req, res) => {
  res.sendFile(path.join(FRONTEND_DIR, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 FraudSignal x402 Server on http://localhost:${PORT}`);
  console.log(`💳 Paying to: ${STELLAR_ADDRESS}`);
  console.log(`📊 Dashboard: http://localhost:${PORT}`);
});
