import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { paymentMiddlewareFromConfig } from '@x402/express';
import { ExactStellarScheme } from '@x402/stellar/exact/server';
import { HTTPFacilitatorClient } from '@x402/core/server';

dotenv.config({ path: '../.env' });

const app = express();
app.use(cors());
app.use(express.json());

const STELLAR_ADDRESS = process.env.STELLAR_ADDRESS || 'GCLAY5XQLFMI42PVN37XVZWS5LP3NX2RRCYNCKPXFVEBEY7ZYEN5ECCR';
const PORT = 3000;
const FACILITATOR_URL = 'https://www.x402.org/facilitator';

const routes = {
  "GET /signal/ip-reputation": {
    accepts: [{ scheme: "exact", price: "$0.001", network: "stellar:testnet", payTo: STELLAR_ADDRESS }],
    description: "IP reputation signal",
  },
  "GET /signal/device-history": {
    accepts: [{ scheme: "exact", price: "$0.003", network: "stellar:testnet", payTo: STELLAR_ADDRESS }],
    description: "Device history signal",
  },
  "GET /signal/tx-velocity": {
    accepts: [{ scheme: "exact", price: "$0.002", network: "stellar:testnet", payTo: STELLAR_ADDRESS }],
    description: "Transaction velocity signal",
  },
};

const facilitatorClient = new HTTPFacilitatorClient({ url: FACILITATOR_URL });

app.use(paymentMiddlewareFromConfig(
  routes,
  facilitatorClient,
  [{ network: "stellar:testnet", server: new ExactStellarScheme() }]
));

app.get('/signal/ip-reputation', (req, res) => {
  const data = [
    { blacklisted: false, vpn: false, country: "AR", risk_score: 0.1 },
    { blacklisted: false, vpn: true,  country: "BR", risk_score: 0.6 },
    { blacklisted: true,  vpn: true,  country: "RU", risk_score: 0.95 },
  ];
  res.json(data[Math.floor(Math.random() * data.length)]);
});

app.get('/signal/device-history', (req, res) => {
  const data = [
    { seen_before: true,  linked_accounts: 1, fraud_flag: false, device_age_days: 180 },
    { seen_before: false, linked_accounts: 3, fraud_flag: false, device_age_days: 2 },
    { seen_before: false, linked_accounts: 7, fraud_flag: true,  device_age_days: 1 },
  ];
  res.json(data[Math.floor(Math.random() * data.length)]);
});

app.get('/signal/tx-velocity', (req, res) => {
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

app.listen(PORT, () => {
  console.log(`🚀 FraudSignal x402 Server on http://localhost:${PORT}`);
  console.log(`💳 Paying to: ${STELLAR_ADDRESS}`);
});
