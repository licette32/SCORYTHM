/**
 * frontend/app.js
 * ===============
 * FraudSignal Agent — Dashboard logic.
 *
 * Functions:
 *   submitTransaction()  — reads form, calls FastAPI, renders result
 *   renderResult(data)   — populates all result UI elements
 *   renderSignals(sigs)  — renders the signal purchase cards
 *   loadPreset(name)     — fills the form with preset values
 *   setToggle(field, val)— handles binary toggle buttons
 *   checkApiHealth()     — polls /health and updates status dot
 */

"use strict";

const API_URL = "http://localhost:8000";

// ─── Preset transaction scenarios ────────────────────────────────────────────
const PRESETS = {
  fraud: {
    amount: 2500.0,
    hour: 3,
    country_mismatch: 1,
    new_account: 1,
    device_age_days: 2.0,
    transactions_last_24h: 28,
  },
  legit: {
    amount: 45.0,
    hour: 14,
    country_mismatch: 0,
    new_account: 0,
    device_age_days: 365.0,
    transactions_last_24h: 2,
  },
  uncertain: {
    amount: 50.0,
    hour: 0,
    country_mismatch: 1,
    new_account: 0,
    device_age_days: 30.0,
    transactions_last_24h: 5,
  },
};

// ─── Decision metadata ────────────────────────────────────────────────────────
const DECISION_META = {
  FRAUD: {
    icon: "🚨",
    sub: "Transaction flagged as fraudulent — block recommended",
  },
  LEGITIMATE: {
    icon: "✅",
    sub: "Transaction appears legitimate — approve",
  },
  UNCERTAIN: {
    icon: "⚠️",
    sub: "Insufficient confidence — manual review recommended",
  },
};

// ─── Load preset ──────────────────────────────────────────────────────────────
function loadPreset(name) {
  const preset = PRESETS[name];
  if (!preset) return;

  document.getElementById("amount").value = preset.amount;
  document.getElementById("hour").value = preset.hour;
  document.getElementById("device_age_days").value = preset.device_age_days;
  document.getElementById("transactions_last_24h").value = preset.transactions_last_24h;

  setToggle("country_mismatch", preset.country_mismatch);
  setToggle("new_account", preset.new_account);
}

// ─── Binary toggle buttons ────────────────────────────────────────────────────
function setToggle(field, value) {
  document.getElementById(field).value = value;

  const prefix = field === "country_mismatch" ? "cm" : "na";
  document.getElementById(`${prefix}-0`).classList.toggle("active", value === 0);
  document.getElementById(`${prefix}-1`).classList.toggle("active", value === 1);
}

// ─── Read form values ─────────────────────────────────────────────────────────
function readForm() {
  return {
    amount: parseFloat(document.getElementById("amount").value) || 0,
    hour: parseInt(document.getElementById("hour").value, 10) || 0,
    country_mismatch: parseInt(document.getElementById("country_mismatch").value, 10),
    new_account: parseInt(document.getElementById("new_account").value, 10),
    device_age_days: parseFloat(document.getElementById("device_age_days").value) || 0,
    transactions_last_24h: parseInt(document.getElementById("transactions_last_24h").value, 10) || 0,
  };
}

// ─── Set loading state ────────────────────────────────────────────────────────
function setLoading(loading) {
  const btn = document.getElementById("eval-btn");
  const spinner = document.getElementById("spinner");
  const btnText = document.getElementById("btn-text");

  btn.disabled = loading;
  spinner.style.display = loading ? "block" : "none";
  btnText.textContent = loading ? "Evaluating…" : "⚡ Evaluate Transaction";
}

// ─── Show / hide error banner ─────────────────────────────────────────────────
function showError(message) {
  const banner = document.getElementById("error-banner");
  const msg = document.getElementById("error-msg");
  msg.textContent = message;
  banner.style.display = "block";
}

function hideError() {
  document.getElementById("error-banner").style.display = "none";
}

// ─── Main: submit transaction ─────────────────────────────────────────────────
async function submitTransaction() {
  hideError();
  setLoading(true);

  const payload = readForm();

  try {
    const response = await fetch(`${API_URL}/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const err = await response.json();
        detail = err.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }

    const data = await response.json();
    renderResult(data);

  } catch (err) {
    let msg = err.message || "Unknown error";

    if (msg.includes("Failed to fetch") || msg.includes("NetworkError") || msg.includes("ERR_CONNECTION_REFUSED")) {
      msg = `Cannot connect to the API at ${API_URL}. Make sure the FastAPI server is running:\n  uvicorn api.main:app --reload --port 8000`;
    }

    showError(msg);
    document.getElementById("result-panel").style.display = "none";
    document.getElementById("empty-state").style.display = "flex";
  } finally {
    setLoading(false);
  }
}

// ─── Render result ────────────────────────────────────────────────────────────
function renderResult(data) {
  const decision = data.decision;
  const meta = DECISION_META[decision] || DECISION_META.UNCERTAIN;

  // Show result panel, hide empty state
  document.getElementById("result-panel").style.display = "block";
  document.getElementById("empty-state").style.display = "none";

  // ── Decision banner ──────────────────────────────────────────────────────
  const banner = document.getElementById("decision-banner");
  banner.className = `decision-banner ${decision}`;

  document.getElementById("decision-icon").textContent = meta.icon;
  document.getElementById("decision-label").textContent = decision;
  document.getElementById("decision-sub").textContent = meta.sub;
  document.getElementById("cost-val").textContent =
    data.total_cost > 0 ? `$${data.total_cost.toFixed(4)}` : "$0.000";

  // ── Probability bar ──────────────────────────────────────────────────────
  const pct = (data.prob_fraud * 100).toFixed(1);
  document.getElementById("prob-value").textContent = `${pct}%`;

  const bar = document.getElementById("prob-bar");
  bar.className = `prob-bar-fill ${decision}`;
  bar.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      bar.style.width = `${pct}%`;
    });
  });

  // ── Confidence Interval ─────────────────────────────────────────────────
  const confLow = data.conf_low || 0;
  const confHigh = data.conf_high || 0;
  const ciFill = document.getElementById("ci-fill");
  const ciMarker = document.getElementById("ci-marker");
  const ciValues = document.getElementById("ci-values");

  ciFill.style.left = `${confLow * 100}%`;
  ciFill.style.width = `${(confHigh - confLow) * 100}%`;
  ciMarker.style.left = `${data.prob_fraud * 100}%`;
  ciValues.textContent = `[${(confLow * 100).toFixed(0)}%, ${(confHigh * 100).toFixed(0)}%]`;

  // ── Risk zone ────────────────────────────────────────────────────────────
  const riskZone = document.getElementById("risk-zone");
  const riskZoneData = data.risk_zone || "UNKNOWN";
  const riskColors = {
    RISKY: "background:rgba(248,81,73,0.1);border:1px solid rgba(248,81,73,0.3);color:var(--red);",
    SAFE: "background:rgba(63,185,80,0.1);border:1px solid rgba(63,185,80,0.3);color:var(--green);",
    AMBIGUOUS: "background:rgba(210,153,34,0.1);border:1px solid rgba(210,153,34,0.3);color:var(--yellow);",
  };
  riskZone.style.cssText = riskColors[riskZoneData] || riskColors.AMBIGUOUS;
  riskZone.textContent = `Zone: ${riskZoneData}`;

  // ── Metrics ──────────────────────────────────────────────────────────────
  document.getElementById("m-uncertainty").textContent =
    (data.uncertainty * 100).toFixed(1) + "%";
  document.getElementById("m-signals").textContent =
    data.signals_purchased.length;
  document.getElementById("m-elapsed").textContent =
    data.elapsed_ms < 1000
      ? `${data.elapsed_ms.toFixed(0)}ms`
      : `${(data.elapsed_ms / 1000).toFixed(2)}s`;

  // ── Signals ──────────────────────────────────────────────────────────────
  document.getElementById("signals-count").textContent = data.signals_purchased.length;
  renderSignals(data.signals_purchased);

  // ── Agent Learning (fetch from health endpoint) ───────────────────────────
  fetchBanditStats();

  // ── Reasoning ────────────────────────────────────────────────────────────
  const reasoningEl = document.getElementById("reasoning-text");
  const steps = data.reasoning.split(" | ");
  reasoningEl.innerHTML = steps
    .map((step, i) => `<span class="step">[${i + 1}]</span> ${escapeHtml(step)}`)
    .join("\n");
}

// ─── Fetch and render bandit stats ────────────────────────────────────────────
async function fetchBanditStats() {
  const section = document.getElementById("learning-section");
  const container = document.getElementById("bandit-stats");

  try {
    const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) throw new Error("Failed");

    const data = await res.json();
    const banditData = data.bandit_state;

    if (!banditData || Object.keys(banditData).length === 0) {
      section.style.display = "block";
      container.innerHTML = `<div class="no-learning">No learning data yet — agent will learn as it evaluates transactions.</div>`;
      return;
    }

    section.style.display = "block";
    container.innerHTML = Object.entries(banditData).map(([signalName, stats]) => {
      const expectedValue = (stats.expected_value * 100).toFixed(1);
      return `
        <div class="bandit-card">
          <div class="bandit-signal-name">${escapeHtml(signalName)}</div>
          <div class="bandit-stats">
            <div class="bandit-stat">alpha: <strong>${stats.alpha}</strong></div>
            <div class="bandit-stat">beta: <strong>${stats.beta}</strong></div>
            <div class="bandit-stat">trials: <strong>${stats.n_trials}</strong></div>
            <div class="bandit-stat">E[success]: <strong>${expectedValue}%</strong></div>
          </div>
          <div class="bandit-bar-track">
            <div class="bandit-bar-fill" style="width:${expectedValue}%;"></div>
          </div>
        </div>
      `;
    }).join("");

  } catch (_) {
    section.style.display = "none";
  }
}

// ─── Render signals ───────────────────────────────────────────────────────────
function renderSignals(signals) {
  const container = document.getElementById("signals-list");

  if (!signals || signals.length === 0) {
    container.innerHTML = `<div class="no-signals">
      Model was confident — no signals purchased.
    </div>`;
    return;
  }

  container.innerHTML = signals.map((sig) => {
    const adj = sig.prob_adjustment || 0;
    const adjClass = adj > 0.001 ? "pos" : adj < -0.001 ? "neg" : "neu";
    const adjText = adj > 0.001
      ? `+${(adj * 100).toFixed(2)}% fraud ↑`
      : adj < -0.001
      ? `${(adj * 100).toFixed(2)}% fraud ↓`
      : "→ no change";

    const offlineBadge = sig.offline_simulation
      ? `<span class="offline-badge">📡 simulated</span>`
      : "";

    const voiBadge = sig.voi !== undefined
      ? `<span class="signal-voi">VOI: ${sig.voi >= 0 ? "+" : ""}${(sig.voi * 100).toFixed(1)}%</span>`
      : "";

    const banditBadge = sig.bandit_priority !== undefined
      ? `<span class="signal-bandit">prio: ${sig.bandit_priority.toFixed(2)}</span>`
      : "";

    const errorNote = sig.error
      ? `<div style="font-size:0.7rem;color:var(--yellow);margin-top:6px;">⚠️ ${escapeHtml(sig.error)}</div>`
      : "";

    const txHash = sig.tx_hash && sig.tx_hash !== 'N/A' ? sig.tx_hash : null;
    const txSection = txHash
      ? `<div class="signal-tx-section">
           <div class="signal-tx"><span>TX:</span> <a href="https://stellar.expert/explorer/testnet/tx/${txHash}" target="_blank" rel="noopener">${escapeHtml(txHash.slice(0, 16))}...${escapeHtml(txHash.slice(-8))}</a></div>
           <a href="https://stellar.expert/explorer/testnet/tx/${txHash}" target="_blank" rel="noopener" class="signal-stellar-link">
             <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
               <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3"/>
             </svg>
             View on Stellar Expert
           </a>
         </div>`
      : '';

    const dataHtml = renderSignalData(sig.signal_name, sig.data);

    return `
      <div class="signal-card">
        <div class="signal-header">
          <span class="signal-name">${escapeHtml(sig.signal_name)}</span>
          <span class="signal-cost">$${sig.cost_usd.toFixed(3)}</span>
          ${voiBadge}
          ${banditBadge}
          ${offlineBadge}
          <span class="signal-adj ${adjClass}">${adjText}</span>
        </div>
        ${txSection}
        ${dataHtml}
        ${errorNote}
      </div>
    `;
  }).join("");
}

// ─── Render signal data fields ────────────────────────────────────────────────
function renderSignalData(signalName, data) {
  if (!data) return "";

  // Pick the most relevant fields per signal type
  const fieldSets = {
    "ip-reputation": [
      ["risk_score", "Risk Score"],
      ["is_vpn", "VPN"],
      ["is_tor", "Tor"],
      ["country_code", "Country"],
      ["abuse_reports_30d", "Abuse Reports"],
    ],
    "device-history": [
      ["device_age_days", "Device Age (days)"],
      ["previous_fraud_flags", "Fraud Flags"],
      ["risk_tier", "Risk Tier"],
      ["unique_accounts_linked", "Linked Accounts"],
    ],
    "tx-velocity": [
      ["transactions_last_24h", "Tx / 24h"],
      ["velocity_score", "Velocity Score"],
      ["declined_transactions_24h", "Declined"],
      ["anomaly_detected", "Anomaly"],
    ],
  };

  const fields = fieldSets[signalName] || Object.keys(data).slice(0, 6).map(k => [k, k]);

  const items = fields
    .filter(([key]) => data[key] !== undefined)
    .map(([key, label]) => {
      let val = data[key];
      if (typeof val === "boolean") val = val ? "Yes" : "No";
      if (typeof val === "number") val = Number.isInteger(val) ? val : val.toFixed(3);
      return `<div class="signal-datum">${escapeHtml(label)}<strong>${escapeHtml(String(val))}</strong></div>`;
    });

  if (items.length === 0) return "";
  return `<div class="signal-data">${items.join("")}</div>`;
}

// ─── Escape HTML ──────────────────────────────────────────────────────────────
function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ─── API health check ─────────────────────────────────────────────────────────
async function checkApiHealth() {
  const dot = document.getElementById("api-status");
  const infoEl = document.getElementById("model-info-content");

  try {
    const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const data = await res.json();
      dot.className = "online";
      dot.title = "API online";

      if (data.model_loaded && data.model_metrics) {
        const m = data.model_metrics;
        infoEl.innerHTML = `
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div>
              <div style="font-family:var(--mono);font-size:0.95rem;font-weight:700;color:var(--green);">
                ${m.roc_auc ? (m.roc_auc * 100).toFixed(1) + "%" : "N/A"}
              </div>
              <div style="font-size:0.68rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;">ROC-AUC</div>
            </div>
            <div>
              <div style="font-family:var(--mono);font-size:0.95rem;font-weight:700;color:var(--accent);">
                ${m.average_precision ? (m.average_precision * 100).toFixed(1) + "%" : "N/A"}
              </div>
              <div style="font-size:0.68rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;">Avg Precision</div>
            </div>
          </div>
          <div style="margin-top:10px;font-size:0.72rem;color:var(--text-dim);">
            XGBoost + CalibratedClassifierCV (isotonic) · 50k synthetic samples
          </div>
        `;
      } else if (!data.model_loaded) {
        infoEl.innerHTML = `<span style="color:var(--yellow);">⚠️ Model not loaded — will train on first request</span>`;
      }
    } else {
      throw new Error("Non-OK response");
    }
  } catch (_) {
    dot.className = "offline";
    dot.title = "API offline";
    infoEl.innerHTML = `
      <span style="color:var(--red);">API offline</span>
      <div style="margin-top:6px;font-size:0.72rem;color:var(--text-dim);">
        Start with: <code style="font-family:var(--mono);color:var(--accent);">uvicorn api.main:app --reload --port 8000</code>
      </div>
    `;
  }
}

// ─── Init ─────────────────────────────────────────────────────────────────────
(function init() {
  // Check API health on load and every 10 seconds
  checkApiHealth();
  setInterval(checkApiHealth, 10_000);

  // Allow Enter key to submit
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey) {
      const active = document.activeElement;
      if (active && active.tagName === "INPUT") {
        submitTransaction();
      }
    }
  });
})();
