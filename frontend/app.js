/**
 * frontend/app.js — SCORYTHM Fraud Intelligence Agent
 * Muestra en tiempo real cómo el agente analiza, razona y decide.
 */
"use strict";

// API_URL se carga dinámicamente desde /config (inyectado por el servidor Node).
// Fallback a localhost:8000 para desarrollo local.
let API_URL = "http://localhost:8000";

(async function loadConfig() {
  try {
    const res = await fetch("/config", { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const cfg = await res.json();
      if (cfg.apiUrl) API_URL = cfg.apiUrl;
    }
  } catch (_) {
    // En desarrollo local /config puede no existir — usar fallback
  }
})();

// ─── Presets ──────────────────────────────────────────────────────────────────
const PRESETS = {
  // CASO D — Fraude claro (no compra señales, prob > 0.65)
  fraud: {
    amount: 4500.0, hour: 2, country_mismatch: 1, new_account: 1,
    device_age_days: 1.0, transactions_last_24h: 28,
    device_risk_score: 0.90, email_domain_risk: 0.85,
  },
  // CASO A — Legítima clara (no compra señales, prob < 0.35)
  legit: {
    amount: 45.0, hour: 14, country_mismatch: 0, new_account: 0,
    device_age_days: 365.0, transactions_last_24h: 2,
    device_risk_score: 0.10, email_domain_risk: 0.05,
  },
  // CASO B — Ambigua (compra 1-2 señales, prob en zona 0.40-0.60)
  uncertain: {
    amount: 850.0, hour: 23, country_mismatch: 1, new_account: 0,
    device_age_days: 20.0, transactions_last_24h: 12,
    device_risk_score: 0.55, email_domain_risk: 0.50,
  },
};

// ─── DOM helpers ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
function setText(id, t) { const e = $(id); if (e) e.textContent = t; }
function setHTML(id, h) { const e = $(id); if (e) e.innerHTML = h; }
function show(id) { const e = $(id); if (e) e.style.display = ""; }
function hide(id) { const e = $(id); if (e) e.style.display = "none"; }
function showFlex(id) { const e = $(id); if (e) e.style.display = "flex"; }
function showBlock(id) { const e = $(id); if (e) e.style.display = "block"; }
function addClass(id, c) { const e = $(id); if (e) e.classList.add(c); }
function removeClass(id, c) { const e = $(id); if (e) e.classList.remove(c); }
function setClass(id, c) { const e = $(id); if (e) e.className = c; }
function setStyle(id, p, v) { const e = $(id); if (e) e.style[p] = v; }

// ─── Decision helpers ─────────────────────────────────────────────────────────
function decisionFromProb(prob) {
  if (prob >= 0.65) return "FRAUD";
  if (prob <= 0.35) return "LEGITIMATE";
  return "UNCERTAIN";
}

function decisionColor(d) {
  return { FRAUD: "var(--fraud)", LEGITIMATE: "var(--legit)", UNCERTAIN: "var(--uncertain)" }[d] || "var(--text)";
}

function decisionBadgeClass(d) {
  return { FRAUD: "fraud", LEGITIMATE: "legitimate", UNCERTAIN: "uncertain" }[d] || "";
}

function decisionBadgeHTML(d) {
  const cls = decisionBadgeClass(d);
  return `<span class="ev-decision-badge ${cls}">${d}</span>`;
}

function formatMoney(v) {
  if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(1)}K`;
  return `$${v.toFixed(2)}`;
}

function escapeHtml(s) {
  if (!s && s !== 0) return "";
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ─── Controls ─────────────────────────────────────────────────────────────────
function updateHour(val) {
  setText("hour-display", val);
  const e = $("hour-range"); if (e) e.value = val;
}

function setToggle(field, value) {
  const hidden = $(field); if (hidden) hidden.value = value;
  const prefix = field === "country_mismatch" ? "cm" : "na";
  const b0 = $(`${prefix}-0`), b1 = $(`${prefix}-1`);
  if (b0) b0.className = `toggle-btn ${value === 0 ? "off" : ""}`;
  if (b1) b1.className = `toggle-btn ${value === 1 ? "on" : ""}`;
}

function loadPreset(name) {
  const p = PRESETS[name]; if (!p) return;
  const a = $("amount"); if (a) a.value = p.amount;
  updateHour(p.hour);
  const da = $("device_age_days"); if (da) da.value = p.device_age_days;
  const tx = $("transactions_last_24h"); if (tx) tx.value = p.transactions_last_24h;
  setToggle("country_mismatch", p.country_mismatch);
  setToggle("new_account", p.new_account);

  // Load new risk sliders
  const drs = $("device_risk_score");
  if (drs && p.device_risk_score != null) {
    drs.value = p.device_risk_score;
    const drsDisplay = $("drs-display");
    if (drsDisplay) drsDisplay.textContent = parseFloat(p.device_risk_score).toFixed(2);
  }
  const edr = $("email_domain_risk");
  if (edr && p.email_domain_risk != null) {
    edr.value = p.email_domain_risk;
    const edrDisplay = $("edr-display");
    if (edrDisplay) edrDisplay.textContent = parseFloat(p.email_domain_risk).toFixed(2);
  }

  document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
  const btn = $(`btn-${name}`); if (btn) btn.classList.add("active");
}

function readForm() {
  return {
    amount: parseFloat($("amount")?.value) || 0,
    hour: parseInt($("hour-range")?.value, 10) || 0,
    country_mismatch: parseInt($("country_mismatch")?.value, 10) || 0,
    new_account: parseInt($("new_account")?.value, 10) || 0,
    device_age_days: parseFloat($("device_age_days")?.value) || 0,
    transactions_last_24h: parseInt($("transactions_last_24h")?.value, 10) || 0,
    // New fields — use hidden inputs if present, otherwise default to neutral values
    device_risk_score: parseFloat($("device_risk_score")?.value ?? "0.3"),
    email_domain_risk: parseFloat($("email_domain_risk")?.value ?? "0.2"),
  };
}

// ─── Loading / Error ──────────────────────────────────────────────────────────
function setLoading(on) {
  const btn = $("eval-btn"); if (btn) btn.disabled = on;
  setStyle("spinner", "display", on ? "inline-block" : "none");
  setText("btn-text", on ? "ANALYZING..." : "ANALYZE TRANSACTION");
}

function showError(msg) { setHTML("error-msg", escapeHtml(msg)); addClass("error-banner", "visible"); }
function hideError() { removeClass("error-banner", "visible"); }

// ─── Pipeline step helpers ────────────────────────────────────────────────────
function pipelineStepActive(id) {
  const el = $(id); if (!el) return;
  el.className = "pipeline-step active";
}

function pipelineStepDone(id, extraClass) {
  const el = $(id); if (!el) return;
  el.className = `pipeline-step done ${extraClass || ""}`;
}

function pipelineShowBadge(badgeId, text, cls) {
  const b = $(badgeId); if (!b) return;
  b.textContent = text;
  b.className = `step-badge ${cls}`;
  b.style.display = "";
}

function pipelineShowDetail(detailId, html) {
  const d = $(detailId); if (!d) return;
  d.innerHTML = html;
  d.style.display = "";
}

// ─── Reset UI ─────────────────────────────────────────────────────────────────
function resetUI() {
  hide("empty-state");
  setClass("decision-card", "decision-card");
  showBlock("pipeline-card");
  addClass("pipeline-live", "active");

  // Reset pipeline steps
  ["ps-model", "ps-zone", "ps-signals", "ps-decision"].forEach(id => {
    const el = $(id); if (el) el.className = "pipeline-step";
  });
  ["ps-model-badge", "ps-zone-badge", "ps-signals-badge", "ps-decision-badge"].forEach(id => hide(id));
  ["ps-model-detail", "ps-zone-detail", "ps-signals-detail", "ps-decision-detail"].forEach(id => hide(id));
  setHTML("ps-signals-list", "");

  // Reset prob section
  removeClass("prob-section", "visible");
  hide("chart-card");
  setClass("evidence-card", "evidence-card");
  hide("cf-card");
  hide("econ-card");
  setClass("signals-section", "signals-section");
  setClass("reasoning-section", "reasoning-section");
  setClass("audit-card", "audit-card");
  removeClass("explanation-card", "visible");
  setClass("stats-row", "stats-row");
  hide("ci-row");

  // Reset evidence flow
  hide("ev-sig1-arrow"); hide("ev-sig1-row");
  hide("ev-sig2-arrow"); hide("ev-sig2-row");
  setText("ev-baseline-prob", "—"); setHTML("ev-baseline-decision", "");
  setText("ev-sig1-prob", "—"); setHTML("ev-sig1-decision", ""); setHTML("ev-sig1-delta", "");
  setText("ev-sig2-prob", "—"); setHTML("ev-sig2-decision", ""); setHTML("ev-sig2-delta", "");
  setText("ev-final-prob", "—"); setHTML("ev-final-decision", "");
}

// ─── Streaming state ──────────────────────────────────────────────────────────
let _state = {};

// Race condition guard: each evaluation gets a unique ID.
// If a new evaluation starts before the previous one finishes,
// the old one is cancelled and its results are discarded.
let _currentEvalId = 0;
let _currentAbortController = null;

function resetState(payload) {
  _state = {
    initialProb: 0,
    finalProb: 0,
    decision: null,
    signals: [],
    riskZone: null,
    amount: payload.amount || 0,
    totalCost: 0,
    elapsedMs: 0,
    explanation: null,
    reasoning: null,
    confLow: 0,
    confHigh: 0,
    uncertainty: 0,
  };
}

// ─── SUBMIT ───────────────────────────────────────────────────────────────────
async function submitTransaction() {
  hideError();
  const payload = readForm();
  if (!payload.amount || payload.amount <= 0) { showError("Please enter a valid amount."); return; }

  // Cancel any in-flight request (race condition prevention)
  if (_currentAbortController) {
    _currentAbortController.abort();
  }

  // Assign a new evaluation ID — stale callbacks will check this and bail out
  const evalId = ++_currentEvalId;
  _currentAbortController = new AbortController();
  const signal = _currentAbortController.signal;

  setLoading(true);
  resetState(payload);
  resetUI();

  // Guard: returns true if this evaluation has been superseded
  const isStale = () => evalId !== _currentEvalId;

  let streamOk = false;
  try {
    const resp = await fetch(`${API_URL}/evaluate-stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";

    while (true) {
      if (isStale()) { reader.cancel(); break; }

      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() || "";
      for (const line of lines) {
        if (isStale()) break;
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6);
        if (raw === "[DONE]") { streamOk = true; break; }
        try { handleEvent(JSON.parse(raw)); } catch (_) {}
      }
      if (streamOk || isStale()) break;
    }
  } catch (err) {
    if (err.name === "AbortError") {
      // Request was intentionally cancelled — do nothing
      return;
    }
    console.warn("Stream failed:", err.message);
  }

  if (isStale()) return;

  if (!streamOk) {
    // Fallback to /evaluate
    try {
      const resp = await fetch(`${API_URL}/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      if (!isStale()) applyFallbackResult(data);
    } catch (err) {
      if (err.name === "AbortError") return;
      if (!isStale()) {
        let msg = err.message || "Unknown error";
        if (msg.includes("fetch") || msg.includes("ERR_CONNECTION") || msg.includes("Failed to fetch")) {
          msg = "API offline — run: uvicorn api.main:app --reload --port 8000";
        }
        showError(msg);
      }
    }
  }

  if (isStale()) return;

  removeClass("pipeline-live", "active");
  setLoading(false);
  renderFinalUI();
}

// ─── Stream event handler ─────────────────────────────────────────────────────
function handleEvent(ev) {
  switch (ev.type) {
    case "step_start":
      if (ev.step === "model") {
        pipelineStepActive("ps-model");
        pipelineShowBadge("ps-model-badge", "running XGBoost...", "thinking");
      } else if (ev.step === "zone") {
        pipelineStepActive("ps-zone");
        pipelineShowBadge("ps-zone-badge", "assessing risk zone...", "thinking");
      } else if (ev.step === "signals") {
        pipelineStepActive("ps-signals");
        pipelineShowBadge("ps-signals-badge", "computing VOI...", "thinking");
      } else if (ev.step === "decision") {
        pipelineStepActive("ps-decision");
        pipelineShowBadge("ps-decision-badge", "deciding...", "thinking");
      }
      break;

    case "step_complete":
      if (ev.step === "model") {
        const d = ev.data;
        _state.initialProb = d.prob_fraud;
        _state.confLow = d.conf_low;
        _state.confHigh = d.conf_high;
        _state.uncertainty = d.uncertainty;
        if (d.amount) _state.amount = d.amount;

        const dec = decisionFromProb(d.prob_fraud);
        const unc = (d.uncertainty * 100).toFixed(0);
        const ci = `[${(d.conf_low * 100).toFixed(1)}%, ${(d.conf_high * 100).toFixed(1)}%]`;

        pipelineStepDone("ps-model");
        pipelineShowBadge("ps-model-badge", "done", "done-badge");
        pipelineShowDetail("ps-model-detail", `
          prob_fraud: <span class="highlight">${(d.prob_fraud * 100).toFixed(2)}%</span>
          &nbsp;|&nbsp; uncertainty: <span class="highlight">${unc}%</span>
          &nbsp;|&nbsp; CI: <span class="dim">${ci}</span>
        `);

        // Update evidence baseline
        setText("ev-baseline-prob", `${(d.prob_fraud * 100).toFixed(1)}%`);
        setHTML("ev-baseline-decision", decisionBadgeHTML(dec));

      } else if (ev.step === "zone") {
        const zone = ev.data.risk_zone || "AMBIGUOUS";
        _state.riskZone = zone;

        const zoneMap = {
          CONFIDENT_FRAUD: { label: "CONFIDENT FRAUD", cls: "fraud-badge", stepCls: "fraud-step" },
          CONFIDENT_LEGIT: { label: "CONFIDENT LEGIT", cls: "legit-badge", stepCls: "legit-step" },
          AMBIGUOUS: { label: "AMBIGUOUS — signals needed", cls: "uncertain-badge", stepCls: "uncertain-step" },
        };
        const zm = zoneMap[zone] || zoneMap.AMBIGUOUS;

        pipelineStepDone("ps-zone", zm.stepCls);
        pipelineShowBadge("ps-zone-badge", zm.label, zm.cls);
        pipelineShowDetail("ps-zone-detail", `
          Zone: <span class="highlight">${zone}</span>
          &nbsp;|&nbsp; ${zone === "AMBIGUOUS"
            ? '<span class="uncertain-val">→ Agent will purchase signals to reduce uncertainty</span>'
            : '<span class="dim">→ Confidence sufficient, no signals needed</span>'}
        `);

      } else if (ev.step === "decision") {
        const d = ev.data;
        _state.finalProb = d.prob_fraud;
        _state.decision = d.decision;
        _state.totalCost = d.total_cost || 0;
        _state.elapsedMs = d.elapsed_ms || 0;
        _state.reasoning = d.reasoning;
        if (d.signals_purchased) _state.signals = d.signals_purchased;

        const decCls = { FRAUD: "fraud-badge fraud-step", LEGITIMATE: "legit-badge legit-step", UNCERTAIN: "uncertain-badge uncertain-step" }[d.decision] || "";
        pipelineStepDone("ps-decision", decCls.split(" ")[1]);
        pipelineShowBadge("ps-decision-badge", d.decision, decCls.split(" ")[0]);
        pipelineShowDetail("ps-decision-detail", `
          Final prob: <span class="highlight">${(d.prob_fraud * 100).toFixed(2)}%</span>
          &nbsp;|&nbsp; Cost: <span class="highlight">$${(d.total_cost || 0).toFixed(4)}</span>
          &nbsp;|&nbsp; Time: <span class="dim">${(d.elapsed_ms || 0).toFixed(0)}ms</span>
        `);
      }
      break;

    case "signal_start":
      pipelineStepActive("ps-signals");
      pipelineShowBadge("ps-signals-badge", `buying ${ev.signal_name}...`, "thinking");
      break;

    case "signal_complete": {
      const sig = ev.signal;
      _state.signals.push(sig);

      const adj = sig.prob_adjustment || 0;
      const adjPct = (adj * 100).toFixed(2);
      const adjCls = adj > 0.001 ? "up" : adj < -0.001 ? "down" : "neutral";
      const adjText = adj > 0.001 ? `+${adjPct}% ↑ fraud risk` : adj < -0.001 ? `${adjPct}% ↓ fraud risk` : "no change";

      const dataHtml = renderSigDataMini(sig.signal_name, sig.data);

      const item = document.createElement("div");
      item.className = "signal-purchase-item";
      item.innerHTML = `
        <div class="sig-purchase-header">
          <span class="sig-purchase-name">${escapeHtml(sig.signal_name)}</span>
          <span class="sig-purchase-cost">$${sig.cost_usd.toFixed(3)}</span>
        </div>
        ${dataHtml}
        <div class="sig-purchase-delta ${adjCls}">${adjText}</div>
      `;
      const list = $("ps-signals-list"); if (list) list.appendChild(item);

      // Update evidence flow
      const round = ev.round || _state.signals.length;
      const probAfter = ev.prob_after || 0;
      const probBefore = ev.prob_before || _state.initialProb;
      const delta = (probAfter - probBefore) * 100;
      const deltaCls = delta > 0.1 ? "up" : delta < -0.1 ? "down" : "neutral";
      const deltaText = `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}%`;

      if (round === 1) {
        show("ev-sig1-arrow"); show("ev-sig1-row");
        setText("ev-sig1-label", sig.signal_name);
        setText("ev-sig1-prob", `${(probAfter * 100).toFixed(1)}%`);
        setHTML("ev-sig1-decision", decisionBadgeHTML(decisionFromProb(probAfter)));
        setHTML("ev-sig1-delta", `<span class="ev-delta-badge ${deltaCls}">${deltaText}</span>`);
      } else if (round === 2) {
        show("ev-sig2-arrow"); show("ev-sig2-row");
        setText("ev-sig2-label", sig.signal_name);
        setText("ev-sig2-prob", `${(probAfter * 100).toFixed(1)}%`);
        setHTML("ev-sig2-decision", decisionBadgeHTML(decisionFromProb(probAfter)));
        setHTML("ev-sig2-delta", `<span class="ev-delta-badge ${deltaCls}">${deltaText}</span>`);
      }
      break;
    }

    case "explanation":
      _state.explanation = ev.text;
      break;

    case "error":
      showError(ev.error || "Streaming error");
      break;
  }
}

// ─── Fallback: apply full result at once ──────────────────────────────────────
function applyFallbackResult(data) {
  _state.initialProb = data.initial_prob_fraud || 0;
  _state.finalProb = data.prob_fraud || 0;
  _state.decision = data.decision;
  _state.signals = data.signals_purchased || [];
  _state.totalCost = data.total_cost || 0;
  _state.elapsedMs = data.elapsed_ms || 0;
  _state.reasoning = data.reasoning;
  _state.explanation = data.explanation;
  _state.confLow = data.conf_low || 0;
  _state.confHigh = data.conf_high || 0;
  _state.uncertainty = data.uncertainty || 0;
  _state.riskZone = data.risk_zone;
  if (data.amount) _state.amount = data.amount;

  // Fill pipeline steps quickly
  const dec = _state.decision;
  pipelineStepDone("ps-model");
  pipelineShowBadge("ps-model-badge", "done", "done-badge");
  pipelineShowDetail("ps-model-detail", `prob_fraud: <span class="highlight">${(_state.initialProb * 100).toFixed(2)}%</span> | uncertainty: <span class="highlight">${(_state.uncertainty * 100).toFixed(0)}%</span>`);

  pipelineStepDone("ps-zone");
  pipelineShowBadge("ps-zone-badge", _state.riskZone || "—", "done-badge");
  pipelineShowDetail("ps-zone-detail", `Zone: <span class="highlight">${_state.riskZone || "—"}</span>`);

  if (_state.signals.length > 0) {
    pipelineStepDone("ps-signals");
    pipelineShowBadge("ps-signals-badge", `${_state.signals.length} signal(s) purchased`, "done-badge");
    _state.signals.forEach(sig => {
      const adj = sig.prob_adjustment || 0;
      const adjCls = adj > 0.001 ? "up" : adj < -0.001 ? "down" : "neutral";
      const adjText = adj > 0.001 ? `+${(adj*100).toFixed(2)}% ↑ fraud risk` : adj < -0.001 ? `${(adj*100).toFixed(2)}% ↓ fraud risk` : "no change";
      const item = document.createElement("div");
      item.className = "signal-purchase-item";
      item.innerHTML = `
        <div class="sig-purchase-header">
          <span class="sig-purchase-name">${escapeHtml(sig.signal_name)}</span>
          <span class="sig-purchase-cost">$${sig.cost_usd.toFixed(3)}</span>
        </div>
        ${renderSigDataMini(sig.signal_name, sig.data)}
        <div class="sig-purchase-delta ${adjCls}">${adjText}</div>
      `;
      const list = $("ps-signals-list"); if (list) list.appendChild(item);
    });
  } else {
    pipelineStepDone("ps-signals");
    pipelineShowBadge("ps-signals-badge", "no signals needed", "done-badge");
    pipelineShowDetail("ps-signals-detail", `<span class="dim">Model confidence sufficient — VOI below threshold</span>`);
  }

  const decCls = { FRAUD: "fraud-step", LEGITIMATE: "legit-step", UNCERTAIN: "uncertain-step" }[dec] || "";
  pipelineStepDone("ps-decision", decCls);
  const decBadgeCls = { FRAUD: "fraud-badge", LEGITIMATE: "legit-badge", UNCERTAIN: "uncertain-badge" }[dec] || "done-badge";
  pipelineShowBadge("ps-decision-badge", dec, decBadgeCls);
  pipelineShowDetail("ps-decision-detail", `Final prob: <span class="highlight">${(_state.finalProb * 100).toFixed(2)}%</span> | Cost: <span class="highlight">$${_state.totalCost.toFixed(4)}</span>`);
}

// ─── Render final UI after streaming/fallback ─────────────────────────────────
function renderFinalUI() {
  const { decision, finalProb, initialProb, signals, totalCost, elapsedMs, amount, confLow, confHigh, uncertainty } = _state;
  if (!decision) return;

  // Decision card
  const ICONS = { FRAUD: "🚨", LEGITIMATE: "✓", UNCERTAIN: "◈" };
  const SUBS = {
    FRAUD: "Transaction flagged — block recommended",
    LEGITIMATE: "Transaction approved — low risk",
    UNCERTAIN: "Insufficient confidence — manual review required",
  };
  const dc = $("decision-card");
  if (dc) {
    dc.className = `decision-card visible ${decision}`;
    setHTML("decision-icon", ICONS[decision] || "?");
    setText("decision-label", decision);
    setText("decision-sub", SUBS[decision] || "");
    setText("cost-chip", `$${totalCost.toFixed(4)} signals`);
    setText("time-chip", elapsedMs < 1000 ? `${elapsedMs.toFixed(0)}ms` : `${(elapsedMs/1000).toFixed(2)}s`);
  }

  // Probability section
  addClass("prob-section", "visible");
  const pn = $("prob-number");
  if (pn) { pn.textContent = `${(finalProb * 100).toFixed(1)}%`; pn.className = `prob-big ${decision}`; }
  const pf = $("prob-fill");
  if (pf) { pf.className = `prob-fill ${decision}`; pf.style.width = `${finalProb * 100}%`; }

  // CI
  if (confLow > 0 || confHigh > 0) {
    show("ci-row");
    setText("ci-low", `${(confLow * 100).toFixed(1)}%`);
    setText("ci-point", `${(finalProb * 100).toFixed(1)}%`);
    setText("ci-high", `${(confHigh * 100).toFixed(1)}%`);
  }

  // Evidence flow final
  addClass("evidence-card", "visible");
  setText("ev-final-prob", `${(finalProb * 100).toFixed(1)}%`);
  setHTML("ev-final-decision", decisionBadgeHTML(decision));

  // Evidence baseline (if not set by streaming)
  if ($("ev-baseline-prob")?.textContent === "—") {
    setText("ev-baseline-prob", `${(initialProb * 100).toFixed(1)}%`);
    setHTML("ev-baseline-decision", decisionBadgeHTML(decisionFromProb(initialProb)));
  }

  // Evidence signals (if not set by streaming)
  signals.forEach((sig, i) => {
    let probAfter = initialProb;
    for (let j = 0; j <= i; j++) probAfter += signals[j].prob_adjustment || 0;
    const probBefore = i === 0 ? initialProb : initialProb + signals.slice(0, i).reduce((s, x) => s + (x.prob_adjustment || 0), 0);
    const delta = (probAfter - probBefore) * 100;
    const deltaCls = delta > 0.1 ? "up" : delta < -0.1 ? "down" : "neutral";
    const deltaText = `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}%`;

    if (i === 0 && $("ev-sig1-prob")?.textContent === "—") {
      show("ev-sig1-arrow"); show("ev-sig1-row");
      setText("ev-sig1-label", sig.signal_name);
      setText("ev-sig1-prob", `${(probAfter * 100).toFixed(1)}%`);
      setHTML("ev-sig1-decision", decisionBadgeHTML(decisionFromProb(probAfter)));
      setHTML("ev-sig1-delta", `<span class="ev-delta-badge ${deltaCls}">${deltaText}</span>`);
    } else if (i === 1 && $("ev-sig2-prob")?.textContent === "—") {
      show("ev-sig2-arrow"); show("ev-sig2-row");
      setText("ev-sig2-label", sig.signal_name);
      setText("ev-sig2-prob", `${(probAfter * 100).toFixed(1)}%`);
      setHTML("ev-sig2-decision", decisionBadgeHTML(decisionFromProb(probAfter)));
      setHTML("ev-sig2-delta", `<span class="ev-delta-badge ${deltaCls}">${deltaText}</span>`);
    }
  });

  // Chart
  renderProbChart(initialProb, finalProb, signals);

  // Counterfactual
  renderCounterfactual(initialProb, finalProb, decision, amount);

  // Economic impact
  renderEconomicImpact(finalProb, decision, totalCost, amount, initialProb);

  // Signals section
  if (signals.length > 0) {
    addClass("signals-section", "visible");
    setText("signals-count", signals.length);
    setHTML("signals-list", signals.map(sig => renderSignalCard(sig, initialProb)).join(""));
  }

  // Explanation
  if (_state.explanation) {
    addClass("explanation-card", "visible");
    setText("explanation-text", _state.explanation);
  }

  // Reasoning
  if (_state.reasoning) {
    addClass("reasoning-section", "visible");
    setText("reasoning-text", _state.reasoning);
  }

  // Audit
  renderAudit(signals, finalProb, decision, totalCost, initialProb, uncertainty);

  // Stats
  const elapsed = elapsedMs < 1000 ? `${elapsedMs.toFixed(0)}ms` : `${(elapsedMs/1000).toFixed(2)}s`;
  setHTML("stats-row", `
    <div class="stat-box"><div class="stat-val">${(uncertainty * 100).toFixed(0)}%</div><div class="stat-lbl">Uncertainty</div></div>
    <div class="stat-box"><div class="stat-val">${signals.length}</div><div class="stat-lbl">Signals</div></div>
    <div class="stat-box"><div class="stat-val">$${totalCost.toFixed(4)}</div><div class="stat-lbl">Signal Cost</div></div>
    <div class="stat-box"><div class="stat-val">${elapsed}</div><div class="stat-lbl">Eval Time</div></div>
  `);
  addClass("stats-row", "visible");

  // Bandit
  fetchBanditStats();
}

// ─── Probability Evolution Chart ──────────────────────────────────────────────
function renderProbChart(initialProb, finalProb, signals) {
  const svg = $("prob-chart-svg");
  if (!svg) return;

  showBlock("chart-card");

  const W = 520, H = 200;
  const padL = 48, padR = 24, padT = 24, padB = 36;
  const cW = W - padL - padR;
  const cH = H - padT - padB;

  // Build data points
  const points = [{ label: "Baseline", prob: initialProb }];
  let running = initialProb;
  signals.forEach(sig => {
    running = Math.max(0, Math.min(1, running + (sig.prob_adjustment || 0)));
    points.push({ label: sig.signal_name.replace(/-/g, " "), prob: running, delta: sig.prob_adjustment });
  });
  if (points.length === 1 || Math.abs(points[points.length - 1].prob - finalProb) > 0.005) {
    points.push({ label: "Final", prob: finalProb });
  }

  const n = points.length;
  const xOf = i => padL + (n === 1 ? cW / 2 : (i / (n - 1)) * cW);
  const yOf = p => padT + (1 - p) * cH;

  let out = `<defs>
    <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#D4A574" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#D4A574"/>
    </linearGradient>
  </defs>`;

  // Zone backgrounds
  out += `<rect x="${padL}" y="${padT}" width="${cW}" height="${yOf(0.65) - padT}" fill="rgba(248,81,73,0.07)"/>`;
  out += `<rect x="${padL}" y="${yOf(0.65)}" width="${cW}" height="${yOf(0.35) - yOf(0.65)}" fill="rgba(210,153,34,0.07)"/>`;
  out += `<rect x="${padL}" y="${yOf(0.35)}" width="${cW}" height="${padT + cH - yOf(0.35)}" fill="rgba(63,185,80,0.07)"/>`;

  // Zone labels
  out += `<text x="${padL + 4}" y="${padT + 12}" fill="rgba(248,81,73,0.5)" font-size="9" font-family="JetBrains Mono,monospace">FRAUD</text>`;
  out += `<text x="${padL + 4}" y="${yOf(0.5) + 4}" fill="rgba(210,153,34,0.5)" font-size="9" font-family="JetBrains Mono,monospace">UNCERTAIN</text>`;
  out += `<text x="${padL + 4}" y="${padT + cH - 4}" fill="rgba(63,185,80,0.5)" font-size="9" font-family="JetBrains Mono,monospace">LEGIT</text>`;

  // Grid lines
  [0, 0.25, 0.5, 0.75, 1].forEach(p => {
    const y = yOf(p);
    out += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#30363D" stroke-width="0.5"/>`;
    out += `<text x="${padL - 4}" y="${y + 4}" fill="#484F58" font-size="9" font-family="JetBrains Mono,monospace" text-anchor="end">${(p * 100).toFixed(0)}%</text>`;
  });

  // Threshold lines
  [0.35, 0.65].forEach(th => {
    const y = yOf(th);
    out += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#D4A574" stroke-width="1" stroke-dasharray="4,3" opacity="0.4"/>`;
  });

  // Line path
  if (n > 1) {
    let path = `M ${xOf(0)} ${yOf(points[0].prob)}`;
    for (let i = 1; i < n; i++) {
      path += ` L ${xOf(i)} ${yOf(points[i - 1].prob)} L ${xOf(i)} ${yOf(points[i].prob)}`;
    }
    out += `<path d="${path}" stroke="url(#lineGrad)" stroke-width="2.5" fill="none" stroke-linecap="round"/>`;
  }

  // Dots, labels, delta annotations
  points.forEach((pt, i) => {
    const cx = xOf(i), cy = yOf(pt.prob);
    const col = pt.prob >= 0.65 ? "#F85149" : pt.prob <= 0.35 ? "#3FB950" : "#D29922";

    out += `<circle cx="${cx}" cy="${cy}" r="5" fill="${col}" stroke="#0F1117" stroke-width="2"/>`;
    out += `<text x="${cx}" y="${cy - 10}" fill="#E6EDF3" font-size="10" font-family="JetBrains Mono,monospace" text-anchor="middle" font-weight="600">${(pt.prob * 100).toFixed(1)}%</text>`;

    // Step label below
    const labelY = H - padB + 14;
    const shortLabel = pt.label.length > 12 ? pt.label.slice(0, 11) + "…" : pt.label;
    out += `<text x="${cx}" y="${labelY}" fill="#484F58" font-size="9" font-family="JetBrains Mono,monospace" text-anchor="middle">${escapeHtml(shortLabel)}</text>`;

    // Delta annotation
    if (pt.delta !== undefined && Math.abs(pt.delta) > 0.001) {
      const dPct = pt.delta * 100;
      const dCol = dPct > 0 ? "#F85149" : "#3FB950";
      const annX = i < n - 1 ? cx + 28 : cx - 28;
      const anchor = i < n - 1 ? "start" : "end";
      out += `<text x="${annX}" y="${cy - 2}" fill="${dCol}" font-size="9" font-family="JetBrains Mono,monospace" text-anchor="${anchor}" font-weight="600">${dPct >= 0 ? "+" : ""}${dPct.toFixed(1)}%</text>`;
    }
  });

  svg.innerHTML = out;
}

// ─── Counterfactual ───────────────────────────────────────────────────────────
function renderCounterfactual(initialProb, finalProb, decision, amount) {
  showBlock("cf-card");

  // "Without Agent" = static system with fixed threshold 0.5 (industry standard)
  // This is what most fraud systems do: no uncertainty quantification, no signal purchase
  const withoutDecision = initialProb >= 0.5 ? "FRAUD" : "LEGITIMATE";

  // Ground truth: use the agent's final decision as reference
  // (the agent has more information after purchasing signals)
  const isFraud = decision === "FRAUD";

  const withoutCorrect = isFraud ? withoutDecision === "FRAUD" : withoutDecision === "LEGITIMATE";
  const withCorrect = true; // Agent's decision is always the informed one

  setText("cf-without-prob", `${(initialProb * 100).toFixed(1)}%`);
  setHTML("cf-without-verdict", `
    <span class="cf-verdict ${decisionBadgeClass(withoutDecision)}">${withoutDecision}</span>
    <span class="${withoutCorrect ? "cf-correct" : "cf-wrong"}">${withoutCorrect ? "✓ matches agent" : "✗ differs from agent"}</span>
  `);

  setText("cf-with-prob", `${(finalProb * 100).toFixed(1)}%`);
  setHTML("cf-with-verdict", `
    <span class="cf-verdict ${decisionBadgeClass(decision)}">${decision}</span>
    <span class="cf-correct">✓ informed by signals</span>
  `);

  if (withoutDecision !== decision) {
    addClass("cf-lift", "visible");
    setText("cf-lift-text", `Agent changed decision: ${withoutDecision} → ${decision} after purchasing signals (static threshold would have been wrong)`);
  } else if (_state.signals && _state.signals.length > 0) {
    addClass("cf-lift", "visible");
    setText("cf-lift-text", `Agent confirmed ${decision} with signal evidence — static threshold agrees but lacks justification`);
  } else {
    removeClass("cf-lift", "visible");
  }
}

// ─── Economic Impact ──────────────────────────────────────────────────────────
function renderEconomicImpact(finalProb, decision, totalCost, amount, initialProb) {
  showBlock("econ-card");

  const isFraud = finalProb > 0.5;
  const withoutDecision = decisionFromProb(initialProb);
  const withoutCorrect = isFraud ? withoutDecision === "FRAUD" : withoutDecision === "LEGITIMATE";
  const withCorrect = isFraud ? decision === "FRAUD" : decision === "LEGITIMATE";

  setText("econ-cost", `$${totalCost.toFixed(4)}`);

  const lossBefore = isFraud && withoutDecision !== "FRAUD" ? amount : 0;
  const lossAfter = isFraud && decision !== "FRAUD" ? amount : 0;

  setText("econ-loss-before", lossBefore > 0 ? formatMoney(lossBefore) : "$0.00");
  setText("econ-loss-after", lossAfter > 0 ? formatMoney(lossAfter) : "$0.00");

  const netValue = lossBefore - lossAfter - totalCost;
  const netEl = $("econ-net-val");
  if (netEl) {
    if (netValue > 0) {
      netEl.textContent = `+${formatMoney(netValue)} SAVED`;
      netEl.style.color = "var(--legit)";
    } else if (netValue < 0) {
      netEl.textContent = `${formatMoney(netValue)} LOST`;
      netEl.style.color = "var(--fraud)";
    } else {
      netEl.textContent = "CORRECT — no loss prevented";
      netEl.style.color = "var(--text-muted)";
    }
  }
}

// ─── Signal card (detailed) ───────────────────────────────────────────────────
function renderSignalCard(sig, initialProb) {
  const adj = sig.prob_adjustment || 0;
  const adjAbs = Math.abs(adj) * 100;
  const adjCls = adj > 0.001 ? "pos" : adj < -0.001 ? "neg" : "neu";
  const adjText = adj > 0.001 ? `+${adjAbs.toFixed(2)}% fraud risk ↑` : adj < -0.001 ? `-${adjAbs.toFixed(2)}% fraud risk ↓` : "no change";

  const txHash = sig.tx_hash && sig.tx_hash !== "N/A" ? sig.tx_hash : null;
  const txUrl = txHash ? `https://stellar.expert/explorer/testnet/tx/${txHash}` : null;

  const voi = sig.voi != null ? sig.voi.toFixed(4) : "—";
  const bandit = (sig.bandit_priority || 0.5).toFixed(2);

  // Detect payment asset from signal data (_asset field injected by server)
  const payAsset = sig.data?._asset || sig.payment_info?.asset_code || (sig.offline_simulation ? "SIM" : "XLM");
  const assetBadge = payAsset === "USDC"
    ? `<span style="font-family:var(--font-mono);font-size:0.65rem;padding:2px 6px;border-radius:4px;background:rgba(88,166,255,0.12);color:var(--blue);border:1px solid rgba(88,166,255,0.25);">USDC</span>`
    : payAsset === "SIM"
    ? `<span style="font-family:var(--font-mono);font-size:0.65rem;padding:2px 6px;border-radius:4px;background:rgba(255,255,255,0.05);color:var(--text-dim);border:1px solid var(--border);">SIMULATED</span>`
    : `<span style="font-family:var(--font-mono);font-size:0.65rem;padding:2px 6px;border-radius:4px;background:rgba(212,165,116,0.1);color:var(--copper);border:1px solid rgba(212,165,116,0.25);">XLM</span>`;

  return `
    <div class="signal-card">
      <div class="signal-header">
        <span class="signal-name">${escapeHtml(sig.signal_name)}</span>
        <span class="signal-cost-badge">$${sig.cost_usd.toFixed(3)}</span>
        ${assetBadge}
        <span class="signal-impact ${adjCls}">${adjText}</span>
      </div>
      ${txHash ? `<a class="signal-tx-link" href="${txUrl}" target="_blank" rel="noopener">↗ Stellar TX (${payAsset}): ${txHash.slice(0, 10)}...${txHash.slice(-6)}</a>` : ""}
      <div class="signal-data-grid">${renderSigDataFull(sig.signal_name, sig.data)}</div>
      <div class="signal-why">
        <span class="signal-why-label">Why purchased:</span>
        VOI=${voi} (highest value of information under uncertainty). Bandit priority=${bandit}.
        Signal ${adj > 0 ? "increased" : adj < 0 ? "decreased" : "did not change"} fraud probability by ${adjAbs.toFixed(2)}%.
        Payment: ${payAsset} on Stellar testnet.
      </div>
    </div>
  `;
}

// ─── Signal data renderers ────────────────────────────────────────────────────
const SIG_FIELDS = {
  "ip-reputation": [["risk_score","Risk Score"],["is_vpn","VPN"],["is_tor","Tor"],["country","Country"],["blacklisted","Blacklisted"]],
  "device-history": [["seen_before","Seen Before"],["fraud_flag","Fraud Flag"],["linked_accounts","Linked Accts"],["device_age_days","Device Age"]],
  "tx-velocity": [["transactions_1h","Tx/1h"],["transactions_24h","Tx/24h"],["velocity_flag","Velocity Flag"],["anomaly_detected","Anomaly"]],
};

function renderSigDataFull(name, data) {
  if (!data) return "";
  const fields = SIG_FIELDS[name] || Object.keys(data).slice(0, 5).map(k => [k, k]);
  return fields.filter(([k]) => data[k] !== undefined).map(([k, label]) => {
    let v = data[k];
    if (typeof v === "boolean") v = v ? "Yes" : "No";
    if (typeof v === "number") v = Number.isInteger(v) ? v : v.toFixed(3);
    return `<div class="signal-datum">${escapeHtml(label)}<strong>${escapeHtml(String(v))}</strong></div>`;
  }).join("");
}

function renderSigDataMini(name, data) {
  if (!data) return "";
  const fields = SIG_FIELDS[name] || Object.keys(data).slice(0, 3).map(k => [k, k]);
  const items = fields.filter(([k]) => data[k] !== undefined).slice(0, 3).map(([k, label]) => {
    let v = data[k];
    if (typeof v === "boolean") v = v ? "Yes" : "No";
    if (typeof v === "number") v = Number.isInteger(v) ? v : v.toFixed(2);
    return `<span class="sig-datum">${escapeHtml(label)}: <strong>${escapeHtml(String(v))}</strong></span>`;
  });
  return `<div class="sig-purchase-data">${items.join("")}</div>`;
}

// ─── Audit Trail ──────────────────────────────────────────────────────────────
function renderAudit(signals, finalProb, decision, totalCost, initialProb, uncertainty) {
  addClass("audit-card", "visible");
  const ts = new Date().toISOString();
  const uid = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2);
  setText("audit-meta", `Decision ID: ${uid} | Timestamp: ${ts}`);

  const lines = [
    `<span class="log-step">[1] MODEL</span>   prob_fraud=<span class="log-key">${initialProb.toFixed(4)}</span>  uncertainty=<span class="log-key">${uncertainty.toFixed(4)}</span>`,
    `<span class="log-step">[2] ZONE</span>    ${_state.riskZone || "AMBIGUOUS"}`,
  ];

  let running = initialProb;
  signals.forEach((sig, i) => {
    running = Math.max(0, Math.min(1, running + (sig.prob_adjustment || 0)));
    const col = (sig.prob_adjustment || 0) > 0 ? "log-fraud" : "log-legit";
    lines.push(`<span class="log-step">[${i + 3}] SIGNAL</span>  ${sig.signal_name}  cost=<span class="log-key">$${sig.cost_usd.toFixed(4)}</span>  voi=<span class="log-key">${sig.voi?.toFixed(4) || "—"}</span>  Δprob=<span class="${col}">${((sig.prob_adjustment || 0) * 100).toFixed(2)}%</span>  prob_after=<span class="log-key">${running.toFixed(4)}</span>`);
  });

  const decCol = decision === "FRAUD" ? "log-fraud" : decision === "LEGITIMATE" ? "log-legit" : "log-key";
  lines.push(`<span class="log-step">[${signals.length + 3}] DECISION</span> <span class="${decCol}">${decision}</span>  prob=<span class="log-key">${finalProb.toFixed(4)}</span>  total_cost=<span class="log-key">$${totalCost.toFixed(4)}</span>`);

  setHTML("audit-log", lines.map(l => `<div>${l}</div>`).join(""));

  const txLinks = signals.filter(s => s.tx_hash && s.tx_hash !== "N/A").map(s =>
    `<a class="audit-tx-link" href="https://stellar.expert/explorer/testnet/tx/${s.tx_hash}" target="_blank">${s.signal_name}: ${s.tx_hash.slice(0, 12)}…</a>`
  ).join("");
  setHTML("audit-tx-links", txLinks);
}

function toggleAudit() {
  const body = $("audit-body"); if (!body) return;
  body.classList.toggle("open");
  setText("audit-toggle", body.classList.contains("open") ? "▲ collapse" : "▼ expand");
}

// ─── Bandit Stats ─────────────────────────────────────────────────────────────
let _banditHistory = {};

async function fetchBanditStats() {
  try {
    const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error();
    const data = await res.json();
    const bd = data.bandit_state;
    if (!bd || !Object.keys(bd).length) { removeClass("bandit-section", "visible"); return; }

    addClass("bandit-section", "visible");
    const entries = Object.entries(bd);

    setHTML("bandit-list", entries.map(([name, stats]) => {
      const pct = Math.round(stats.expected_value * 100);
      const uses = stats.alpha + stats.beta;
      const hist = _banditHistory[name] || [];
      hist.push(stats.expected_value);
      if (hist.length > 5) hist.shift();
      _banditHistory[name] = hist;
      const trend = hist.length >= 2 ? (hist[hist.length-1] - hist[hist.length-2] > 0.05 ? "up" : hist[hist.length-1] - hist[hist.length-2] < -0.05 ? "down" : "neutral") : "neutral";
      const trendText = { up: "↑ improving", down: "↓ declining", neutral: "→ stable" }[trend];

      return `
        <div class="bandit-row">
          <span class="bandit-name">${escapeHtml(name)}</span>
          <div class="bandit-bar-wrap"><div class="bandit-bar-fill" style="width:${Math.max(pct, 3)}%"></div></div>
          <span class="bandit-pct">${pct}%</span>
          <span class="bandit-trend ${trend}">${trendText}</span>
        </div>
      `;
    }).join(""));
  } catch (_) {
    removeClass("bandit-section", "visible");
  }
}

// ─── Health Check ─────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) throw new Error();
    const data = await res.json();
    setClass("health-dot", "status-dot");
    setText("health-status", "Online");
    if (data.model_metrics?.roc_auc) {
      setText("health-auc", `${(data.model_metrics.roc_auc * 100).toFixed(1)}%`);
    }
  } catch (_) {
    setClass("health-dot", "status-dot offline");
    setText("health-status", "Offline");
    setText("health-auc", "—");
  }
}

// ─── Init ─────────────────────────────────────────────────────────────────────
(function init() {
  checkHealth();
  setInterval(checkHealth, 30000);

  document.addEventListener("keydown", e => {
    if (e.key === "Enter" && document.activeElement?.tagName === "INPUT") {
      submitTransaction();
    }
  });
})();
