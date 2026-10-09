const state = {
  activeView: "overview",
  datasetId: "ibm",
  uploadedDataset: null,
  pipeline: null,
  selectedShapCustomer: 0,
};

const MODEL_LABELS = {
  xgb: "XGBoost",
  gradient_boosting: "Gradient Boosting",
  random_forest: "Random Forest",
  lightgbm: "LightGBM",
  catboost: "CatBoost",
  ebm: "EBM",
};

function formatModelName(name) {
  return MODEL_LABELS[name] || name;
}

function formatFeatureName(name) {
  return String(name || "").replace(/([a-z0-9])([A-Z])/g, "$1 $2");
}

// Real pipeline results (artifacts/dashboard.json, written by
// `python -m pipeline.run`) only apply to the IBM baseline dataset - other
// dataset picks in the selector are illustrative scenario profiles, not
// something the pipeline has actually scored.
function pipelineActive() {
  return Boolean(state.pipeline) && state.datasetId === "ibm";
}

// The dashboard shows only the hybrid's build story: the three component
// models it is made of - Random Forest (donor A), LightGBM (donor B) and
// XGBoost (recipient) - kept as separate rows, PLUS hybrid_union itself.
// The other run variants (rank_fusion / intersection / blend, CatBoost,
// EBM) remain in artifacts/model_evaluation.csv and the Power BI export;
// they are simply hidden from this dashboard's model views.
const DASHBOARD_MODELS = ["hybrid_union", "xgb", "random_forest", "lightgbm"];

async function loadPipelineData() {
  try {
    const response = await fetch("../artifacts/dashboard.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();

    // Keep only the component models + hybrid_union before anything renders
    // (table, AUC stack cards, chart legends, EMP card all read from here).
    payload.models = (payload.models || []).filter((model) =>
      DASHBOARD_MODELS.includes(model.name)
    );

    state.pipeline = payload;

    const bestRow = payload.models.find((model) => model.name === payload.best_model) || payload.models[0];
    Object.assign(datasetProfiles.ibm, {
      name: `${payload.dataset.name} — live pipeline run`,
      pill: "IBM Telco · live results",
      note: `Live pipeline run (${new Date(payload.generated_at).toLocaleString()}): ${formatNumber(payload.dataset.rows)} customers, best model ${formatModelName(payload.best_model)}.`,
      source: "Live pipeline run",
      rows: payload.dataset.rows,
      columns: payload.dataset.columns,
      churnCount: payload.dataset.churn_count,
      churnRate: payload.dataset.churn_rate,
      avgTenure: payload.dataset.avg_tenure,
      avgMonthlyCharge: payload.dataset.avg_monthly_charge,
      eprofitAvg: bestRow ? bestRow.eprofits_avg : datasetProfiles.ibm.eprofitAvg,
      eprofitTenure: bestRow ? bestRow.eprofits_tenure : datasetProfiles.ibm.eprofitTenure,
    });
  } catch (error) {
    // No pipeline run yet (or the page was opened via file:// instead of a
    // local server) - the dashboard falls back to the built-in demo data.
    state.pipeline = null;
  }
}

const models = [
  {
    name: "EBM",
    accuracy: 0.809,
    f1: 0.602,
    auc: 0.863,
    avg: 508658,
    tenure: 3796930,
  },
  {
    name: "XGBoost",
    accuracy: 0.802,
    f1: 1.0,
    auc: 0.856,
    avg: 588397,
    tenure: 3684440,
  },
  {
    name: "CoxPH + ARR",
    accuracy: 0.807,
    f1: 0.585,
    auc: 0.846,
    avg: 510344,
    tenure: 3779420,
  },
];

const datasetProfiles = {
  ibm: {
    id: "ibm",
    name: "IBM Telco baseline",
    pill: "IBM Telco dataset",
    note: "Notebook dataset with contract, tenure, services, billing, charges, and churn labels.",
    source: "Notebook baseline",
    rows: 7043,
    columns: 21,
    churnCount: 1869,
    churnRate: 0.2654,
    avgTenure: 32.37,
    avgMonthlyCharge: 64.76,
    eprofitAvg: 588397,
    eprofitTenure: 3796930,
    growth: 4.5,
    horizon: 72,
    activity: [
      ["0-6", 88, 34],
      ["7-18", 72, 18],
      ["19-36", 58, 42],
      ["37-54", 44, 16],
      ["55-72", 32, 20],
    ],
    segments: [
      ["Electronic check", "Payment method segment with the strongest churn signal.", 45.3],
      ["Month-to-month", "Flexible contract customers churn far above long-term contracts.", 42.7],
      ["Fiber optic", "High monthly charges and service expectations lift risk.", 41.9],
      ["No tech support", "Support absence tracks with preventable churn.", 41.6],
    ],
  },
  stress: {
    id: "stress",
    name: "High-risk pilot sample",
    pill: "High-risk sample",
    note: "Stress-test profile for a month-to-month, digital-billing-heavy customer base.",
    source: "Scenario profile",
    rows: 1800,
    columns: 21,
    churnCount: 621,
    churnRate: 0.345,
    avgTenure: 17.8,
    avgMonthlyCharge: 78.4,
    eprofitAvg: 402800,
    eprofitTenure: 2180000,
    growth: 7.8,
    horizon: 48,
    activity: [
      ["0-6", 96, 38],
      ["7-18", 86, 24],
      ["19-36", 68, 28],
      ["37-54", 48, 18],
      ["55-72", 36, 12],
    ],
    segments: [
      ["New fiber users", "Early-tenure fiber customers need the fastest intervention.", 52.8],
      ["Electronic check", "Manual billing customers remain expensive to retain late.", 49.7],
      ["No online security", "Security add-ons are absent in most risky accounts.", 45.9],
      ["No support plan", "Support bundle is the strongest campaign lever.", 43.6],
    ],
  },
  loyal: {
    id: "loyal",
    name: "Loyal customer sample",
    pill: "Loyal sample",
    note: "Lower-risk profile for long-tenure customers with automatic payments and bundled services.",
    source: "Scenario profile",
    rows: 5200,
    columns: 21,
    churnCount: 510,
    churnRate: 0.098,
    avgTenure: 51.2,
    avgMonthlyCharge: 58.9,
    eprofitAvg: 241600,
    eprofitTenure: 1260000,
    growth: 2.1,
    horizon: 72,
    activity: [
      ["0-6", 54, 18],
      ["7-18", 42, 14],
      ["19-36", 34, 18],
      ["37-54", 24, 10],
      ["55-72", 18, 8],
    ],
    segments: [
      ["Month-to-month", "Even in a loyal base, flexible contracts need monitoring.", 18.6],
      ["Paperless billing", "Digital billing slightly lifts churn pressure.", 13.9],
      ["DSL without support", "Service support remains a retention differentiator.", 12.4],
      ["Short tenure", "Small new-customer cohort carries most avoidable risk.", 11.8],
    ],
  },
};

const planSteps = [
  ["01", "Stabilize", "Target high-risk customers with contract upgrade incentives."],
  ["02", "Support", "Bundle tech support and online security for risky service profiles."],
  ["03", "Billing", "Move electronic-check customers toward automatic payment incentives."],
  ["04", "Value", "Rank outreach by E-profits tenure rather than churn probability alone."],
];

const inputs = {
  contract: document.querySelector("#contract"),
  internet: document.querySelector("#internet"),
  payment: document.querySelector("#payment"),
  support: document.querySelector("#support"),
  security: document.querySelector("#security"),
  paperless: document.querySelector("#paperless"),
  tenure: document.querySelector("#tenure"),
  charge: document.querySelector("#charge"),
};

const datasetControls = {
  select: document.querySelector("#dataset-select"),
  file: document.querySelector("#dataset-file"),
};

function currentDataset() {
  return state.datasetId === "uploaded" && state.uploadedDataset
    ? state.uploadedDataset
    : datasetProfiles[state.datasetId];
}

function money(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(Number.isFinite(value) ? value : 0);
}

function compactMoney(value) {
  const safeValue = Number.isFinite(value) ? value : 0;
  if (Math.abs(safeValue) >= 1000000) return `$${(safeValue / 1000000).toFixed(2)}M`;
  if (Math.abs(safeValue) >= 1000) return `$${Math.round(safeValue / 1000)}K`;
  return money(safeValue);
}

function formatNumber(value, digits = 0) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(Number.isFinite(value) ? value : 0);
}

function percent(value, digits = 0) {
  return `${formatNumber((Number.isFinite(value) ? value : 0) * 100, digits)}%`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function logit(value) {
  const bounded = clamp(value, 0.01, 0.99);
  return Math.log(bounded / (1 - bounded));
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeKey(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function toNumber(value) {
  const parsed = Number.parseFloat(String(value ?? "").replace(/[$,%\s,]/g, ""));
  return Number.isFinite(parsed) ? parsed : null;
}

function isChurnValue(value) {
  const normalized = String(value ?? "").trim().toLowerCase();
  return ["1", "yes", "y", "true", "churn", "churned", "exited", "attrited"].includes(normalized);
}

function average(values) {
  const clean = values.filter((value) => Number.isFinite(value));
  if (!clean.length) return 0;
  return clean.reduce((sum, value) => sum + value, 0) / clean.length;
}

function findColumn(columns, aliases) {
  const normalizedAliases = aliases.map(normalizeKey);
  return columns.find((column) => normalizedAliases.includes(normalizeKey(column)))
    || columns.find((column) => normalizedAliases.some((alias) => normalizeKey(column).includes(alias)));
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (char === '"' && quoted && next === '"') {
      cell += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell.trim());
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(cell.trim());
      if (row.some(Boolean)) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }

  row.push(cell.trim());
  if (row.some(Boolean)) rows.push(row);
  if (rows.length < 2) return [];

  const headers = rows[0].map((header, index) => header || `Column ${index + 1}`);
  return rows.slice(1).map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

function buildSegments(rows, columns, churnColumn) {
  const preferred = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
    "PaperlessBilling",
    "SeniorCitizen",
    "gender",
  ];
  const candidates = preferred.filter((name) => columns.includes(name))
    .concat(columns.filter((name) => !preferred.includes(name)).slice(0, 4));
  const minimumSupport = Math.max(5, Math.floor(rows.length * 0.02));
  const groups = [];

  candidates.forEach((column) => {
    const values = new Map();
    rows.forEach((row) => {
      const rawValue = String(row[column] || "Missing").trim() || "Missing";
      if (!values.has(rawValue)) values.set(rawValue, { count: 0, churn: 0 });
      const item = values.get(rawValue);
      item.count += 1;
      if (churnColumn && isChurnValue(row[churnColumn])) item.churn += 1;
    });

    values.forEach((item, value) => {
      if (item.count < minimumSupport) return;
      const rate = churnColumn ? (item.churn / item.count) * 100 : (item.count / rows.length) * 100;
      groups.push({
        title: value,
        column,
        note: churnColumn
          ? `${column} segment across ${formatNumber(item.count)} customers.`
          : `${column} segment share across ${formatNumber(item.count)} customers.`,
        rate,
      });
    });
  });

  return groups
    .sort((a, b) => b.rate - a.rate)
    .slice(0, 4)
    .map((item) => [item.title, item.note, item.rate]);
}

function buildActivity(rows, tenureColumn, churnColumn) {
  if (!tenureColumn) {
    return [
      ["0-6", 44, 16],
      ["7-18", 50, 22],
      ["19-36", 38, 18],
      ["37-54", 28, 12],
      ["55-72", 20, 8],
    ];
  }

  const bins = [
    ["0-6", 0, 6],
    ["7-18", 7, 18],
    ["19-36", 19, 36],
    ["37-54", 37, 54],
    ["55-72", 55, 72],
  ].map(([label, min, max]) => ({ label, min, max, count: 0, churn: 0 }));

  rows.forEach((row) => {
    const tenure = toNumber(row[tenureColumn]);
    if (tenure === null) return;
    const bin = bins.find((item) => tenure >= item.min && tenure <= item.max);
    if (!bin) return;
    bin.count += 1;
    if (churnColumn && isChurnValue(row[churnColumn])) bin.churn += 1;
  });

  return bins.map((bin) => {
    const rate = bin.count && churnColumn ? bin.churn / bin.count : bin.count / Math.max(1, rows.length);
    const high = clamp(18 + rate * 130, 12, 96);
    const low = clamp(high * 0.38, 8, 42);
    return [bin.label, high, low];
  });
}

function analyzeRows(rows, filename) {
  const columns = Object.keys(rows[0] || {});
  const churnColumn = findColumn(columns, ["Churn", "Exited", "Attrition", "Label", "Target"]);
  const tenureColumn = findColumn(columns, ["tenure", "monthsascustomer", "customertenure"]);
  const monthlyColumn = findColumn(columns, ["MonthlyCharges", "MonthlyCharge", "MonthlyFee", "Charge", "Revenue"]);
  const totalColumn = findColumn(columns, ["TotalCharges", "TotalCharge", "LifetimeValue", "CLV"]);

  const churnCount = churnColumn ? rows.filter((row) => isChurnValue(row[churnColumn])).length : 0;
  const churnRate = churnColumn ? churnCount / rows.length : 0;
  const avgTenure = average(rows.map((row) => toNumber(row[tenureColumn])));
  const avgMonthlyCharge = average(rows.map((row) => toNumber(row[monthlyColumn])));
  const avgTotalCharge = average(rows.map((row) => toNumber(row[totalColumn])));
  const estimatedValue = avgMonthlyCharge * Math.max(6, 72 - (avgTenure || 24)) * Math.max(churnCount, rows.length * 0.08);

  return {
    id: "uploaded",
    name: filename.replace(/\.csv$/i, ""),
    pill: "Uploaded CSV",
    note: churnColumn
      ? `Detected ${churnColumn} as churn label and recalculated dashboard details from the uploaded file.`
      : "No churn label column was detected, so the dashboard shows dataset shape and segment share.",
    source: churnColumn ? "Uploaded dataset" : "Uploaded dataset, no churn label",
    rows: rows.length,
    columns: columns.length,
    churnCount,
    churnRate,
    avgTenure,
    avgMonthlyCharge: avgMonthlyCharge || avgTotalCharge / Math.max(1, avgTenure || 1),
    eprofitAvg: estimatedValue * 0.16,
    eprofitTenure: estimatedValue,
    growth: churnColumn ? (churnRate - datasetProfiles.ibm.churnRate) * 100 : 0,
    horizon: Math.max(12, Math.ceil((avgTenure || 72) / 12) * 12),
    activity: buildActivity(rows, tenureColumn, churnColumn),
    segments: buildSegments(rows, columns, churnColumn),
  };
}

function renderActivity(dataset) {
  document.querySelector("#activity-bars").innerHTML = dataset.activity
    .map(([label, high, low]) => `
      <span style="--high: ${clamp(high, 8, 96).toFixed(1)}%; --low: ${clamp(low, 6, 44).toFixed(1)}%">
        <i></i><b>${escapeHtml(label)}</b>
      </span>
    `)
    .join("");
}

function renderDataset() {
  const dataset = currentDataset();
  const retained = 1 - dataset.churnRate;
  const growthPrefix = dataset.growth >= 0 ? "+" : "";

  document.querySelector("#dataset-title").textContent = dataset.name;
  document.querySelector("#dataset-note").textContent = dataset.note;
  document.querySelector("#dataset-pill-name").textContent = dataset.pill;
  document.querySelector("#dataset-rows").textContent = formatNumber(dataset.rows);
  document.querySelector("#dataset-columns").textContent = formatNumber(dataset.columns);
  document.querySelector("#dataset-avg-tenure").textContent = formatNumber(dataset.avgTenure, 1);
  document.querySelector("#dataset-avg-charge").textContent = money(dataset.avgMonthlyCharge);

  document.querySelector("#retained-rate").textContent = percent(retained, 1);
  document.querySelector("#churn-rate").textContent = percent(dataset.churnRate, 1);
  document.querySelector("#churn-count").textContent = formatNumber(dataset.churnCount);
  document.querySelector("#eprofit-tenure").textContent = compactMoney(dataset.eprofitTenure);
  document.querySelector("#dataset-growth").textContent = `${growthPrefix}${formatNumber(dataset.growth, 1)}%`;
  document.querySelector("#outcome-donut").style.background =
    `conic-gradient(from 18deg, var(--rose) 0 ${dataset.churnRate * 100}%, #e64b5c ${dataset.churnRate * 100}% ${(dataset.churnRate * 100) + 19}%, #ffbc99 ${(dataset.churnRate * 100) + 19}% 100%)`;
  document.querySelector("#survival-horizon-label").textContent = `${formatNumber(dataset.horizon)} months`;
  document.querySelector("#chart-subtitle").textContent = dataset.source;
  document.querySelector("#chart-risk-label").textContent = "Dataset churn";
  document.querySelector("#campaign-score").textContent = compactMoney(dataset.eprofitAvg);
  document.querySelector("#campaign-score-note").textContent =
    dataset.id === "uploaded"
      ? "Estimated campaign value from uploaded rows; retrain the notebook model for final metrics."
      : "Projected E-profits average from tuned notebook runs.";
  document.querySelector("#segments-subtitle").textContent = dataset.churnCount ? "Churn rate" : "Segment share";

  renderActivity(dataset);
  renderModels();
  renderModelStack();
  renderXai();
  renderRetention();
  renderShapCustomers();
  updateCustomer();
}

function riskDetails() {
  const dataset = currentDataset();
  const tenure = Number(inputs.tenure.value);
  const charge = Number(inputs.charge.value);
  let score = logit(dataset.churnRate || 0.18) - 0.85;
  const drivers = [];

  const add = (label, impact) => {
    score += impact;
    drivers.push({ label, impact });
  };

  if (inputs.contract.value === "Month-to-month") add("Month-to-month contract", 1.0);
  if (inputs.contract.value === "One year") add("One-year contract", -0.8);
  if (inputs.contract.value === "Two year") add("Two-year contract", -1.55);

  if (inputs.internet.value === "Fiber optic") add("Fiber optic service", 0.55);
  if (inputs.internet.value === "No") add("No internet service", -0.82);

  if (inputs.payment.value === "Electronic check") add("Electronic check", 0.55);
  if (inputs.payment.value.includes("automatic")) add("Automatic payment", -0.45);

  if (inputs.support.value === "No") add("No tech support", 0.48);
  if (inputs.support.value === "Yes") add("Tech support active", -0.36);

  if (inputs.security.value === "No") add("No online security", 0.42);
  if (inputs.security.value === "Yes") add("Online security active", -0.34);

  if (inputs.paperless.value === "Yes") add("Paperless billing", 0.28);
  if (inputs.paperless.value === "No") add("Paper billing", -0.18);

  if (tenure < 12) add("Early tenure", 0.74);
  if (tenure >= 12 && tenure < 36) add("Mid tenure", 0.22);
  if (tenure >= 60) add("Loyal tenure", -0.62);

  if (charge > 85) add("High monthly charge", 0.34);
  if (charge < 35) add("Low monthly charge", -0.26);

  return {
    probability: sigmoid(score),
    tenure,
    charge,
    drivers: drivers.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact)).slice(0, 5),
  };
}

function updateCustomer() {
  const details = riskDetails();
  const risk = details.probability;
  const riskDegrees = Math.round(risk * 100);
  const band = risk > 0.62 ? "High" : risk > 0.34 ? "Moderate" : "Low";
  const saveWindow = risk > 0.62 ? "3 mo" : risk > 0.34 ? "12 mo" : "24 mo";
  const value = details.charge * Math.max(6, 72 - details.tenure) * risk;

  document.querySelector("#risk-score").textContent = percent(risk);
  document.querySelector("#risk-band").textContent = band;
  document.querySelector("#tenure-value").textContent = details.tenure;
  document.querySelector("#charge-value").textContent = details.charge;
  document.querySelector("#retention-value").textContent = money(value);
  document.querySelector("#save-window").textContent = saveWindow;
  document.querySelector("#risk-ring").style.background =
    `conic-gradient(var(--rose) 0 ${riskDegrees}%, #ccd4e3 ${riskDegrees}% 100%)`;

  document.querySelector("#customer-summary-text").textContent =
    `${inputs.contract.value} ${inputs.internet.value.toLowerCase()} customer with ${inputs.payment.value.toLowerCase()} billing.`;

  renderSurvival(details);
  renderDrivers(details.drivers);
}

function renderSurvival(details) {
  const svg = document.querySelector("#survival-chart");
  const width = 620;
  const height = 300;
  const pad = 34;
  const risk = details.probability;
  const decay = 0.004 + risk * 0.018;
  const points = [];

  for (let month = 0; month <= 72; month += 6) {
    const probability = Math.max(0.04, Math.exp(-decay * month));
    const x = pad + (month / 72) * (width - pad * 2);
    const y = height - pad - probability * (height - pad * 2);
    points.push([x, y]);
  }

  const line = points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  const area = `${line} L ${width - pad} ${height - pad} L ${pad} ${height - pad} Z`;
  const markers = points
    .map(([x, y]) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4"></circle>`)
    .join("");

  svg.innerHTML = `
    <defs>
      <linearGradient id="survival-fill" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="#ef416d" stop-opacity="0.24"/>
        <stop offset="100%" stop-color="#ef416d" stop-opacity="0.02"/>
      </linearGradient>
    </defs>
    <rect x="${pad}" y="${pad}" width="${width - pad * 2}" height="${height - pad * 2}" rx="18" fill="#f7f9fd"></rect>
    <path d="${area}" fill="url(#survival-fill)"></path>
    <path d="${line}" fill="none" stroke="#ef416d" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"></path>
    <g fill="#ffffff" stroke="#ef416d" stroke-width="3">${markers}</g>
    <text x="${pad}" y="${height - 8}" fill="#8a90a3" font-size="12" font-weight="800">0 months</text>
    <text x="${width - pad - 60}" y="${height - 8}" fill="#8a90a3" font-size="12" font-weight="800">72 months</text>
    <text x="${pad}" y="22" fill="#8a90a3" font-size="12" font-weight="800">Retention probability</text>
  `;
}

function renderDrivers(drivers) {
  const list = document.querySelector("#driver-list");
  list.innerHTML = drivers
    .map((driver) => {
      const magnitude = Math.min(100, Math.round(Math.abs(driver.impact) * 78));
      const signed = driver.impact > 0 ? "+" : "";
      return `
        <div class="driver">
          <span>${escapeHtml(driver.label)}<b>${signed}${driver.impact.toFixed(2)}</b></span>
          <div class="track"><i style="--value: ${magnitude}%"></i></div>
        </div>
      `;
    })
    .join("");
}

function renderModels() {
  const tbody = document.querySelector("#model-table");

  if (pipelineActive()) {
    const rows = state.pipeline.models;
    const bestName = state.pipeline.best_model;
    const bestAuc = Math.max(...rows.map((model) => model.roc_auc));

    tbody.innerHTML = [...rows]
      .sort((a, b) => b.eprofits_tenure - a.eprofits_tenure)
      .map((model) => `
        <tr class="${model.name === bestName ? "winner-row" : ""}">
          <td>${escapeHtml(formatModelName(model.name))}${model.name === bestName ? " ★" : ""}</td>
          <td>${model.accuracy.toFixed(3)}</td>
          <td>${model.f1.toFixed(3)}</td>
          <td class="${model.roc_auc === bestAuc ? "winner" : ""}">${model.roc_auc.toFixed(3)}</td>
          <td>${money(model.eprofits_avg)}</td>
          <td class="${model.name === bestName ? "winner" : ""}">${money(model.eprofits_tenure)}</td>
        </tr>
      `)
      .join("");
    document.querySelector("#model-subtitle").textContent = `Live results · best: ${formatModelName(bestName)}`;
    return;
  }

  const bestAuc = Math.max(...models.map((model) => model.auc));
  const bestAvg = Math.max(...models.map((model) => model.avg));

  tbody.innerHTML = models
    .map((model) => `
      <tr>
        <td>${escapeHtml(model.name)}</td>
        <td>${model.accuracy.toFixed(3)}</td>
        <td>${model.f1.toFixed(3)}</td>
        <td class="${model.auc === bestAuc ? "winner" : ""}">${model.auc.toFixed(3)}</td>
        <td class="${model.avg === bestAvg ? "winner" : ""}">${money(model.avg)}</td>
        <td>${money(model.tenure)}</td>
      </tr>
    `)
    .join("");
  document.querySelector("#model-subtitle").textContent =
    currentDataset().id === "uploaded" ? "Notebook metrics, dataset changed" : currentDataset().source;
}

function renderModelStack() {
  if (!pipelineActive()) {
    document.querySelector("#legend-a-label").textContent = "EBM AUC";
    document.querySelector("#legend-b-label").textContent = "XGB AUC";
    document.querySelector("#stack-a-name").textContent = "EBM";
    document.querySelector("#stack-a-value").textContent = "0.862 AUC";
    document.querySelector("#stack-b-name").textContent = "XGBoost";
    document.querySelector("#stack-b-value").textContent = "0.856 AUC";
    document.querySelector("#stack-emp-value").textContent = "10.28";
    document.querySelector("#stack-arr-value").textContent = "ARR 0.951";
    return;
  }

  const ranked = [...state.pipeline.models].sort((a, b) => b.roc_auc - a.roc_auc);
  const [first, second] = ranked;
  const best = ranked.find((model) => model.name === state.pipeline.best_model) || first;
  const arr = state.pipeline.dataset.avg_retention_rate;

  document.querySelector("#legend-a-label").textContent = `${formatModelName(first.name)} AUC`;
  document.querySelector("#legend-b-label").textContent = second ? `${formatModelName(second.name)} AUC` : "—";
  document.querySelector("#stack-a-name").textContent = formatModelName(first.name);
  document.querySelector("#stack-a-value").textContent = `${first.roc_auc.toFixed(3)} AUC`;
  if (second) {
    document.querySelector("#stack-b-name").textContent = formatModelName(second.name);
    document.querySelector("#stack-b-value").textContent = `${second.roc_auc.toFixed(3)} AUC`;
  }
  document.querySelector("#stack-emp-value").textContent = best.emp.toFixed(2);
  document.querySelector("#stack-arr-value").textContent = Number.isFinite(arr) ? `ARR ${arr.toFixed(3)}` : "—";
}

function renderXai() {
  if (pipelineActive() && state.pipeline.shap_global.length) {
    const top = state.pipeline.shap_global.slice(0, 8);
    const maxShap = Math.max(...top.map((feature) => feature.mean_abs_shap));

    document.querySelector("#xai-bars").innerHTML = top
      .map((feature) => {
        const value = Math.round((feature.mean_abs_shap / maxShap) * 100);
        return `
          <div class="bar-row">
            <span>${escapeHtml(formatFeatureName(feature.feature))}<b>${feature.mean_abs_shap.toFixed(3)}</b></span>
            <div class="track"><i style="--value: ${value}%"></i></div>
          </div>
        `;
      })
      .join("");
    document.querySelector("#xai-subtitle").textContent = `SHAP · ${formatModelName(state.pipeline.best_model)}`;
    return;
  }

  const dataset = currentDataset();
  const drivers = dataset.segments.map(([title, , rate]) => [title, Math.round(clamp(rate * 1.8, 8, 96))]);

  document.querySelector("#xai-bars").innerHTML = drivers
    .map(([label, value]) => `
      <div class="bar-row">
        <span>${escapeHtml(label)}<b>${value}</b></span>
        <div class="track"><i style="--value: ${value}%"></i></div>
      </div>
    `)
    .join("");
  document.querySelector("#xai-subtitle").textContent = dataset.id === "uploaded" ? "Uploaded CSV drivers" : "Dataset drivers";
}

function renderShapCustomers() {
  const card = document.querySelector("#shap-customers-card");
  const customers = pipelineActive() ? state.pipeline.shap_customers : [];

  if (!customers.length) {
    card.hidden = true;
    return;
  }
  card.hidden = false;

  document.querySelector("#shap-customers-subtitle").textContent =
    `SHAP · ${formatModelName(state.pipeline.best_model)} · ${customers.length} customers`;

  const list = document.querySelector("#shap-customer-list");
  list.innerHTML = customers
    .map((customer, index) => `
      <div class="segment-row customer-row" data-index="${index}" tabindex="0" role="button" aria-label="View explanation for customer ${customer.id}">
        <div>
          <span>Customer #${customer.id}${customer.actual_churn ? " · actually churned" : ""}</span>
          <p>${escapeHtml(customer.contract)} · ${escapeHtml(customer.internet_service)} · ${Math.round(customer.tenure)} mo tenure · ${money(customer.monthly_charges)}/mo</p>
        </div>
        <b class="segment-rate">${percent(customer.churn_probability)}</b>
      </div>
    `)
    .join("");

  list.querySelectorAll(".customer-row").forEach((row) => {
    row.addEventListener("click", () => selectShapCustomer(Number(row.dataset.index)));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectShapCustomer(Number(row.dataset.index));
      }
    });
  });

  const initialIndex = state.selectedShapCustomer < customers.length ? state.selectedShapCustomer : 0;
  selectShapCustomer(initialIndex);
}

function selectShapCustomer(index) {
  const customers = state.pipeline.shap_customers;
  const customer = customers[index];
  if (!customer) return;

  state.selectedShapCustomer = index;

  document.querySelectorAll("#shap-customer-list .customer-row").forEach((row) => {
    row.classList.toggle("active", Number(row.dataset.index) === index);
  });

  const riskDegrees = Math.round(customer.churn_probability * 100);
  const maxAbsShap = Math.max(...customer.top_features.map((feature) => Math.abs(feature.shap_value)), 0.0001);

  const driverRows = customer.top_features
    .map((feature) => {
      const magnitude = Math.round((Math.abs(feature.shap_value) / maxAbsShap) * 100);
      const signed = feature.shap_value >= 0 ? "+" : "";
      const directionClass = feature.direction === "toward churn" ? "toward" : "away";
      return `
        <div class="driver ${directionClass}">
          <span>${escapeHtml(formatFeatureName(feature.feature))} = ${escapeHtml(feature.value)}<b>${signed}${feature.shap_value.toFixed(3)}</b></span>
          <div class="track"><i style="--value: ${magnitude}%"></i></div>
        </div>
      `;
    })
    .join("");

  document.querySelector("#shap-customer-detail").innerHTML = `
    <div class="shap-detail-head">
      <div class="risk-ring small" style="background: conic-gradient(var(--rose) 0 ${riskDegrees}%, #ccd4e3 ${riskDegrees}% 100%)">
        <div><span>Churn</span><strong>${percent(customer.churn_probability)}</strong></div>
      </div>
      <div>
        <strong>Customer #${customer.id}</strong>
        <p class="muted">${escapeHtml(customer.contract)} · ${escapeHtml(customer.internet_service)} · ${escapeHtml(customer.payment_method)}</p>
        <p class="muted">Tenure ${Math.round(customer.tenure)} mo · ${money(customer.monthly_charges)}/mo · Tech support: ${escapeHtml(customer.tech_support)} · Online security: ${escapeHtml(customer.online_security)}</p>
      </div>
    </div>
    <div class="driver-list">${driverRows}</div>
  `;
}

function renderRetention() {
  const dataset = currentDataset();

  document.querySelector("#plan-steps").innerHTML = planSteps
    .map(([step, title, text]) => `
      <div class="plan-step">
        <b>${step}</b>
        <div>
          <span>${escapeHtml(title)}</span>
          <p>${escapeHtml(text)}</p>
        </div>
      </div>
    `)
    .join("");

  document.querySelector("#segments").innerHTML = dataset.segments
    .map(([title, note, rate]) => `
      <div class="segment-row">
        <div>
          <span>${escapeHtml(title)}</span>
          <p>${escapeHtml(note)}</p>
        </div>
        <b class="segment-rate">${formatNumber(rate, 1)}%</b>
      </div>
    `)
    .join("");
}

function switchView(view) {
  state.activeView = view;
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelectorAll(".view").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.viewPanel === view);
  });
}

function initNavigation() {
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
}

function initInputs() {
  Object.values(inputs).forEach((input) => {
    input.addEventListener("input", updateCustomer);
    input.addEventListener("change", updateCustomer);
  });
}

function initDatasets() {
  datasetControls.select.addEventListener("change", () => {
    const nextDataset = datasetControls.select.value;
    if (nextDataset === "uploaded" && !state.uploadedDataset) {
      datasetControls.file.click();
      datasetControls.select.value = state.datasetId;
      return;
    }

    state.datasetId = nextDataset;
    renderDataset();
  });

  datasetControls.file.addEventListener("change", async () => {
    const [file] = datasetControls.file.files;
    if (!file) return;

    try {
      const rows = parseCsv(await file.text());
      if (!rows.length) throw new Error("The CSV file has no readable rows.");
      state.uploadedDataset = analyzeRows(rows, file.name);
      state.datasetId = "uploaded";
      datasetControls.select.value = "uploaded";
      renderDataset();
    } catch (error) {
      document.querySelector("#dataset-note").textContent = error.message;
    }
  });
}

function initChartPoints() {
  const group = document.querySelector(".chart-points");
  const colors = ["#1aa579", "#4b5fff", "#ef416d"];
  const paths = [
    "18 192,58 166,98 176,138 142,178 152,218 122,258 136,298 116,338 128,378 98,418 110,458 92,498 104,538 86,578 102,620 88",
    "18 210,58 188,98 194,138 174,178 164,218 154,258 144,298 136,338 132,378 122,418 118,458 112,498 106,538 98,578 94,620 90",
    "18 222,58 230,98 218,138 226,178 214,218 224,258 206,298 216,338 202,378 210,418 194,458 204,498 190,538 202,578 214,620 224",
  ];

  group.innerHTML = paths
    .map((path, index) => path
      .split(",")
      .map((point) => {
        const [x, y] = point.split(" ");
        return `<circle cx="${x}" cy="${y}" r="5" fill="#ffffff" stroke="${colors[index]}" stroke-width="3"></circle>`;
      })
      .join(""))
    .join("");
}

async function bootstrap() {
  initNavigation();
  initInputs();
  initDatasets();
  initChartPoints();
  await loadPipelineData();
  renderDataset();
}

bootstrap();
