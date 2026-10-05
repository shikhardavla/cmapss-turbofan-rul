const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const summaryBody = document.querySelector("#summaryTable tbody");
const detailBody = document.querySelector("#detailTable tbody");

const policyToggle = document.getElementById("policyToggle");
const policyControls = document.getElementById("policyControls");
const safetyFactorInput = document.getElementById("safetyFactor");
const safetyFactorValue = document.getElementById("safetyFactorValue");
const criticalThresholdInput = document.getElementById("criticalThreshold");
const criticalThresholdValue = document.getElementById("criticalThresholdValue");
const policyColHeader = document.getElementById("policyColHeader");

// Cache of the last successful "Run Evaluation" click. Policy toggle/sliders
// recompute and re-render from this instantly -- no new API calls -- so the
// effect of safety factor / threshold changes is visible live.
let lastRun = null; // { dataset, records: [{ unit, model, actual, predicted }] }

// Mirrors evaluate.py's nasa_score exactly: asymmetric exponential penalty,
// summed (not averaged) -- late predictions penalized more steeply.
function nasaScore(actualArr, predictedArr) {
  let total = 0;
  for (let i = 0; i < actualArr.length; i++) {
    const error = predictedArr[i] - actualArr[i];
    total += error < 0
      ? Math.exp(-error / 13) - 1
      : Math.exp(error / 10) - 1;
  }
  return total;
}

function rmse(actualArr, predictedArr) {
  const mse = actualArr.reduce((sum, a, i) => sum + (predictedArr[i] - a) ** 2, 0) / actualArr.length;
  return Math.sqrt(mse);
}

function mae(actualArr, predictedArr) {
  return actualArr.reduce((sum, a, i) => sum + Math.abs(predictedArr[i] - a), 0) / actualArr.length;
}

async function predictOne(apiUrl, dataset, model, unit) {
  const response = await fetch(`${apiUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset,
      model,
      unit_id: String(unit.unit),
      cycles: unit.cycles,
    }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }

  const data = await response.json();
  return data.predicted_rul;
}

function getPolicyState() {
  return {
    enabled: policyToggle.checked,
    safetyFactor: parseFloat(safetyFactorInput.value),
    criticalThreshold: parseFloat(criticalThresholdInput.value),
  };
}

// The core policy transform: adjusted RUL = predicted RUL / safety factor.
// When policy is off, the raw model prediction passes through unchanged.
function adjustedRul(predicted, policy) {
  return policy.enabled ? predicted / policy.safetyFactor : predicted;
}

function clearResults() {
  summaryBody.innerHTML = "";
  detailBody.innerHTML = "";
}

function renderDetailRow(unit, model, displayedPredicted, actual, policyEnabled, unitIsCritical) {
  const row = document.createElement("tr");
  const overestimated = displayedPredicted > actual; // An important flag:
  // Overestimating RUL can be safety-critical because it may delay maintenance.

  if (overestimated) {
    row.classList.add("overestimated");
  }

  let rowHtml = `
    <td>${unit}</td>
    <td>${model}</td>
    <td>${displayedPredicted.toFixed(2)}</td>
    <td>${actual}</td>
    <td>${Math.abs(displayedPredicted - actual).toFixed(2)}</td>
  `;

  if (policyEnabled) {
    rowHtml += unitIsCritical
      ? `<td class="policy-col"><span class="status-dot critical"></span>Maintenance Critical</td>`
      : `<td class="policy-col"><span class="status-dot healthy"></span>Engine Healthy</td>`;
  }

  row.innerHTML = rowHtml;
  detailBody.appendChild(row);
}

function renderSummaryRow(model, n, rmseVal, maeVal, nasaVal, overestimated) {
  const row = document.createElement("tr");
  row.innerHTML = `
    <td>${model}</td>
    <td>${n}</td>
    <td>${rmseVal.toFixed(2)}</td>
    <td>${maeVal.toFixed(2)}</td>
    <td>${nasaVal.toFixed(2)}</td>
    <td>${overestimated} (${((overestimated / n) * 100).toFixed(0)}%)</td>
  `;
  summaryBody.appendChild(row);
}

// Recomputes and re-renders both tables from the cached last run, applying
// the current policy state (toggle + sliders). Called after a fresh
// evaluation completes, and again every time the toggle or sliders change --
// no API calls happen here, it's pure client-side recomputation.
function renderAll() {
  if (!lastRun) return;

  const policy = getPolicyState();
  policyColHeader.style.display = policy.enabled ? "" : "none";
  clearResults();

  const displayed = lastRun.records.map((r) => ({
    unit: r.unit,
    model: r.model,
    actual: r.actual,
    displayedPredicted: adjustedRul(r.predicted, policy),
  }));

  // Per-engine worst-case policy status: if ANY active model's adjusted RUL
  // for that unit is at or below the critical threshold, the whole engine
  // is flagged critical -- matches "any of the active models" precedence.
  const unitCritical = {};
  if (policy.enabled) {
    displayed.forEach((r) => {
      if (r.displayedPredicted <= policy.criticalThreshold) {
        unitCritical[r.unit] = true;
      }
    });
  }

  displayed.forEach((r) => {
    renderDetailRow(
      r.unit,
      r.model,
      r.displayedPredicted,
      r.actual,
      policy.enabled,
      !!unitCritical[r.unit]
    );
  });

  // Group by model for the summary table -- RMSE/MAE/NASA Score/overestimate
  // count are all computed from the (possibly safety-factor-adjusted)
  // displayed values, so moving the slider visibly moves these numbers.
  const byModel = {};
  displayed.forEach((r) => {
    if (!byModel[r.model]) {
      byModel[r.model] = { actual: [], predicted: [], overestimated: 0 };
    }
    byModel[r.model].actual.push(r.actual);
    byModel[r.model].predicted.push(r.displayedPredicted);
    if (r.displayedPredicted > r.actual) byModel[r.model].overestimated++;
  });

  Object.keys(byModel).forEach((model) => {
    const { actual, predicted, overestimated } = byModel[model];
    renderSummaryRow(
      model,
      actual.length,
      rmse(actual, predicted),
      mae(actual, predicted),
      nasaScore(actual, predicted),
      overestimated
    );
  });
}

runBtn.addEventListener("click", async () => {
  const dataset = document.getElementById("dataset").value;
  const apiUrl = document.getElementById("apiUrl").value.replace(/\/$/, "");
  const models = Array.from(document.querySelectorAll(".model-checkbox:checked"))
    .map((cb) => cb.value);

  if (models.length === 0) {
    statusEl.textContent = "Select at least one model.";
    return;
  }

  clearResults();
  runBtn.disabled = true;

  let testData;
  try {
    statusEl.textContent = `Loading ${dataset} test data...`;
    const res = await fetch(`data/${dataset}.json`);
    if (!res.ok) throw new Error(`Could not load data/${dataset}.json -- run export_dashboard_data.py first`);
    testData = await res.json();
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
    runBtn.disabled = false;
    return;
  }

  // Sequential requests, not Promise.all -- a full dataset x 3 models can be
  // hundreds of requests; sequential keeps this simple and avoids hammering
  // a locally-running dev server. Slower, but predictable.
  const records = [];
  let done = 0;
  const total = testData.units.length * models.length;

  for (const unit of testData.units) {
    for (const model of models) {
      try {
        const predicted = await predictOne(apiUrl, dataset, model, unit);
        records.push({ unit: unit.unit, model, actual: unit.true_rul, predicted });
      } catch (e) {
        console.error(`Unit ${unit.unit}, model ${model}:`, e.message);
      }
      done++;
      statusEl.textContent = `Running predictions... (${done}/${total})`;
    }
  }

  lastRun = { dataset, records };
  renderAll();

  statusEl.textContent = `Done. Evaluated ${testData.units.length} units across ${models.length} model(s).`;
  runBtn.disabled = false;
});

policyToggle.addEventListener("change", () => {
  policyControls.disabled = !policyToggle.checked;
  renderAll();
});

safetyFactorInput.addEventListener("input", () => {
  safetyFactorValue.textContent = parseFloat(safetyFactorInput.value).toFixed(1);
  renderAll();
});

criticalThresholdInput.addEventListener("input", () => {
  criticalThresholdValue.textContent = criticalThresholdInput.value;
  renderAll();
});