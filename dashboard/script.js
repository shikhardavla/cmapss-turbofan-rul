const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const summaryBody = document.querySelector("#summaryTable tbody");
const detailBody = document.querySelector("#detailTable tbody");

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

function clearResults() {
  summaryBody.innerHTML = "";
  detailBody.innerHTML = "";
}

function renderDetailRow(unit, model, predicted, actual) {
  const row = document.createElement("tr");
  const overestimated = predicted > actual; // An important flag: 
  // Overestimating RUL can be safety-critical because it may delay maintenance.

  if (overestimated) {
    row.classList.add("overestimated");
  }
  row.innerHTML = `
    <td>${unit}</td>
    <td>${model}</td>
    <td>${predicted.toFixed(2)}</td>
    <td>${actual}</td>
    <td>${Math.abs(predicted - actual).toFixed(2)}</td>
  `;
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

runBtn.addEventListener("click", async () => {
  const dataset = document.getElementById("dataset").value;
  const apiUrl = document.getElementById("apiUrl").value.replace(/\/$/, "");
  const models = Array.from(document.querySelectorAll('input[type="checkbox"]:checked'))
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
  const predictions = {};   // model -> { actual: [], predicted: [] }
  models.forEach((m) => (predictions[m] = { actual: [], predicted: [], overestimated: 0 }));

  let done = 0;
  const total = testData.units.length * models.length;

  for (const unit of testData.units) {
    for (const model of models) {
      try {
        const predicted = await predictOne(apiUrl, dataset, model, unit);
        predictions[model].actual.push(unit.true_rul);
        predictions[model].predicted.push(predicted);
        if (predicted > unit.true_rul) {
            predictions[model].overestimated++;
        }
        renderDetailRow(unit.unit, model, predicted, unit.true_rul);
      } catch (e) {
        console.error(`Unit ${unit.unit}, model ${model}:`, e.message);
      }
      done++;
      statusEl.textContent = `Running predictions... (${done}/${total})`;
    }
  }

  models.forEach((model) => {
    const { actual, predicted, overestimated } = predictions[model];
    if (actual.length === 0) return;
    renderSummaryRow(
      model,
      actual.length,
      rmse(actual, predicted),
      mae(actual, predicted),
      nasaScore(actual, predicted),
      overestimated
    );
  });

  statusEl.textContent = `Done. Evaluated ${testData.units.length} units across ${models.length} model(s).`;
  runBtn.disabled = false;
});