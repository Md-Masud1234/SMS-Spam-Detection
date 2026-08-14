const API_BASE = window.location.port === "8000" ? window.location.origin : "http://127.0.0.1:8000";

const $ = (id) => document.getElementById(id);

async function checkBackend() {
  const status = $("backendStatus");
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.model_loaded) {
      status.className = "status ok";
      status.innerHTML = "<span></span> Backend + model online";
    } else {
      status.className = "status bad";
      status.innerHTML = "<span></span> Backend online / model missing";
    }
  } catch {
    status.className = "status bad";
    status.innerHTML = "<span></span> Backend offline";
  }
}

document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    $(btn.dataset.tab).classList.add("active");
  });
});

$("analyzeBtn").addEventListener("click", async () => {
  const message = $("message").value.trim();
  const threshold = Number($("threshold").value);
  const error = $("errorBox");
  error.classList.add("hidden");

  if (!message) {
    error.textContent = "Please enter an SMS message.";
    error.classList.remove("hidden");
    return;
  }

  $("analyzeBtn").disabled = true;
  $("analyzeBtn").textContent = "INSPECTING...";

  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message, threshold})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Analysis failed.");
    renderResult(data);
  } catch (err) {
    error.textContent = err.message;
    error.classList.remove("hidden");
  } finally {
    $("analyzeBtn").disabled = false;
    $("analyzeBtn").textContent = "INSPECT MESSAGE →";
  }
});

function renderResult(data) {
  $("emptyResult").classList.add("hidden");
  $("result").classList.remove("hidden");
  const verdict = $("verdict");
  verdict.textContent = data.label;
  verdict.className = `verdict ${data.label.toLowerCase()}`;
  $("risk").textContent = data.risk;
  $("score").textContent = `${data.spam_percentage}%`;
  $("scoreBar").style.width = `${data.spam_percentage}%`;
  $("modelDevice").textContent = (data.device || "").toUpperCase();

  const rules = $("rules");
  rules.innerHTML = "";
  if (!data.triggered_rules.length) {
    rules.innerHTML = '<span class="chip">No rule triggers</span>';
  } else {
    data.triggered_rules.forEach(rule => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = rule;
      rules.appendChild(chip);
    });
  }
  $("explanation").textContent = data.explanation;
}

$("chooseBtn").addEventListener("click", () => $("csvInput").click());
$("csvInput").addEventListener("change", () => {
  if ($("csvInput").files[0]) uploadBatch($("csvInput").files[0]);
});

const dropZone = $("dropZone");
["dragenter", "dragover"].forEach(evt => {
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.style.borderColor = "#e4572e";
  });
});
["dragleave", "drop"].forEach(evt => {
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.style.borderColor = "";
  });
});
dropZone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (file) uploadBatch(file);
});

let lastBatchRows = [];

async function uploadBatch(file) {
  const error = $("batchError");
  error.classList.add("hidden");
  if (!file.name.toLowerCase().endsWith(".csv")) {
    error.textContent = "Please choose a CSV file.";
    error.classList.remove("hidden");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/api/batch`, {method: "POST", body: form});
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Batch analysis failed.");

    lastBatchRows = data.rows;
    $("batchResult").classList.remove("hidden");
    $("totalCount").textContent = data.total;
    $("spamCount").textContent = data.spam_count;
    $("hamCount").textContent = data.ham_count;

    const tbody = $("batchRows");
    tbody.innerHTML = "";
    data.rows.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml(row.message)}</td>
        <td><strong>${escapeHtml(row.prediction)}</strong></td>
        <td>${row.spam_percentage}%</td>
        <td>${escapeHtml(row.risk)}</td>
        <td>${escapeHtml(row.triggered_rules || "—")}</td>`;
      tbody.appendChild(tr);
    });
  } catch (err) {
    error.textContent = err.message;
    error.classList.remove("hidden");
  }
}

$("downloadBtn").addEventListener("click", () => {
  if (!lastBatchRows.length) return;
  const headers = ["message", "prediction", "spam_percentage", "risk", "triggered_rules", "explanation"];
  const csv = [
    headers.join(","),
    ...lastBatchRows.map(r => headers.map(h => csvCell(r[h] ?? "")).join(","))
  ].join("\n");
  const blob = new Blob([csv], {type: "text/csv;charset=utf-8;"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "sms_spam_results.csv";
  a.click();
  URL.revokeObjectURL(url);
});

function csvCell(value) {
  return `"${String(value).replaceAll('"', '""')}"`;
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
  }[c]));
}

checkBackend();
