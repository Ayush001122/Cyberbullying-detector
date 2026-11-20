const $ = (s) => document.querySelector(s);
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x * 100).toFixed(1) + "%";

async function api(url, opts) {
  const r = await fetch(url, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || "Request failed");
  return data;
}
const post = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// ---- tabs ----
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === b.dataset.tab));
  if (b.dataset.tab === "dashboard") loadDashboard();
}));

// ---- analyze ----
const verdict = (r) => (r.is_toxic ? ["⚠️ Bullying detected", "bad"] : ["✅ Looks safe", "good"]);

async function analyze() {
  const text = $("#text").value.trim();
  if (!text) return;
  const btn = $("#analyzeBtn"); btn.disabled = true; btn.textContent = "Analyzing…";
  try {
    const r = await post("/api/predict", { text });
    const [label, cls] = verdict(r);
    $("#badge").textContent = label; $("#badge").className = "badge " + cls;
    $("#toxPct").textContent = pct(r.toxicity);
    const fill = $("#meterFill"); fill.style.width = pct(r.toxicity);
    fill.style.background = r.toxicity > 0.7 ? "var(--bad)" : r.toxicity > 0.4 ? "var(--warn)" : "var(--good)";
    $("#meta").textContent = `Severity: ${r.severity} · Confidence: ${pct(r.confidence)} · Raw SVM score: ${r.score}`;
    $("#highlight").innerHTML = r.tokens.map((t) =>
      t.weight > 0.15 ? `<mark style="--a:${Math.min(0.15 + t.weight * 0.6, 0.75)}">${esc(t.word)}</mark>` : esc(t.word)).join(" ");
    $("#result").classList.remove("hidden");
  } catch (e) { alert(e.message); }
  btn.disabled = false; btn.textContent = "Analyze message";
}
$("#analyzeBtn").addEventListener("click", analyze);
$("#text").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze(); });
document.querySelectorAll(".ex").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); $("#text").value = a.textContent; analyze(); }));

// ---- batch ----
const stat = (v, l, c = "") => `<div class="stat ${c}"><b>${v}</b><span>${l}</span></div>`;
const pill = (r) => `<span class="pill ${r.is_toxic ? "bad" : "good"}">${r.is_toxic ? "bullying" : "safe"}</span>`;

$("#file").addEventListener("change", async (e) => {
  const f = e.target.files[0]; if (!f) return;
  let t = await f.text();
  if (f.name.endsWith(".csv")) t = t.split(/\r?\n/).slice(1).map((l) => l.replace(/^"|"?,-?\d+$|"$/g, "")).join("\n");
  $("#batchText").value = t;
});
$("#batchBtn").addEventListener("click", async () => {
  const texts = $("#batchText").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!texts.length) return;
  const btn = $("#batchBtn"); btn.disabled = true;
  try {
    const d = await post("/api/predict/batch", { texts });
    $("#batchStats").innerHTML = stat(d.summary.total, "scanned") + stat(d.summary.flagged, "flagged", "bad") + stat(d.summary.clean, "safe", "good");
    $("#batchRows").innerHTML = d.results.map((r) =>
      `<tr><td>${esc(r.text)}</td><td>${pill(r)}</td><td>${r.severity}</td><td>${r.score}</td></tr>`).join("");
    $("#batchOut").classList.remove("hidden");
  } catch (e) { alert(e.message); }
  btn.disabled = false;
});

// ---- dashboard ----
async function loadDashboard() {
  try {
    const m = await api("/api/metrics");
    const d = m.dataset;
    $("#dsStats").innerHTML = stat(d.rows.toLocaleString(), "unique rows") + stat(d.toxic.toLocaleString(), "toxic", "bad") +
      stat(d.non_toxic.toLocaleString(), "non-toxic", "good") + stat(d.test_rows.toLocaleString(), "test rows");
    const c = m.confusion_matrix;
    $("#cm").innerHTML = `<div class="ok"><b>${c.tn}</b><span>True non-toxic</span></div><div class="err"><b>${c.fp}</b><span>False alarm</span></div>
      <div class="err"><b>${c.fn}</b><span>Missed bullying</span></div><div class="ok"><b>${c.tp}</b><span>True bullying</span></div>`;
    $("#bars").innerHTML = `<div class="legend"><b style="background:var(--brand)"></b>F1<b style="background:#06b6d4"></b>Accuracy</div>` +
      m.results.map((r) => `<div class="bar-row"><span>${esc(r.algorithm)}</span>
        <div class="bar"><i class="f1" style="width:${r.f1 * 100}%"></i><i class="acc" style="width:${r.accuracy * 100}%"></i></div>
        <span>${pct(r.f1)}</span></div>`).join("");
    $("#prodNote").textContent = `Model used by the app: ${m.production_model.algorithm} (C = ${m.production_model.params.clf__C}). Trained on: ${d.source}.`;
  } catch (e) { $("#bars").textContent = e.message; }
  loadHistory();
}
async function loadHistory() {
  const h = await api("/api/history");
  $("#histStats").innerHTML = stat(h.stats.total, "analyzed") + stat(h.stats.bullying, "flagged", "bad") + stat(h.stats.non_bullying, "safe", "good");
  $("#histRows").innerHTML = h.items.map((r) =>
    `<tr><td>${esc(r.text)}</td><td>${pill({ is_toxic: r.label === "bullying" })}</td><td>${r.severity}</td><td>${r.created_at}</td></tr>`).join("")
    || `<tr><td colspan="4" class="muted">Nothing yet — analyze a message first.</td></tr>`;
}
$("#clearBtn").addEventListener("click", async () => { await api("/api/history", { method: "DELETE" }); loadHistory(); });
