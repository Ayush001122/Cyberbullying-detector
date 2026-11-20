"""Flask backend: serves the UI and a small JSON API around the trained model."""
import json
import sqlite3
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from ml.predictor import Predictor
from ml.train import METRICS_PATH

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "history.db"
MAX_TEXT = 5000
MAX_BATCH = 200

app = Flask(__name__)
predictor = Predictor()


# ---------- tiny SQLite layer (analysis history) ----------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL, label TEXT NOT NULL, score REAL NOT NULL,
            severity TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")


def save(result: dict):
    with db() as c:
        c.execute("INSERT INTO predictions (text,label,score,severity) VALUES (?,?,?,?)",
                  (result["text"][:500], result["label"], result["score"], result["severity"]))


init_db()


def _clean_input(text):
    if not isinstance(text, str) or not text.strip():
        return None
    return text.strip()[:MAX_TEXT]


# ---------- routes ----------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.post("/api/predict")
def predict():
    text = _clean_input((request.get_json(silent=True) or {}).get("text"))
    if text is None:
        return jsonify(error="Please provide some text."), 400
    result = predictor.predict(text)
    save(result)
    return jsonify(result)


@app.post("/api/predict/batch")
def predict_batch():
    texts = (request.get_json(silent=True) or {}).get("texts")
    if not isinstance(texts, list) or not texts:
        return jsonify(error="Provide a non-empty list called 'texts'."), 400
    results = []
    for t in texts[:MAX_BATCH]:
        t = _clean_input(t)
        if t:
            r = predictor.predict(t)
            r.pop("tokens")
            save(r)
            results.append(r)
    flagged = sum(r["is_toxic"] for r in results)
    return jsonify(results=results, summary={"total": len(results), "flagged": flagged,
                                             "clean": len(results) - flagged})


@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return jsonify(error="No metrics found. Run: python -m ml.train"), 404
    return jsonify(json.loads(METRICS_PATH.read_text()))


@app.get("/api/history")
def history():
    with db() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT id,text,label,score,severity,created_at FROM predictions ORDER BY id DESC LIMIT 30")]
        tot = c.execute("SELECT COUNT(*) n, SUM(label='bullying') b FROM predictions").fetchone()
    total, bullying = tot["n"] or 0, tot["b"] or 0
    return jsonify(items=rows, stats={"total": total, "bullying": bullying, "non_bullying": total - bullying})


@app.delete("/api/history")
def clear_history():
    with db() as c:
        c.execute("DELETE FROM predictions")
    return jsonify(status="cleared")


if __name__ == "__main__":
    print("\n  Cyberbullying Detector running at  http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
