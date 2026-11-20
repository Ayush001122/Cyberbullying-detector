# Cyberbullying Detector

A working end-to-end project built from your `Cyberbullying_Model` notebook:
**dataset → cleaning → TF-IDF → model comparison → saved model → Flask API → web UI.**

## Quick start

```bash
python -m venv venv
venv\Scripts\activate          # Windows      (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open **http://127.0.0.1:5000** (or double-click `run.bat` on Windows / run `./run.sh` on macOS/Linux).

## Use YOUR dataset (important)

Your notebook's `dataset.csv` was not included in the upload, so this zip ships with a
**small synthetic demo dataset** (`data/dataset.csv`) and a model trained on it, just so
everything runs immediately. Its scores (~100%) are NOT meaningful.

To use the real data:
1. Replace `data/dataset.csv` with your file (columns `headline,label`; `-1`/`1` = toxic, `0` = non-toxic).
2. Run `python -m ml.train` (add `--slow` to also train BaggingClassifier).
3. Restart `python app.py`. The Dashboard tab now shows your real metrics.

## What's inside

| Path | Purpose |
|---|---|
| `app.py` | Flask backend: UI + JSON API + SQLite history |
| `ml/preprocess.py` | Cleaning from the notebook (remove @mentions, non-letters, short words) |
| `ml/train.py` | Trains/compares LinearSVC, LogReg, NB, DecisionTree, AdaBoost, SGD (+Bagging), grid-tunes LinearSVC, saves model + `metrics.json` |
| `ml/predictor.py` | Loads model, returns label, score, severity, per-word highlights |
| `templates/`, `static/` | Frontend (Analyze, Batch, Dashboard), no external dependencies |
| `tests/test_api.py` | `python -m unittest discover tests -v` |

## API

| Method | Endpoint | Body / result |
|---|---|---|
| POST | `/api/predict` | `{"text": "..."}` → label, toxicity, severity, highlighted tokens |
| POST | `/api/predict/batch` | `{"texts": [...]}` → results + summary (max 200) |
| GET | `/api/metrics` | model comparison, confusion matrix, dataset stats |
| GET / DELETE | `/api/history` | recent analyses + totals / clear them |
| GET | `/api/health` | liveness check |

## Changes from the notebook (and why)

- **Removed data leakage:** the notebook did `pd.concat([df, df])` before splitting, so every
  test sample also appeared in training and inflated the scores. Duplicates are now removed
  before a stratified 80/20 split, so expect lower but honest numbers on your real data.
- **One saved Pipeline** (TF-IDF + classifier) instead of pickling only the vocabulary and
  re-fitting the vectorizer at prediction time.
- **No NLTK downloads:** stop-words come from scikit-learn; WordNet lemmatization was dropped
  to keep it offline.
- The notebook's rule of dropping words of 3 letters or fewer is kept (`MIN_WORD_LEN` in
  `ml/preprocess.py`); lower it to 3 if you want words like "ass" or "die" to count.
- The notebook's demo printed "bulling"; the app uses "bullying".

This is a classic bag-of-words model: it can miss sarcasm, context and obfuscated spelling,
and can over-flag quoted or reclaimed language. Treat it as a screening aid, not a verdict.
