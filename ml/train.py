"""Train + compare the notebook's models and save the production model.

Usage:
    python -m ml.train                    # uses data/dataset.csv
    python -m ml.train --data my.csv      # custom CSV (columns: headline,label)
    python -m ml.train --slow             # also train BaggingClassifier (slow on big data)

Improvements over the notebook:
  * the notebook did pd.concat([df, df]) before splitting, so every test row
    also existed in the training set (data leakage -> inflated scores).
    Here duplicates are removed BEFORE a stratified split.
  * TF-IDF + classifier are saved as ONE sklearn Pipeline (no hand-rolled
    vocabulary pickle / re-fitting the vectorizer at prediction time).
  * stop-words come from scikit-learn, so no NLTK download is needed.
"""
import argparse
import json
from pathlib import Path
from time import time

import joblib
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DEFAULT_DATA = ROOT / "data" / "dataset.csv"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    text_col = "headline" if "headline" in df.columns else "text"
    df = df[[text_col, "label"]].rename(columns={text_col: "text"})
    df["label"] = (df["label"] != 0).astype(int)          # -1 / 1 -> 1 (toxic), 0 -> 0
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() > 0]
    df = df.drop_duplicates(subset="clean").reset_index(drop=True)  # prevents train/test leakage
    return df


def make_pipeline(clf) -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(stop_words="english", lowercase=True)), ("clf", clf)])


def evaluate(pipe, X_tr, y_tr, X_te, y_te, name, train_time):
    t = time()
    pred_te = pipe.predict(X_te)
    pred_time = time() - t
    pred_tr = pipe.predict(X_tr)
    return {
        "algorithm": name,
        "accuracy": accuracy_score(y_te, pred_te),
        "precision": precision_score(y_te, pred_te, zero_division=0),
        "recall": recall_score(y_te, pred_te, zero_division=0),
        "f1": f1_score(y_te, pred_te, zero_division=0),
        "train_accuracy": accuracy_score(y_tr, pred_tr),
        "train_f1": f1_score(y_tr, pred_tr, zero_division=0),
        "train_time": train_time,
        "predict_time": pred_time,
    }


def train(data_path: Path = DEFAULT_DATA, include_slow: bool = False, verbose: bool = True):
    log = print if verbose else (lambda *a, **k: None)
    df = load_data(data_path)
    log(f"Loaded {len(df)} unique rows from {data_path}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    models = {
        "LinearSVC": LinearSVC(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "MultinomialNB": MultinomialNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "SGDClassifier": SGDClassifier(random_state=42),
    }
    if include_slow:
        models["Bagging"] = BaggingClassifier(random_state=42)

    results = []
    for name, clf in models.items():
        pipe = make_pipeline(clf)
        t = time()
        pipe.fit(X_tr, y_tr)
        res = evaluate(pipe, X_tr, y_tr, X_te, y_te, name, time() - t)
        results.append(res)
        log(f"  {name:<20} acc={res['accuracy']:.4f} f1={res['f1']:.4f} ({res['train_time']:.2f}s)")

    # fine-tune LinearSVC exactly like the notebook (grid over C, 5-fold CV, F1)
    grid = GridSearchCV(make_pipeline(LinearSVC()), {"clf__C": [0.25, 0.5, 0.75, 1, 1.2]},
                        scoring=make_scorer(f1_score), cv=5)
    t = time()
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    tuned = evaluate(best, X_tr, y_tr, X_te, y_te, "LinearSVC (tuned)", time() - t)
    results.append(tuned)
    log(f"  {tuned['algorithm']:<20} acc={tuned['accuracy']:.4f} f1={tuned['f1']:.4f} best={grid.best_params_}")

    tn, fp, fn, tp = confusion_matrix(y_te, best.predict(X_te), labels=[0, 1]).ravel()
    results.sort(key=lambda r: r["f1"], reverse=True)

    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "toxic": int(df["label"].sum()),
            "non_toxic": int((df["label"] == 0).sum()),
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
            "source": str(data_path.name),
        },
        "production_model": {"algorithm": "LinearSVC (tuned)", "params": grid.best_params_},
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "results": results,
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log(f"Saved model -> {MODEL_PATH}\nSaved metrics -> {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--slow", action="store_true", help="also train BaggingClassifier")
    a = ap.parse_args()
    train(a.data, a.slow)
