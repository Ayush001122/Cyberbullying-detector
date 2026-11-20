"""Loads the saved pipeline and turns raw text into a prediction + explanation."""
import math
import re
from pathlib import Path

import joblib
import numpy as np

from .preprocess import MIN_WORD_LEN, clean_text, normalize_token
from .train import MODEL_PATH, train


class Predictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not Path(model_path).exists():
            print("No trained model found - training one now...")
            train()
        self.pipe = joblib.load(model_path)
        self.tfidf = self.pipe.named_steps["tfidf"]
        self.clf = self.pipe.named_steps["clf"]
        self.vocab = self.tfidf.vocabulary_
        self.coef = np.asarray(self.clf.coef_).ravel()
        self.max_coef = max(float(self.coef.max()), 1e-9)

    @staticmethod
    def _severity(score: float) -> str:
        if score <= 0:
            return "none"
        if score < 0.6:
            return "low"
        if score < 1.3:
            return "medium"
        return "high"

    def predict(self, text: str) -> dict:
        text = (text or "").strip()
        cleaned = clean_text(text)
        if not cleaned:
            score = 0.0
        else:
            score = float(self.pipe.decision_function([cleaned])[0])
        toxic = score > 0
        confidence = 1 / (1 + math.exp(-abs(score) * 2.5))   # squashed margin -> 0.5..1.0
        return {
            "text": text,
            "label": "bullying" if toxic else "non-bullying",
            "is_toxic": bool(toxic),
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "toxicity": round(1 / (1 + math.exp(-score * 2.5)), 4),
            "severity": self._severity(score),
            "tokens": self.explain(text),
        }

    def explain(self, text: str):
        """Per-word contribution so the UI can highlight what triggered the flag."""
        out = []
        for raw in re.findall(r"\S+", text):
            norm = normalize_token(raw)
            weight = 0.0
            if len(norm) >= MIN_WORD_LEN and norm in self.vocab:
                weight = float(self.coef[self.vocab[norm]])
            out.append({"word": raw, "weight": round(max(weight, 0) / self.max_coef, 3)})
        return out
