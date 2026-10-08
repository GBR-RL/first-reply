"""TF-IDF + logistic regression: the cheap baseline every other router has to beat."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from first_reply.routing.base import Prediction, Probabilities
from first_reply.routing.targets import TARGETS, tag_matrix


class TfidfRouter:
    name = "tfidf_lr"

    def __init__(self, c: float = 4.0) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000
        )
        self.c = c
        self.models: dict[str, LogisticRegression] = {}
        self.tagger: OneVsRestClassifier | None = None

    def _features(self, df: pd.DataFrame) -> csr_matrix:
        return csr_matrix(self.vectorizer.transform(df.text))

    def fit(self, train: pd.DataFrame, tags: list[str]) -> None:
        x = csr_matrix(self.vectorizer.fit_transform(train.text))
        for target in TARGETS:
            model = LogisticRegression(max_iter=2000, C=self.c, class_weight="balanced")
            self.models[target] = model.fit(x, train[target])
        if tags:
            tagger = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=self.c), n_jobs=-1)
            self.tagger = tagger.fit(x, tag_matrix(train, tags))

    def predict(self, df: pd.DataFrame) -> Prediction:
        x = self._features(df)
        targets = {
            t: Probabilities([str(c) for c in m.classes_], m.predict_proba(x))
            for t, m in self.models.items()
        }
        scores = None if self.tagger is None else np.asarray(self.tagger.predict_proba(x))
        return Prediction(targets, scores)
