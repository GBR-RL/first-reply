"""Routers on cached sentence embeddings: a linear probe and a nearest-neighbour vote."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from first_reply import embed
from first_reply.routing.base import Prediction, Probabilities
from first_reply.routing.targets import TARGETS, tag_matrix

CORPUS = "tickets"


def _vectors(df: pd.DataFrame, model_key: str) -> NDArray[np.float32]:
    return embed.load(model_key, CORPUS, df.id.tolist())


class EmbeddingLRRouter:
    """Logistic regression on frozen embeddings."""

    def __init__(self, model_key: str, c: float = 4.0) -> None:
        self.model_key = model_key
        self.name = f"{model_key}_lr"
        self.c = c
        self.models: dict[str, LogisticRegression] = {}
        self.tagger: OneVsRestClassifier | None = None

    def fit(self, train: pd.DataFrame, tags: list[str]) -> None:
        x = _vectors(train, self.model_key)
        for target in TARGETS:
            model = LogisticRegression(max_iter=3000, C=self.c, class_weight="balanced")
            self.models[target] = model.fit(x, train[target])
        if tags:
            tagger = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=self.c), n_jobs=-1)
            self.tagger = tagger.fit(x, tag_matrix(train, tags))

    def predict(self, df: pd.DataFrame) -> Prediction:
        x = _vectors(df, self.model_key)
        targets = {
            t: Probabilities([str(c) for c in m.classes_], m.predict_proba(x))
            for t, m in self.models.items()
        }
        scores = None if self.tagger is None else np.asarray(self.tagger.predict_proba(x))
        return Prediction(targets, scores)


class KnnRouter:
    """Similarity-weighted vote of the k most similar training tickets.

    This is the retrieval step of the triage workflow used directly as a classifier: the
    neighbours are the same "similar past tickets" a human reviewer would see.
    """

    def __init__(self, model_key: str, k: int = 25, temperature: float = 0.05) -> None:
        self.model_key = model_key
        self.name = f"{model_key}_knn"
        self.k = k
        self.temperature = temperature
        self.train: pd.DataFrame | None = None
        self.x: NDArray[np.float32] | None = None
        self.tags: list[str] = []

    def fit(self, train: pd.DataFrame, tags: list[str]) -> None:
        self.train = train.reset_index(drop=True)
        self.x = _vectors(train, self.model_key)
        self.tags = tags

    def neighbours(self, q: NDArray[np.float32]) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        assert self.x is not None
        sims = q @ self.x.T
        idx = np.argpartition(-sims, self.k, axis=1)[:, : self.k]
        top = np.take_along_axis(sims, idx, axis=1)
        weights = np.exp((top - top.max(axis=1, keepdims=True)) / self.temperature)
        return idx, weights / weights.sum(axis=1, keepdims=True)

    def predict(self, df: pd.DataFrame) -> Prediction:
        assert self.train is not None
        q = _vectors(df, self.model_key)
        idx, w = self.neighbours(q)
        targets = {}
        for target in TARGETS:
            classes = sorted(self.train[target].unique())
            codes = self.train[target].map({c: i for i, c in enumerate(classes)}).to_numpy()
            proba = np.zeros((len(df), len(classes)))
            for j in range(self.k):
                np.add.at(proba, (np.arange(len(df)), codes[idx[:, j]]), w[:, j])
            targets[target] = Probabilities([str(c) for c in classes], proba)
        scores = None
        if self.tags:
            m = tag_matrix(self.train, self.tags).astype(np.float64)
            scores = np.einsum("nk,nkt->nt", w, m[idx])
        return Prediction(targets, scores, {"k": self.k, "temperature": self.temperature})
