"""Fine-tuned multilingual encoder with one head per routing target.

multilingual-e5-small (118M parameters) is fine-tuned end to end with a shared encoder and four
heads: queue, priority and type (cross-entropy) and the tag vocabulary (binary cross-entropy).
The epoch with the best mean dev macro-F1 over the three single-label targets is kept.
Training runs on CPU, so this is meant for a CI runner rather than a laptop.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from first_reply.config import SEED
from first_reply.eval.metrics import macro_f1
from first_reply.routing.base import Prediction, Probabilities
from first_reply.routing.targets import TARGETS, tag_matrix


class FinetunedRouter:
    def __init__(
        self,
        hf_name: str = "intfloat/multilingual-e5-small",
        *,
        epochs: int = 2,
        batch_size: int = 32,
        lr: float = 5e-5,
        max_length: int = 192,
        dev_eval_size: int = 2000,
    ) -> None:
        self.name = "e5-small_ft"
        self.hf_name = hf_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.dev_eval_size = dev_eval_size
        self.classes: dict[str, list[str]] = {}
        self.tags: list[str] = []
        self.model: Any = None
        self.tokenizer: Any = None
        self.history: list[dict[str, float]] = []

    # The torch module is built lazily so importing this file needs no torch.
    def _build(self) -> Any:
        import torch
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(self.hf_name)
        hidden = encoder.config.hidden_size
        sizes = {t: len(c) for t, c in self.classes.items()}
        n_tags = len(self.tags)

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = encoder
                self.dropout = torch.nn.Dropout(0.1)
                self.heads = torch.nn.ModuleDict(
                    {f"head_{t}": torch.nn.Linear(hidden, n) for t, n in sizes.items()}
                )
                self.tag_head = torch.nn.Linear(hidden, max(n_tags, 1))

            def forward(self, ids: Any, mask: Any) -> dict[str, Any]:
                out = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                m = mask.unsqueeze(-1).float()
                pooled = self.dropout((out * m).sum(1) / m.sum(1).clamp(min=1.0))
                logits = {k.removeprefix("head_"): h(pooled) for k, h in self.heads.items()}
                logits["tags"] = self.tag_head(pooled)
                return logits

        return Net()

    def _tokenize(self, texts: list[str]) -> Any:
        return self.tokenizer(
            ["query: " + t for t in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _logits(self, df: pd.DataFrame) -> dict[str, NDArray[np.float64]]:
        import torch

        self.model.eval()
        chunks: dict[str, list[NDArray[np.float64]]] = {}
        texts = df.text.tolist()
        with torch.inference_mode():
            for start in range(0, len(texts), 128):
                batch = self._tokenize(texts[start : start + 128])
                out = self.model(batch["input_ids"], batch["attention_mask"])
                for k, v in out.items():
                    chunks.setdefault(k, []).append(v.float().numpy())
        return {k: np.concatenate(v) for k, v in chunks.items()}

    def _dev_score(self, dev: pd.DataFrame) -> float:
        logits = self._logits(dev)
        scores = []
        for t in TARGETS:
            pred = np.asarray(self.classes[t])[logits[t].argmax(1)]
            scores.append(macro_f1(dev[t].to_numpy(), pred))
        return float(np.mean(scores))

    def fit(self, train: pd.DataFrame, tags: list[str], dev: pd.DataFrame | None = None) -> None:
        import torch
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup

        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        self.tags = tags
        self.classes = {t: sorted(train[t].unique().tolist()) for t in TARGETS}
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_name)
        self.model = self._build()

        y = {
            t: torch.tensor(train[t].map({c: i for i, c in enumerate(self.classes[t])}).to_numpy())
            for t in TARGETS
        }
        y_tags = torch.tensor(tag_matrix(train, tags), dtype=torch.float32)
        texts = train.text.tolist()
        steps = math.ceil(len(texts) / self.batch_size) * self.epochs
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        ce, bce = torch.nn.CrossEntropyLoss(), torch.nn.BCEWithLogitsLoss()
        if dev is not None and len(dev) > self.dev_eval_size:
            dev = dev.sample(self.dev_eval_size, random_state=SEED)

        best: tuple[float, dict[str, Any]] | None = None
        for epoch in range(self.epochs):
            self.model.train()
            order = rng.permutation(len(texts))
            t0, total = time.perf_counter(), 0.0
            for step, start in enumerate(range(0, len(order), self.batch_size), 1):
                idx = order[start : start + self.batch_size]
                batch = self._tokenize([texts[i] for i in idx])
                out = self.model(batch["input_ids"], batch["attention_mask"])
                loss = sum(ce(out[t], y[t][idx]) for t in TARGETS)
                if tags:
                    loss = loss + bce(out["tags"], y_tags[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += float(loss.detach())
                if step % 100 == 0:
                    rate = step * self.batch_size / (time.perf_counter() - t0)
                    print(f"epoch {epoch} step {step} loss {total / step:.3f} ({rate:.1f}/s)")
            record = {"epoch": epoch, "train_loss": total / max(step, 1)}
            if dev is not None:
                record["dev_mean_macro_f1"] = self._dev_score(dev)
                score = record["dev_mean_macro_f1"]
                if best is None or score > best[0]:
                    state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    best = (score, state)
            self.history.append(record)
            print(record)
        if best is not None:
            self.model.load_state_dict(best[1])

    def predict(self, df: pd.DataFrame) -> Prediction:
        logits = self._logits(df)
        targets = {}
        for t in TARGETS:
            z = logits[t] - logits[t].max(1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
            targets[t] = Probabilities(self.classes[t], p)
        scores = 1 / (1 + np.exp(-logits["tags"])) if self.tags else None
        extra = {"history": self.history, "epochs": self.epochs, "lr": self.lr}
        return Prediction(targets, scores, extra)
