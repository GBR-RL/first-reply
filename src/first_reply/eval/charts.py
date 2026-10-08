"""README charts from the committed result files, in a light and a dark variant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from first_reply.config import ROOT

RESULTS = ROOT / "docs" / "results"
ASSETS = ROOT / "docs" / "assets"

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "s1": "#2a78d6", "s2": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "muted": "#c3c2b7", "grid": "#33322f",
             "s1": "#3987e5", "s2": "#d95926"},
}  # fmt: skip


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _style(ax: Any, t: dict[str, str]) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: Any, name: str, theme: str) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"{name}-{theme}.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return path


def leakage(theme: str) -> Path:
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    data = _json(RESULTS / "routing" / "leakage_check.json")
    splits = list(data)
    labels = ["random,\nwith duplicates", "random,\nde-duplicated", "families\n>= 0.95",
              "families\n>= 0.93", "families\n>= 0.90"]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=t["surface"])
    _style(ax, t)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for key, name, color in (("tfidf_lr", "TF-IDF + LR", t["s1"]),
                             ("bge-m3_knn", "bge-m3 nearest neighbours", t["s2"])):  # fmt: skip
        xs = [i for i, s in enumerate(splits) if key in data[s]]
        ys = [data[splits[i]][key]["queue"] for i in xs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7, label=name,
                markeredgecolor=t["surface"], markeredgewidth=2)  # fmt: skip
        ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=t["ink"], fontsize=9)  # fmt: skip
        ax.annotate(f"{ys[0]:.2f}", (xs[0], ys[0]), xytext=(0, 8), textcoords="offset points",
                    ha="center", color=t["ink"], fontsize=9)  # fmt: skip
    ax.set_xticks(range(len(splits)), labels)
    ax.set_ylabel("queue macro-F1 (test)", color=t["muted"], fontsize=9)
    ax.set_ylim(0, 0.8)
    ax.set_xlim(-0.4, len(splits) - 0.5)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in leg.get_texts():
        text.set_color(t["ink"])
    ax.set_title("The same router, five ways to split the data", loc="left", color=t["ink"],
                 fontsize=11, fontweight="bold")  # fmt: skip
    path = _save(fig, "leakage", theme)
    plt.close(fig)
    return path


ROUTERS = (
    ("tfidf_lr", "TF-IDF + LR"),
    ("e5-small_lr", "e5-small + LR"),
    ("e5-small_knn", "e5-small kNN"),
    ("bge-m3_lr", "bge-m3 + LR"),
    ("bge-m3_knn", "bge-m3 kNN"),
    ("e5-small_ft", "e5-small fine-tuned"),
)


def routing(theme: str) -> Path:
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    runs = {k: _json(RESULTS / "routing" / f"{k}.json") for k, _ in ROUTERS}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True, facecolor=t["surface"])
    names = [n for _, n in ROUTERS]
    for ax, target in zip(axes, ("queue", "priority", "type"), strict=True):
        _style(ax, t)
        vals = [runs[k]["targets"][target]["macro_f1"] for k, _ in ROUTERS]
        best = max(vals)
        ys = range(len(vals))
        ax.barh(ys, vals, height=0.55, color=t["s1"])
        for y, v in zip(ys, vals, strict=True):
            ax.text(v + 0.015, y, f"{v:.2f}", va="center", fontsize=8.5,
                    color=t["ink"], fontweight="bold" if v == best else "normal")  # fmt: skip
        ax.set_xlim(0, 1)
        ax.set_title(target, loc="left", color=t["ink"], fontsize=10)
        ax.invert_yaxis()
    axes[0].set_yticks(range(len(names)), names)
    fig.suptitle("Routing unseen ticket families: macro-F1 on 8,032 test tickets", x=0.01,
                 ha="left", color=t["ink"], fontsize=11, fontweight="bold")  # fmt: skip
    fig.tight_layout()
    path = _save(fig, "routing", theme)
    plt.close(fig)
    return path


RETRIEVAL = (
    ("sections_bm25", "BM25"),
    ("sections_dense_e5-small", "dense e5-small"),
    ("sections_dense_bge-m3", "dense bge-m3"),
    ("sections_hybrid_e5-small", "hybrid e5-small"),
    ("sections_hybrid_bge-m3", "hybrid bge-m3"),
    ("sections_bm25_rerank-minilm", "BM25 + mMiniLM reranker"),
    ("sections_bm25_rerank-bge-reranker", "BM25 + bge reranker"),
)


def retrieval(theme: str) -> Path:
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    rows = [(n, _json(RESULTS / "retrieval" / f"{k}.json")) for k, n in RETRIEVAL]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=t["surface"])
    _style(ax, t)
    ys = range(len(rows))
    vals = [r["ndcg@10"] for _, r in rows]
    lo = [r["ndcg@10"] - r["ndcg@10_ci"][0] for _, r in rows]
    hi = [r["ndcg@10_ci"][1] - r["ndcg@10"] for _, r in rows]
    ax.barh(ys, vals, height=0.55, color=t["s1"])
    ax.errorbar(vals, ys, xerr=[lo, hi], fmt="none", ecolor=t["muted"], elinewidth=1.2, capsize=3)
    best = max(vals)
    for y, v, h in zip(ys, vals, hi, strict=True):
        ax.text(v + h + 0.015, y, f"{v:.3f}", va="center", fontsize=8.5, color=t["ink"],
                fontweight="bold" if v == best else "normal")  # fmt: skip
    ax.set_yticks(list(ys), [n for n, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("nDCG@10 on the 118 answerable test questions (95% bootstrap interval)",
                  color=t["muted"], fontsize=9)  # fmt: skip
    ax.set_title("Finding the right IBM support document", loc="left", color=t["ink"],
                 fontsize=11, fontweight="bold")  # fmt: skip
    path = _save(fig, "retrieval", theme)
    plt.close(fig)
    return path


def all_charts() -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    return [fn(theme) for fn in (leakage, routing, retrieval) for theme in THEMES]
