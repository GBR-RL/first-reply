"""Reports over finished runs: LLM triage, drafting, the judge, and how much can be automated."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from first_reply.config import ROOT, RUNS
from first_reply.data import techqa, tickets
from first_reply.eval.llm_runs import run_dir
from first_reply.eval.metrics import bootstrap_ci, classification_report, multilabel_report
from first_reply.routing.base import rounded
from first_reply.routing.targets import TARGETS, tag_matrix, tag_vocab

RESULTS = ROOT / "docs" / "results"
ERROR = "<error>"


def _load(name: str) -> pd.DataFrame:
    return pd.read_json(run_dir(name) / "merged.jsonl", lines=True, dtype={"id": str})


def _write(sub: str, name: str, report: dict[str, Any]) -> Path:
    path = RESULTS / sub / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rounded(report), indent=2, ensure_ascii=False) + "\n")
    return path


def ops(df: pd.DataFrame) -> dict[str, Any]:
    ok = df[df.get("error", pd.Series(index=df.index, dtype=object)).isna()]
    out: dict[str, Any] = {"items": len(df), "errors": len(df) - len(ok)}
    for col in ("prompt_tokens", "completion_tokens", "latency_s"):
        if col in ok:
            out[col] = {
                "mean": float(ok[col].mean()),
                "p50": float(ok[col].quantile(0.5)),
                "p95": float(ok[col].quantile(0.95)),
            }
    if "cost" in ok and ok.cost.notna().any():
        out["chargeback_cost_per_1000_items"] = float(ok.cost.mean() * 1000)
    if "latency_s" in ok and "completion_tokens" in ok:
        out["output_tokens_per_s"] = float(ok.completion_tokens.sum() / ok.latency_s.sum())
    return out


def triage_report(name: str, classical: tuple[str, ...] = ()) -> dict[str, Any]:
    """LLM triage on the fixed 1,000-ticket test subset, next to classical routers."""
    runs = _load(name)
    df = tickets.load()
    gold = df.set_index("id").loc[runs.id]
    pred = pd.json_normalize(runs.pred.where(runs.pred.notna(), None).map(lambda p: p or {}))
    report: dict[str, Any] = {"run": name, "targets": {}}
    for t in TARGETS:
        p = pred[t].fillna(ERROR).to_numpy() if t in pred else np.full(len(runs), ERROR)
        report["targets"][t] = classification_report(
            gold[t].to_numpy(), p, gold.lang.to_numpy(), n_boot=500
        )
    vocab = tag_vocab(df[df.split == "train"])
    pred_tags = pd.DataFrame({"tags": pred.get("tags", pd.Series([[]] * len(runs)))})
    pred_tags["tags"] = pred_tags.tags.map(lambda x: x if isinstance(x, list) else [])
    report["tags"] = multilabel_report(tag_matrix(gold, vocab), tag_matrix(pred_tags, vocab))
    report["ops"] = ops(runs)
    # Classical routers scored on exactly the same tickets.
    same: dict[str, Any] = {}
    for method in classical:
        path = RUNS / "routing" / method / "predictions_test.parquet"
        if not path.exists():
            continue
        cls = pd.read_parquet(path).set_index("id").loc[runs.id]
        same[method] = {
            t: float(f1_score(gold[t], cls[f"{t}_pred"], average="macro")) for t in TARGETS
        }
    report["classical_on_same_tickets"] = same
    _write("llm", name, report)
    return report


def draft_report(name: str, judged: str | None = None) -> dict[str, Any]:
    """Decline behaviour and grounding of drafted replies."""
    runs = _load(name)
    q = techqa.load_questions().set_index("qid").loc[runs.id]
    ok = runs.get("error", pd.Series(index=runs.index, dtype=object)).isna().to_numpy()
    answered = runs.answerable.fillna(False).astype(bool).to_numpy() & ok
    gold_shown = np.array(
        [
            bool(set(g) & set(r or []))
            for g, r in zip(q.gold_doc_ids, runs.retrieved_doc_ids, strict=True)
        ]
    )
    cited_gold = np.array(
        [
            bool(set(g) & set(c or []))
            for g, c in zip(q.gold_doc_ids, runs.cited_doc_ids, strict=True)
        ]
    )
    cite_prec = [
        len(set(g) & set(c)) / len(c)
        for g, c, a in zip(q.gold_doc_ids, runs.cited_doc_ids, answered, strict=True)
        if a and c
    ]
    unanswerable = ~q.answerable.to_numpy()
    report: dict[str, Any] = {
        "run": name,
        "questions": len(runs),
        "answer_rate": float(answered.mean()),
        "gold_document_shown": float(gold_shown.mean()),
        # Of the answered questions, how many cite a relevant document.
        "answered_citing_gold": float(cited_gold[answered].mean()) if answered.any() else None,
        "citation_precision": float(np.mean(cite_prec)) if cite_prec else None,
        # Declining is right when no relevant document was shown to the model.
        "declined_when_gold_not_shown": float((~answered[~gold_shown]).mean()),
        "answered_when_gold_shown": float(answered[gold_shown].mean()),
        "answered_unanswerable_questions": float(answered[unanswerable].mean()),
        "ops": ops(runs),
    }
    if judged:
        j = _load(judged).set_index("id")
        j = j[j.get("error", pd.Series(index=j.index, dtype=object)).isna()]
        report["judged_answers"] = len(j)
        report["judge_supported_share"] = float(j.supported.mean())
        sup = j.supported.astype(float).to_numpy()
        report["judge_supported_ci"] = bootstrap_ci(
            lambda a, _b: float(np.mean(a)), sup, sup, n_boot=1000
        )
    _write("drafting", name, report)
    return report


def judge_validation(name: str) -> dict[str, Any]:
    """Agreement of our judge with RAGBench's GPT-4 support labels."""
    runs = _load(name)
    ok = runs[runs.get("error", pd.Series(index=runs.index, dtype=object)).isna()]
    y, p = ok.label_supported.astype(bool), ok.supported.astype(bool)
    report = {
        "run": name,
        "responses": len(ok),
        "errors": len(runs) - len(ok),
        "accuracy": float((y == p).mean()),
        "cohen_kappa": float(cohen_kappa_score(y, p)),
        # How often an unsupported response is passed as supported.
        "false_pass_rate": float(p[~y].mean()),
        "false_fail_rate": float((~p[y]).mean()),
        "ops": ops(runs),
    }
    _write("judge", name, report)
    return report


def automation(method: str, precision: float = 0.95) -> dict[str, Any]:
    """Share of test tickets a router could route alone at a target precision.

    The confidence threshold (top class probability) is chosen on dev as the lowest one whose
    dev precision reaches the target; coverage and precision are then measured on test.
    """
    run = RUNS / "routing" / method
    out: dict[str, Any] = {"method": method, "target_precision": precision, "targets": {}}
    dev = pd.read_parquet(run / "predictions_dev.parquet")
    test = pd.read_parquet(run / "predictions_test.parquet")
    for t in TARGETS:
        conf_dev = dev[f"{t}_proba"].map(max).to_numpy()
        right_dev = (dev[f"{t}_pred"] == dev[t]).to_numpy()
        order = np.argsort(-conf_dev)
        prec = np.cumsum(right_dev[order]) / np.arange(1, len(order) + 1)
        ok = np.nonzero(prec >= precision)[0]
        # Largest prefix of confident tickets whose precision still meets the target.
        threshold = float(conf_dev[order][ok.max()]) if ok.size else float("inf")
        conf = test[f"{t}_proba"].map(max).to_numpy()
        take = conf >= threshold
        right = (test[f"{t}_pred"] == test[t]).to_numpy()
        out["targets"][t] = {
            "threshold_from_dev": threshold,
            "coverage_test": float(take.mean()),
            "precision_test": float(right[take].mean()) if take.any() else None,
        }
    _write("automation", f"{method}_p{int(precision * 100)}", out)
    return out


def _coverage(take: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    return {
        "coverage": float(take.mean()),
        "precision": float(right[take].mean()) if take.any() else None,
    }


def gate_report(
    llm_run: str, method: str = "bge-m3_knn", precision: float = 0.95
) -> dict[str, Any]:
    """The service's routing gate on the LLM subset: kNN confidence threshold and LLM agreement."""
    llm = _load(llm_run).set_index("id")
    knn = pd.read_parquet(RUNS / "routing" / method / "predictions_test.parquet").set_index("id")
    knn = knn.loc[llm.index]
    gold = tickets.load().set_index("id").loc[llm.index]
    auto = json.loads(
        (RESULTS / "automation" / f"{method}_p{int(precision * 100)}.json").read_text()
    )
    out: dict[str, Any] = {"llm_run": llm_run, "method": method, "tickets": len(llm), "targets": {}}
    for t in TARGETS:
        llm_label = llm.pred.map(lambda p, t=t: (p or {}).get(t))
        right = (knn[f"{t}_pred"] == gold[t]).to_numpy()
        agree = (llm_label == knn[f"{t}_pred"]).to_numpy()
        confident = (
            knn[f"{t}_proba"].map(max) >= auto["targets"][t]["threshold_from_dev"]
        ).to_numpy()
        gate = agree & confident
        out["targets"][t] = {
            "agreement": float(agree.mean()),
            "precision_when_agree": float(right[agree].mean()),
            "threshold_only": _coverage(confident, right),
            "threshold_and_agreement": _coverage(gate, right),
        }
    _write("automation", f"gate_{llm_run}", out)
    return out
