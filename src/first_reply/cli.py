"""Command line entry point."""

from __future__ import annotations

from typing import Annotated

import typer

from first_reply import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Service-desk triage with local LLMs."""


@app.command()
def data() -> None:
    """Download the pinned datasets and prepare the ticket set and knowledge base."""
    from first_reply.data import download, techqa, tickets

    paths = download.fetch_all()
    typer.echo(f"downloaded {len(paths)} files")
    df = tickets.build()
    counts = df.groupby(["split", "lang"]).size().unstack(fill_value=0)
    typer.echo(f"tickets: {len(df)} unique\n{counts}")
    docs, questions, responses = techqa.build()
    answerable = questions.groupby("split").answerable.agg(["size", "sum"])
    typer.echo(
        f"knowledge base: {len(docs)} documents, {len(responses)} reference responses\n"
        f"questions (size, answerable):\n{answerable}"
    )


@app.command("families")
def families_cmd(
    threshold: float = typer.Option(0.90, help="Cosine similarity that links two tickets."),
) -> None:
    """Group tickets into paraphrase families from cached bge-m3 embeddings."""
    from first_reply import embed
    from first_reply.data import families, tickets

    df = tickets.clean(tickets.load_raw())
    out = families.assign(
        df.id.tolist(), embed.load("bge-m3", "tickets", df.id.tolist()), threshold
    )
    out.to_csv(families.ASSET, index=False, compression={"method": "gzip", "mtime": 0})
    sizes = out.family.value_counts()
    typer.echo(
        f"{len(sizes)} families, {int((sizes == 1).sum())} singletons, largest {sizes.max()}"
    )


@app.command("data-card")
def data_card() -> None:
    """Regenerate docs/data.md from the prepared datasets."""
    from first_reply.data import card

    typer.echo(f"wrote {card.write()}")


@app.command()
def route(method: str = typer.Option("tfidf_lr", help="Routing method to evaluate.")) -> None:
    """Train a routing method on train, report macro-F1 on test (95% bootstrap intervals)."""
    from first_reply import routing
    from first_reply.data import tickets
    from first_reply.routing.base import evaluate

    report = evaluate(routing.make(method), tickets.load())
    for target, r in report["targets"].items():
        lo, hi = r["macro_f1_ci"]
        f1, acc = r["macro_f1"], r["accuracy"]
        typer.echo(f"{target:9s} macro-F1 {f1:.3f} [{lo:.3f}, {hi:.3f}]  acc {acc:.3f}")
    if "tags" in report:
        tags = report["tags"]
        typer.echo(f"tags      micro-F1 {tags['micro_f1']:.3f} ({tags['vocab_size']} tags)")


@app.command("leakage-check")
def leakage_check() -> None:
    """Score the TF-IDF baseline on a split drawn with and without de-duplication."""
    import json

    from first_reply.routing.base import RESULTS, rounded
    from first_reply.routing.leakage import check

    report = rounded(check())
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "leakage_check.json").write_text(json.dumps(report, indent=2) + "\n")
    typer.echo(json.dumps(report, indent=2))


@app.command()
def embed(
    model: str = typer.Option("e5-small", help="Embedding model key."),
    corpus: str = typer.Option("tickets", help="Which texts to encode."),
    shard: int = typer.Option(0, help="Shard index (for parallel CI jobs)."),
    shards: int = typer.Option(1, help="Number of shards."),
) -> None:
    """Encode a corpus (or one shard of it) into the embedding cache."""
    from first_reply import corpora
    from first_reply import embed as emb

    ids, texts, kind = corpora.load(corpus)
    path = emb.compute(ids, texts, model, corpus, kind=kind, shard_index=shard, shard_count=shards)
    typer.echo(f"wrote {path}")


@app.command("embed-merge")
def embed_merge(
    model: str = typer.Option(..., help="Embedding model key."),
    corpus: str = typer.Option(..., help="Corpus name."),
) -> None:
    """Merge shard files produced by parallel `embed` runs."""
    from first_reply import embed as emb

    typer.echo(f"wrote {emb.merge_shards(model, corpus)}")


@app.command("kb-chunks")
def kb_chunks() -> None:
    """Cut the knowledge base into chunks with every chunker."""
    from first_reply.data import techqa
    from first_reply.kb import chunking
    from first_reply.kb.index import chunks_path

    docs = techqa.load_docs()
    for name in chunking.CHUNKERS:
        chunks = chunking.chunk_corpus(docs, name)
        chunks.to_parquet(chunks_path(name), index=False)
        typer.echo(f"{name}: {len(chunks)} chunks, median {chunks.words.median():.0f} words")


@app.command("kb-index")
def kb_index(
    chunker: str = typer.Option("sections", help="Chunker whose chunks to index."),
) -> None:
    """Build the collection (BM25 + cached dense vectors) in the Qdrant server at QDRANT_URL."""
    from first_reply.kb import index

    n = index.build(chunker)
    typer.echo(f"indexed {n} chunks; dense vectors: {index.dense_models(chunker) or 'none'}")


@app.command("retrieval-eval")
def retrieval_eval(
    chunker: str = typer.Option("sections"),
    mode: str = typer.Option("hybrid", help="bm25, dense or hybrid"),
    model: str = typer.Option("e5-small", help="Dense embedding model."),
    split: Annotated[
        list[str] | None, typer.Option(help="Question splits (default: test).")
    ] = None,
) -> None:
    """Score document retrieval on the answerable TechQA questions."""
    from first_reply.kb.evaluate import evaluate

    s = evaluate(chunker, mode, model, splits=tuple(split or ["test"]))
    lo, hi = s["ndcg@10_ci"]
    typer.echo(
        f"{s['run']}: recall@5 {s['recall@5']:.3f}  nDCG@10 {s['ndcg@10']:.3f} "
        f"[{lo:.3f}, {hi:.3f}]  MRR@10 {s['mrr@10']:.3f}  ({s['questions']} questions)"
    )


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
