"""Command line entry point."""

from __future__ import annotations

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
    counts = df.groupby(["split", "language"]).size().unstack(fill_value=0)
    typer.echo(f"tickets: {len(df)} unique\n{counts}")
    docs, questions, responses = techqa.build()
    answerable = questions.groupby("split").answerable.agg(["size", "sum"])
    typer.echo(
        f"knowledge base: {len(docs)} documents, {len(responses)} reference responses\n"
        f"questions (size, answerable):\n{answerable}"
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
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
