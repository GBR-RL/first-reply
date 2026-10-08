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
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
