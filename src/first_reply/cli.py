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
    """Download the pinned datasets and prepare the ticket set."""
    from first_reply.data import download, tickets

    paths = download.fetch_all()
    typer.echo(f"downloaded {len(paths)} files")
    df = tickets.build()
    counts = df.groupby(["split", "language"]).size().unstack(fill_value=0)
    typer.echo(f"tickets: {len(df)} unique\n{counts}")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
