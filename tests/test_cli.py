from typer.testing import CliRunner

from first_reply import __version__
from first_reply.cli import app


def test_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
