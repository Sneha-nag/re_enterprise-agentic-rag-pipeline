from typer.testing import CliRunner

from meck_agent.cli import app

runner = CliRunner()


def test_cli_paths():
    result = runner.invoke(app, ["paths"])
    assert result.exit_code == 0
    assert "db:" in result.stdout
    assert "chroma:" in result.stdout


def test_cli_ui_help():
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "Streamlit" in result.stdout
