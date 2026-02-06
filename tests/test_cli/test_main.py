from __future__ import annotations

from typer.testing import CliRunner

from zutils.cli.main import create_app

from loguru import logger


runner = CliRunner()


def test_verbose_flag_enables_debug_logs(monkeypatch) -> None:
    """Arrange–Act–Assert: debug logs should only appear with --verbose."""
    cli_app = create_app()

    def combined_output(result) -> str:
        """Return stdout plus stderr (Typer/Click runner captures both)."""

        stdout = getattr(result, "stdout", result.output)
        stderr = getattr(result, "stderr", "")
        return stdout + stderr

    @cli_app.command("verbose-check")
    def verbose_check() -> None:
        """Emit one debug and one info log line."""

        logger.debug("verbose-check debug")
        logger.info("verbose-check info")

    # Act
    result_default = runner.invoke(cli_app, ["verbose-check"])
    result_verbose = runner.invoke(cli_app, ["--verbose", "verbose-check"])

    # Assert
    assert result_default.exit_code == 0
    assert "verbose-check info" in combined_output(result_default)
    assert "verbose-check debug" not in combined_output(result_default)

    assert result_verbose.exit_code == 0
    assert "verbose-check info" in combined_output(result_verbose)
    assert "verbose-check debug" in combined_output(result_verbose)
