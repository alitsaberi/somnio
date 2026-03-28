from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from somnio.cli.main import create_app
from loguru import logger


runner = CliRunner()

# ANSI escape sequence pattern (e.g. colors, bold) so we can assert on plain text.
# Help output can differ between TTY and CI; stripping ANSI makes assertions stable.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _combined_output(result) -> str:
    stdout = getattr(result, "stdout", result.output)
    stderr = getattr(result, "stderr", "")
    return stdout + stderr


def _plain_output(result) -> str:
    """Combined stdout+stderr with ANSI escape sequences removed."""
    return _ANSI_RE.sub("", _combined_output(result))


def test_log_file_option_enables_file_logging() -> None:
    """With -f/--log-file, debug logs are written to the given file."""
    cli_app = create_app()

    @cli_app.command("log-check")
    def log_check() -> None:
        logger.debug("log-check debug")
        logger.info("log-check info")

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "debug.log"
        result = runner.invoke(cli_app, ["--log-file", str(log_path), "log-check"])
        assert result.exit_code == 0
        assert log_path.exists()
        assert "log-check debug" in log_path.read_text()
        assert "log-check info" in log_path.read_text()


def test_file_logging_option_uses_default_path() -> None:
    """With -l/--file-logging, logs are written to logs/{command}_{timestamp}.log."""
    cli_app = create_app()

    @cli_app.command("log-check")
    def log_check() -> None:
        logger.debug("log-check debug")
        logger.info("log-check info")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        orig_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(cli_app, ["--file-logging", "log-check"])
        finally:
            os.chdir(orig_cwd)
        assert result.exit_code == 0
        logs_dir = tmp_path / "logs"
        assert logs_dir.exists()
        log_files = list(logs_dir.glob("*.log"))
        assert len(log_files) >= 1
        content = log_files[0].read_text()
        assert "log-check debug" in content
        assert "log-check info" in content


def test_no_file_logging_by_default() -> None:
    """Without -l/-f, only stderr receives logs (no file)."""
    cli_app = create_app()

    @cli_app.command("log-check")
    def log_check() -> None:
        logger.debug("log-check debug")
        logger.info("log-check info")

    result = runner.invoke(cli_app, ["log-check"])
    assert result.exit_code == 0
    assert "log-check info" in _combined_output(result)


def test_main_help_includes_global_options() -> None:
    """Main help shows global file-logging options."""
    cli_app = create_app()
    result = runner.invoke(cli_app, ["--help"])
    assert result.exit_code == 0
    out = _plain_output(result)
    assert "file-logging" in out or "file_logging" in out
    assert "log-file" in out or "log_file" in out
    assert "download-nsrr" in out


def test_download_nsrr_help() -> None:
    """download-nsrr subcommand is registered and shows help with expected options."""
    cli_app = create_app()
    result = runner.invoke(cli_app, ["download-nsrr", "--help"])
    assert result.exit_code == 0
    out = _plain_output(result)
    assert "download-nsrr" in out
    assert "slug" in out
    assert "NSRR" in out
    assert "output_dir" in out or "OUTPUT_DIR" in out
    assert "--token" in out or "token" in out or "NSRR_TOKEN" in out
    assert "--path" in out or "path" in out
    assert "timeout" in out
    assert "download-retries" in out or "download_retries" in out
    assert "http-retries" in out or "http_retries" in out
