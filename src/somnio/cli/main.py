"""Somnio CLI app.

This module is the console entry point target.
"""

from __future__ import annotations

from pathlib import Path

from somnio.utils.imports import MissingOptionalDependency

try:
    import typer
    from loguru import logger
except ModuleNotFoundError as e:
    if e.name not in ("typer", "loguru"):
        raise
    raise MissingOptionalDependency(e.name, extra="cli", purpose="Somnio CLI") from e

from .commands.nsrr import download as download_nsrr
from .utils.logging import (
    DEFAULT_LOG_DIRECTORY,
    DEFAULT_LOG_FILE_NAME,
    create_log_file_path,
    setup_logging,
)


def create_app() -> "typer.Typer":
    """Create a new CLI app instance.

    Tests should prefer this factory to avoid mutating global app state.
    """

    app = typer.Typer()
    app.command("download-nsrr")(download_nsrr)

    @app.callback()
    def _main_callback(
        ctx: typer.Context,
        file_logging: bool = typer.Option(
            False,
            "-l",
            "--file-logging",
            help=f"Enable file logging. By default, logs are written to {DEFAULT_LOG_DIRECTORY}/{DEFAULT_LOG_FILE_NAME}. Use before the subcommand.",
        ),
        log_file: Path | None = typer.Option(
            None,
            "-f",
            "--log-file",
            path_type=Path,
            help="Log to this file instead of the default. Implies file logging. Must appear before the subcommand.",
        ),
    ) -> None:
        if file_logging and log_file is None:
            command_name = ctx.invoked_subcommand or app.info.name
            log_file = create_log_file_path(command_name)

        setup_logging(log_file=log_file)
        logger.debug(
            "CLI: command={} subcommand={} params={} args={}",
            ctx.info_name,
            ctx.invoked_subcommand,
            ctx.params,
            ctx.args,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app()
