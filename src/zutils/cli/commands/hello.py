"""Example `hello` command (placeholder)."""

from __future__ import annotations

import typer
from loguru import logger

app = typer.Typer(no_args_is_help=True)


@app.command()
def hello(name: str = "world") -> None:
    """Print a hello message."""

    logger.info("Hello, {}!", name)
