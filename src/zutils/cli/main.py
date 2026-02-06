"""ZUtils CLI app.

This module is the console entry point target.
"""

from __future__ import annotations

import sys

try:
    import typer
    from loguru import logger  # noqa: F401
except ModuleNotFoundError as e:
    if e.name in {"typer", "loguru"}:
        print(
            "The ZUtils CLI requires optional dependencies.\n\n"
            "Install with one of:\n"
            "  - pip install 'zutils[cli]'\n"
            "  - uv add 'zutils[cli]'\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
    raise


from .commands import app as commands_app
from .utils.logging import DEFAULT_LOG_LEVEL, setup_logging


def create_app() -> "typer.Typer":
    """Create a new CLI app instance.

    Tests should prefer this factory to avoid mutating global app state.
    """

    app = typer.Typer(no_args_is_help=True)
    app.add_typer(commands_app)

    @app.callback()
    def _main(
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Enable debug logging.",
        ),
    ) -> None:
        setup_logging(level="DEBUG" if verbose else DEFAULT_LOG_LEVEL)

    return app


app = create_app()


if __name__ == "__main__":
    app()
