"""Import helpers used across `somnio`."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


class MissingOptionalDependency(ModuleNotFoundError):
    """Raised when an optional somnio extra is needed but not installed.

    Attributes:
        module: Python import name of the missing package.
        extra: somnio extra that provides it.
    """

    def __init__(self, module: str, *, extra: str, purpose: str) -> None:
        super().__init__(
            f"{purpose} requires '{module}'. "
            f"Install the '{extra}' extra: `pip install 'somnio[{extra}]'`."
        )
        self.module = module
        self.extra = extra


def resolve_import_string(target: str) -> Callable[..., Any]:
    """Resolve ``\"pkg.module:callable\"`` to a callable."""
    if ":" not in target:
        raise ValueError(
            f"Invalid import string {target!r}; expected format 'pkg.module:callable'"
        )
    module_name, attr = target.split(":", 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, attr, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Import string {target!r} did not resolve to a callable")
    return fn
