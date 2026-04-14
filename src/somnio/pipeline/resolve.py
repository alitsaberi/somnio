"""Pipeline-specific resolution helpers.

Generic import-string resolution lives in `somnio.utils.imports`.
"""

from __future__ import annotations

from somnio.pipeline.types import Transform
from somnio.utils.imports import resolve_import_string


def resolve_transform_target(target: str | Transform) -> Transform:
    """Resolve a transform target to a callable transform."""
    return resolve_import_string(target) if isinstance(target, str) else target
