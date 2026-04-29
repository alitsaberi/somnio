from __future__ import annotations

from pathlib import Path
from typing import Any

from somnio.utils.imports import MissingOptionalDependency

try:
    import yaml
except ModuleNotFoundError as e:
    if e.name != "yaml":
        raise
    raise MissingOptionalDependency(
        "yaml", extra="schemas", purpose="YAML schema loading"
    ) from e


def load_yaml(path: Path | str) -> Any:
    """Load YAML from a file path."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in (".yaml", ".yml"):
        raise ValueError(
            f"Unsupported extension {path.suffix!r} for {path}; expected .yaml or .yml"
        )

    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text)
