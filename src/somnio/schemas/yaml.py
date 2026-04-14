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
    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text)
