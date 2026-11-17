"""Load and discover sleep-scoring model.yaml sidecars."""

from __future__ import annotations

from pathlib import Path

from somnio.tasks.sleep_scoring.schema import ModelMetadata

DEFAULT_META_NAMES = ("model.yaml", "model.yml")


def load_model_metadata(path: Path | str) -> ModelMetadata:
    """Load and validate sidecar metadata from YAML (``.yaml`` / ``.yml``)."""

    from somnio.schemas.yaml import load_yaml

    raw = load_yaml(path)
    return ModelMetadata.model_validate(raw)


def discover_metadata_path(model_path: Path | str) -> Path:
    """Locate ``model.yaml`` next to a file artifact or inside a directory.

    For a file (e.g. ``model.onnx``), search ``{stem}.yaml`` / ``{stem}.yml``,
    then ``model.yaml`` / ``model.yml`` in the same directory.

    For a directory (e.g. a TensorFlow SavedModel), search ``model.yaml`` /
    ``model.yml`` inside that directory.
    """
    model_path = Path(model_path)
    if model_path.is_dir():
        for name in DEFAULT_META_NAMES:
            candidate = model_path / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No sidecar metadata found inside {model_path}. "
            f"Expected one of {list(DEFAULT_META_NAMES)}."
        )

    parent = model_path.parent
    stem = model_path.stem
    candidates = (
        f"{stem}.yaml",
        f"{stem}.yml",
        *DEFAULT_META_NAMES,
    )
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        candidate = parent / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No sidecar metadata found next to {model_path}. "
        "Pass an explicit metadata path or add "
        f"{stem}.yaml / model.yaml (or .yml)."
    )
