"""ONNX Runtime wrapper for sleep stage models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from somnio.tasks.sleep_scoring.schema import ModelMetadata
from somnio.utils.imports import MissingOptionalDependency

import logging

logger = logging.getLogger(__name__)

try:
    from onnxruntime import (
        InferenceSession,
        SessionOptions,
        set_default_logger_severity,
    )

    set_default_logger_severity(3)
except ModuleNotFoundError as e:
    if e.name != "onnxruntime":
        raise
    raise MissingOptionalDependency(
        "onnxruntime",
        extra="onnx",
        purpose="ONNX sleep-scoring inference",
    ) from e


def load_model_metadata(path: Path | str) -> ModelMetadata:
    """Load and validate sidecar metadata from YAML (``.yaml`` / ``.yml``)."""

    from somnio.schemas.yaml import load_yaml

    raw = load_yaml(path)
    return ModelMetadata.model_validate(raw)


def _discover_metadata_path(model_path: Path) -> Path:
    parent = model_path.parent
    stem = model_path.stem
    for name in (
        f"{stem}.yaml",
        f"{stem}.yml",
    ):
        candidate = parent / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No sidecar metadata found next to {model_path}. "
        "Pass an explicit metadata path or add "
        f"{stem}.yaml / metadata.yaml (or .yml)."
    )


def _resolve_io_names(
    session: InferenceSession, metadata: ModelMetadata
) -> tuple[str, str]:
    onnx_conf = metadata.onnx
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs or not outputs:
        raise ValueError("ONNX model must have at least one input and one output")

    in_name = onnx_conf.input_name
    if in_name is None:
        in_name = inputs[0].name
    else:
        names = {i.name for i in inputs}
        if in_name not in names:
            raise ValueError(
                f"metadata.onnx.input_name {in_name!r} not in model inputs {sorted(names)}"
            )

    out_name = onnx_conf.output_name
    if out_name is None:
        out_name = outputs[0].name
    else:
        names = {o.name for o in outputs}
        if out_name not in names:
            raise ValueError(
                f"metadata.onnx.output_name {out_name!r} not in model outputs {sorted(names)}"
            )

    return in_name, out_name


def _onnx_dim_as_int(dim: Any) -> int | None:
    if dim is None:
        return None
    if isinstance(dim, int):
        return dim if dim >= 0 else None
    return None


def _verify_nptc_input(
    session: InferenceSession, metadata: ModelMetadata, input_name: str
) -> None:
    inp = next(i for i in session.get_inputs() if i.name == input_name)
    shape = inp.shape
    if len(shape) != 4:
        raise ValueError(
            f"Expected 4D NPTC input for {input_name!r}, got rank {len(shape)} shape={shape!r}"
        )
    for axis, meta_dim, label in zip(
        shape[1:],
        (
            metadata.n_periods_per_window,
            metadata.n_samples_per_period,
            metadata.n_channels,
        ),
        ("P", "T", "C"),
        strict=True,
    ):
        onnx_i = _onnx_dim_as_int(axis)
        if onnx_i is not None and onnx_i != meta_dim:
            raise ValueError(
                f"Input axis {label} onnx dim {onnx_i} != metadata {meta_dim}"
            )


def _verify_output_logits_dim(
    session: InferenceSession, metadata: ModelMetadata, output_name: str
) -> None:
    """Ensure the output logits dimension matches the number of class labels."""
    out = next(o for o in session.get_outputs() if o.name == output_name)
    shape = out.shape

    k = _onnx_dim_as_int(shape[-1])
    n = len(metadata.class_labels)
    if k != n:
        raise ValueError(
            f"Output last dim {k} for {output_name!r} != len(class_labels) {n}"
        )


class OnnxSleepScoringModel:
    def __init__(
        self,
        session: InferenceSession,
        metadata: ModelMetadata,
        *,
        input_name: str,
        output_name: str,
    ) -> None:
        self._session = session
        self._metadata = metadata
        self._input_name = input_name
        self._output_name = output_name

    @classmethod
    def load(
        cls,
        model_path: Path | str,
        metadata: ModelMetadata | Path | str | None = None,
        *,
        providers: list[str] | None = None,
        session_options: SessionOptions | None = None,
    ) -> OnnxSleepScoringModel:
        """Load ONNX weights and sidecar metadata from disk."""
        model_path = Path(model_path)
        if isinstance(metadata, ModelMetadata):
            meta = metadata
        elif isinstance(metadata, (str, Path)):
            meta = load_model_metadata(metadata)
        elif metadata is None:
            meta = load_model_metadata(_discover_metadata_path(model_path))
        else:
            raise TypeError(
                f"metadata must be ModelMetadata, path, or None, got {type(metadata)}"
            )

        sess = InferenceSession(
            str(model_path.resolve()),
            sess_options=session_options,
            providers=providers,
        )
        in_name, out_name = _resolve_io_names(sess, meta)
        _verify_nptc_input(sess, meta, in_name)
        _verify_output_logits_dim(sess, meta, out_name)
        return cls(sess, meta, input_name=in_name, output_name=out_name)

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def class_labels(self) -> tuple[str, ...]:
        return tuple(self._metadata.class_labels)

    @property
    def n_classes(self) -> int:
        return len(self._metadata.class_labels)

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def output_name(self) -> str:
        return self._output_name

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Forward pass: ``batch`` float32 ``(batch, n_periods_per_window, n_samples_per_period, n_channels)`` → probs ``(batch, n_windows, 1, n_classes)``."""
        x = np.asarray(batch, dtype=np.float32, order="C")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D NPTC batch, got shape {x.shape}")

        p, t, c = x.shape[1], x.shape[2], x.shape[3]
        m = self._metadata
        if (p, t, c) != (m.n_periods_per_window, m.n_samples_per_period, m.n_channels):
            raise ValueError(
                f"Batch shape (P,T,C)=({p},{t},{c}) != metadata "
                f"({m.n_periods_per_window}, {m.n_samples_per_period}, {m.n_channels})"
            )

        outputs = self._session.run([self._output_name], {self._input_name: x})
        logger.debug(f"ONNX output shape: {outputs[0].shape}")

        return np.asarray(outputs[0], dtype=np.float32)
