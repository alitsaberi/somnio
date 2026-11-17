"""TensorFlow SavedModel wrapper for sleep stage models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from somnio.tasks.sleep_scoring.models.metadata import (
    discover_metadata_path,
    load_model_metadata,
)
from somnio.tasks.sleep_scoring.schema import ModelMetadata
from somnio.utils.imports import MissingOptionalDependency

import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
except ModuleNotFoundError as e:
    if e.name != "tensorflow":
        raise
    raise MissingOptionalDependency(
        "tensorflow",
        extra="tensorflow",
        purpose="TensorFlow sleep-scoring inference",
    ) from e


def _dim_as_int(dim: Any) -> int | None:
    if dim is None:
        return None
    try:
        if hasattr(dim, "value"):
            dim = dim.value
        if dim is None:
            return None
        i = int(dim)
    except (TypeError, ValueError):
        return None
    return i if i >= 0 else None


def _resolve_serving_fn(
    loaded: Any, metadata: ModelMetadata
) -> tuple[Any, str, str]:
    """Return ``(concrete_fn, input_name, output_name)`` from SavedModel signatures."""
    signatures = getattr(loaded, "signatures", None) or {}
    if not signatures:
        raise ValueError("TensorFlow SavedModel has no signatures")

    conf = metadata.tensorflow
    # Prefer serving_default; otherwise first available signature.
    if "serving_default" in signatures:
        fn = signatures["serving_default"]
    else:
        fn = next(iter(signatures.values()))

    structured_in = fn.structured_input_signature
    if not structured_in or len(structured_in) < 2 or not structured_in[1]:
        raise ValueError("SavedModel signature has no keyword inputs")
    input_map: dict[str, Any] = dict(structured_in[1])
    output_map: dict[str, Any] = dict(fn.structured_outputs or {})
    if not output_map:
        raise ValueError("SavedModel signature has no outputs")

    if conf.input_name is None:
        in_name = next(iter(input_map))
    else:
        if conf.input_name not in input_map:
            raise ValueError(
                f"metadata.tensorflow.input_name {conf.input_name!r} not in "
                f"signature inputs {sorted(input_map)}"
            )
        in_name = conf.input_name

    if conf.output_name is None:
        out_name = next(iter(output_map))
    else:
        if conf.output_name not in output_map:
            raise ValueError(
                f"metadata.tensorflow.output_name {conf.output_name!r} not in "
                f"signature outputs {sorted(output_map)}"
            )
        out_name = conf.output_name

    return fn, in_name, out_name


def _verify_nptc_input(spec: Any, metadata: ModelMetadata, input_name: str) -> None:
    shape = tuple(getattr(spec, "shape", ()))
    if len(shape) != 4:
        raise ValueError(
            f"Expected 4D NPTC input for {input_name!r}, "
            f"got rank {len(shape)} shape={shape!r}"
        )

    p_meta = metadata.n_periods_per_window
    p_tf = _dim_as_int(shape[1])
    if p_meta is None:
        if p_tf is not None:
            raise ValueError(
                f"Metadata n_periods_per_window is null (dynamic), but TF input "
                f"axis P is fixed to {p_tf}"
            )
    elif p_tf is not None and p_tf != p_meta:
        raise ValueError(f"Input axis P tf dim {p_tf} != metadata {p_meta}")

    for axis, meta_dim, label in zip(
        shape[2:],
        (metadata.n_samples_per_period, metadata.n_channels),
        ("T", "C"),
        strict=True,
    ):
        tf_i = _dim_as_int(axis)
        if tf_i is not None and tf_i != meta_dim:
            raise ValueError(
                f"Input axis {label} tf dim {tf_i} != metadata {meta_dim}"
            )


def _verify_output_logits_dim(
    spec: Any, metadata: ModelMetadata, output_name: str
) -> None:
    shape = tuple(getattr(spec, "shape", ()))
    k = _dim_as_int(shape[-1]) if shape else None
    n = len(metadata.class_labels)
    if k is not None and k != n:
        raise ValueError(
            f"Output last dim {k} for {output_name!r} != len(class_labels) {n}"
        )

    p_meta = metadata.n_periods_per_window

    if len(shape) == 4:
        t_pred = _dim_as_int(shape[-2])
        expected_t = metadata.n_predictions_per_period
        if t_pred is not None and t_pred != expected_t:
            raise ValueError(
                f"Output prediction-count dim {t_pred} for {output_name!r} != "
                f"n_samples_per_period // n_samples_per_prediction ({expected_t})"
            )
        p = _dim_as_int(shape[-3])
        if p_meta is None:
            if p is not None:
                raise ValueError(
                    f"Metadata n_periods_per_window is null (dynamic), but TF "
                    f"output periods dim is fixed to {p}"
                )
        elif p is not None and p != p_meta:
            raise ValueError(
                f"Output periods dim {p} for {output_name!r} != "
                f"metadata n_periods_per_window {p_meta}"
            )
        return

    if len(shape) == 3:
        p = _dim_as_int(shape[-2])
        if p_meta is None:
            if p is not None:
                raise ValueError(
                    f"Metadata n_periods_per_window is null (dynamic), but TF "
                    f"output periods dim is fixed to {p}"
                )
        elif p is not None and p != p_meta:
            raise ValueError(
                f"Output periods dim {p} for {output_name!r} != "
                f"metadata n_periods_per_window {p_meta}"
            )
        expected_t = metadata.n_predictions_per_period
        if expected_t != 1:
            raise ValueError(
                f"Output rank 3 (B,P,K) for {output_name!r} implies one prediction "
                f"per period, but metadata n_predictions_per_period is {expected_t}"
            )
        return

    raise ValueError(
        f"Expected 3D (B,P,K) or 4D (B,P,T_pred,K) output for {output_name!r}, "
        f"got rank {len(shape)} shape={shape!r}"
    )


class TensorflowSleepScoringModel:
    def __init__(
        self,
        serving_fn: Any,
        metadata: ModelMetadata,
        *,
        input_name: str,
        output_name: str,
    ) -> None:
        self._serving_fn = serving_fn
        self._metadata = metadata
        self._input_name = input_name
        self._output_name = output_name

    @classmethod
    def load(
        cls,
        model_path: Path | str,
        metadata: ModelMetadata | Path | str | None = None,
    ) -> TensorflowSleepScoringModel:
        """Load a TensorFlow SavedModel directory and sidecar metadata.

        Uses ``tf.saved_model.load`` (Keras 3 does not reload legacy SavedModel
        directories via ``keras.models.load_model``).
        """
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise NotADirectoryError(
                f"TensorFlow SavedModel path must be a directory, got {model_path}"
            )

        if isinstance(metadata, ModelMetadata):
            meta = metadata
        elif isinstance(metadata, (str, Path)):
            meta = load_model_metadata(metadata)
        elif metadata is None:
            meta = load_model_metadata(discover_metadata_path(model_path))
        else:
            raise TypeError(
                f"metadata must be ModelMetadata, path, or None, got {type(metadata)}"
            )

        loaded = tf.saved_model.load(str(model_path.resolve()))
        fn, in_name, out_name = _resolve_serving_fn(loaded, meta)
        in_spec = fn.structured_input_signature[1][in_name]
        out_spec = fn.structured_outputs[out_name]
        _verify_nptc_input(in_spec, meta, in_name)
        _verify_output_logits_dim(out_spec, meta, out_name)
        return cls(fn, meta, input_name=in_name, output_name=out_name)

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
        """Forward pass: NPTC batch in, class logits/probabilities out.

        Input shape: ``(n_batch, n_periods_per_window, n_samples_per_period, n_channels)``.
        """
        x = np.asarray(batch, dtype=np.float32, order="C")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D NPTC batch, got shape {x.shape}")

        p, t, c = x.shape[1], x.shape[2], x.shape[3]
        m = self._metadata
        if m.n_periods_per_window is None:
            if (t, c) != (m.n_samples_per_period, m.n_channels):
                raise ValueError(
                    f"Batch shape (T,C)=({t},{c}) != metadata "
                    f"({m.n_samples_per_period}, {m.n_channels})"
                )
        elif (p, t, c) != (
            m.n_periods_per_window,
            m.n_samples_per_period,
            m.n_channels,
        ):
            raise ValueError(
                f"Batch shape (P,T,C)=({p},{t},{c}) != metadata "
                f"({m.n_periods_per_window}, {m.n_samples_per_period}, {m.n_channels})"
            )

        outputs = self._serving_fn(
            **{self._input_name: tf.constant(x, dtype=tf.float32)}
        )
        out = outputs[self._output_name]
        arr = np.asarray(out, dtype=np.float32)
        logger.debug(f"TensorFlow output shape: {arr.shape}")
        return arr
