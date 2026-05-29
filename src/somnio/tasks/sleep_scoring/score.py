"""Public scoring API for sleep-stage inference (backend-agnostic)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from somnio.data import Epochs, TimeSeries
from somnio.tasks.sleep_scoring.backend import SleepScoringBackend
from somnio.tasks.sleep_scoring.schema import ModelMetadata
from somnio.tasks.sleep_scoring.windowing import (
    PeriodTimestampAlignment,
    build_nptc_batches_from_metadata,
)


def _as_bptk(
    pred: np.ndarray,
    *,
    n_batch: int,
    n_periods_per_window: int,
    n_predictions_per_period: int,
    n_classes: int,
) -> np.ndarray:
    """Normalize backend output to shape (B, P, n_predictions, K)."""
    x = np.asarray(pred)
    b, p, t_pred, k = (
        n_batch,
        n_periods_per_window,
        n_predictions_per_period,
        n_classes,
    )
    expected = (b, p, t_pred, k)

    if x.ndim == 4:
        if x.shape == expected:
            return x
        raise ValueError(
            "Unexpected prediction shape; expected "
            f"(B,P,T_pred,K)={expected}, got {x.shape}"
        )

    if x.ndim == 3:
        if t_pred == 1 and x.shape == (b, p, k):
            return x[:, :, np.newaxis, :]
        if x.shape == (b * p, t_pred, k):
            return x.reshape(b, p, t_pred, k)
        raise ValueError(
            "Unexpected prediction shape; expected "
            f"(B,P,T_pred,K)={expected}, (B,P,K) with T_pred=1, or "
            f"(B*P,T_pred,K)=({b * p},{t_pred},{k}), got {x.shape}"
        )

    if x.ndim == 2:
        if t_pred == 1 and x.shape == (b * p, k):
            return x.reshape(b, p, 1, k)
        raise ValueError(
            "Unexpected prediction shape; expected "
            f"(B,P,T_pred,K)={expected} or (B*P,K)=({b * p},{k}) with T_pred=1, "
            f"got {x.shape}"
        )

    raise ValueError(f"Unexpected prediction rank {x.ndim}; shape={x.shape}")


def _prediction_offsets_in_period(
    n_samples_per_period: int,
    n_samples_per_prediction: int,
) -> np.ndarray:
    """First source-sample offset of each prediction within one input period."""
    if n_samples_per_period % n_samples_per_prediction != 0:
        raise ValueError(
            f"n_samples_per_period ({n_samples_per_period}) must be divisible by "
            f"n_samples_per_prediction ({n_samples_per_prediction})"
        )
    n_predictions = n_samples_per_period // n_samples_per_prediction
    return np.arange(n_predictions, dtype=np.int64) * n_samples_per_prediction


def _flatten_real_predictions(
    bptk: np.ndarray,
    *,
    batch_slot_is_real_period: np.ndarray,
    period_start_sample: np.ndarray,
    n_samples_per_period: int,
    n_samples_per_prediction: int,
    n_predictions_per_period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep every prediction for real periods.

    Returns:
        probs: ``(n_predictions, n_classes)``
        prediction_start_sample: ``(n_predictions,)`` index into the source series
    """
    slot_real = np.asarray(batch_slot_is_real_period, dtype=bool).reshape(-1)
    probs_slots = np.asarray(bptk, dtype=np.float64).reshape(
        -1, n_predictions_per_period, bptk.shape[-1]
    )
    period_probs = probs_slots[slot_real]
    n_periods = period_probs.shape[0]
    if n_periods != period_start_sample.shape[0]:
        raise RuntimeError(
            "Internal mismatch: real-period predictions vs period starts "
            f"({n_periods} != {period_start_sample.shape[0]})"
        )

    offsets = _prediction_offsets_in_period(
        n_samples_per_period, n_samples_per_prediction
    )
    starts = np.asarray(period_start_sample, dtype=np.int64)
    prediction_start_sample = starts[:, np.newaxis] + offsets[np.newaxis, :]
    probs = period_probs.reshape(n_periods * n_predictions_per_period, -1)
    return probs, prediction_start_sample.reshape(-1)


def score_sleep_stages(
    ts: TimeSeries,
    *,
    backend: SleepScoringBackend,
    metadata: ModelMetadata,
    timestamp_alignment: PeriodTimestampAlignment = PeriodTimestampAlignment.PERIOD_START,
    output: Literal[
        "probs_timeseries", "indices_epochs", "labels_epochs"
    ] = "probs_timeseries",
    period_stride_samples: int | None = None,
) -> TimeSeries | Epochs:
    """Score sleep stages from an input signal `TimeSeries`.

    This function is backend-agnostic: any `SleepScoringBackend` can be used as long as
    its `predict()` output can be normalized to
    ``(n_batch, n_periods_per_window, n_predictions_per_period, n_classes)`` where
    ``n_predictions_per_period = n_samples_per_period // n_samples_per_prediction``.

    Args:
        ts: Input time-series, shape ``(n_samples, n_channels)``.
        backend: Inference backend (e.g. ONNX).
        metadata: Model metadata describing windowing and class labels.
        timestamp_alignment: How to anchor each output period timestamp for the `TimeSeries` output.
        output: Which output to return:
            - ``"probs_timeseries"``: class-probability `TimeSeries` at model prediction
              resolution (one row per aggregated block of ``n_samples_per_prediction``
              input samples), shape ``(n_predictions, n_classes)``
            - ``"indices_epochs"``: `Epochs` of per-epoch argmax class indices
            - ``"labels_epochs"``: `Epochs` of per-epoch argmax class labels (strings)
        period_stride_samples: Step (in samples) between consecutive periods. Defaults
            to non-overlapping periods (equal to `n_samples_per_period`).

    Returns:
        Either a probability `TimeSeries` or an `Epochs` object, depending on `output`.
    """
    w = build_nptc_batches_from_metadata(
        ts,
        metadata,
        period_stride_samples=period_stride_samples,
        timestamp_alignment=timestamp_alignment,
    )

    pred = backend.predict(w.batches)
    bptk = _as_bptk(
        pred,
        n_batch=w.batches.shape[0],
        n_periods_per_window=metadata.n_periods_per_window,
        n_predictions_per_period=metadata.n_predictions_per_period,
        n_classes=len(metadata.class_labels),
    )

    n_sp = int(metadata.n_samples_per_period)
    stride = int(metadata.n_samples_per_prediction)
    n_pred = metadata.n_predictions_per_period

    probs, prediction_start_sample = _flatten_real_predictions(
        bptk,
        batch_slot_is_real_period=w.batch_slot_is_real_period,
        period_start_sample=w.period_start_sample,
        n_samples_per_period=n_sp,
        n_samples_per_prediction=stride,
        n_predictions_per_period=n_pred,
    )

    if output == "probs_timeseries":
        idx = np.clip(prediction_start_sample, 0, ts.n_samples - 1)
        pred_timestamps = ts.timestamps[idx]
        pred_sample_rate_hz = metadata.sample_rate_hz / stride
        return TimeSeries(
            values=np.asarray(probs, dtype=np.float64),
            timestamps=pred_timestamps,
            channel_names=list(metadata.class_labels),
            units=["1"] * len(metadata.class_labels),
            sample_rate=pred_sample_rate_hz,
        )

    onset = int(ts.timestamps[0])
    prediction_length_ns = int(round(1e9 * stride / metadata.sample_rate_hz))
    epoch_indices = np.argmax(probs, axis=1).astype(np.int64)

    if output == "indices_epochs":
        return Epochs(
            labels=epoch_indices, period_length=prediction_length_ns, onset=onset
        )

    epoch_labels = np.asarray(
        [metadata.class_labels[int(i)] for i in epoch_indices], dtype=object
    )
    return Epochs(labels=epoch_labels, period_length=prediction_length_ns, onset=onset)
