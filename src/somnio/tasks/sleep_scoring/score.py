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


def _as_bpk(
    pred: np.ndarray,
    *,
    n_batch: int,
    n_periods_per_window: int,
    n_classes: int,
) -> np.ndarray:
    """Normalize backend output to shape (B, P, K)."""
    x = np.asarray(pred)

    # Common cases:
    # - (B, P, K)
    # - (B, P, 1, K)  (U-Time-like)
    # - (B*P, K)
    if x.ndim == 4 and x.shape[2] == 1:
        x = x[:, :, 0, :]
    if x.ndim == 3:
        if x.shape != (n_batch, n_periods_per_window, n_classes):
            raise ValueError(
                "Unexpected prediction shape; expected "
                f"(B,P,K)=({n_batch},{n_periods_per_window},{n_classes}), got {x.shape}"
            )
        return x
    if x.ndim == 2:
        if x.shape != (n_batch * n_periods_per_window, n_classes):
            raise ValueError(
                "Unexpected prediction shape; expected "
                f"(B*P,K)=({n_batch * n_periods_per_window},{n_classes}), got {x.shape}"
            )
        return x.reshape(n_batch, n_periods_per_window, n_classes)

    raise ValueError(f"Unexpected prediction rank {x.ndim}; shape={x.shape}")


def _aggregate_period_probs_to_epochs(
    probs: np.ndarray,
    *,
    period_start_sample: np.ndarray,
    n_samples_per_period: int,
) -> np.ndarray:
    """Aggregate per-period probabilities into fixed non-overlapping epochs.

    Epoch index is defined by the period start sample:
    ``epoch_id = period_start_sample // n_samples_per_period``.

    When periods overlap (stride < n_samples_per_period), multiple periods map to
    the same epoch; we aggregate by mean probability per class.
    """
    if n_samples_per_period <= 0:
        raise ValueError(
            f"n_samples_per_period must be positive, got {n_samples_per_period}"
        )
    x = np.asarray(probs, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected probs shape (n_periods, n_classes), got {x.shape}")

    starts = np.asarray(period_start_sample, dtype=np.int64)
    if starts.ndim != 1 or starts.shape[0] != x.shape[0]:
        raise ValueError(
            "period_start_sample must have shape (n_periods,), got "
            f"{starts.shape} for probs {x.shape}"
        )

    epoch_ids = (starts // int(n_samples_per_period)).astype(np.int64)
    n_epochs = int(epoch_ids.max()) + 1 if len(epoch_ids) else 0
    if n_epochs == 0:
        return np.empty((0, x.shape[1]), dtype=np.float64)

    sums = np.zeros((n_epochs, x.shape[1]), dtype=np.float64)
    counts = np.zeros((n_epochs,), dtype=np.float64)
    np.add.at(sums, epoch_ids, x)
    np.add.at(counts, epoch_ids, 1.0)
    return sums / counts.reshape(-1, 1)


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
    its `predict()` output can be normalized into per-period class probabilities.

    Args:
        ts: Input time-series, shape ``(n_samples, n_channels)``.
        backend: Inference backend (e.g. ONNX).
        metadata: Model metadata describing windowing and class labels.
        timestamp_alignment: How to anchor each output period timestamp for the `TimeSeries` output.
        output: Which output to return:
            - ``"probs_timeseries"``: class-probability `TimeSeries` aggregated to fixed
              epochs when periods overlap (mean probs per epoch), shape
              ``(n_epochs, n_classes)``
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
    bpk = _as_bpk(
        pred,
        n_batch=w.batches.shape[0],
        n_periods_per_window=metadata.n_periods_per_window,
        n_classes=len(metadata.class_labels),
    )

    # Flatten B,P -> slots; keep only real periods.
    slot_real = w.batch_slot_is_real_period.reshape(-1)
    probs_slots = bpk.reshape(-1, bpk.shape[-1])
    probs = probs_slots[slot_real]

    # Windowing guarantees one timestamp per real period.
    if probs.shape[0] != w.period_timestamp_ns.shape[0]:
        raise RuntimeError(
            "Internal mismatch: number of real-period predictions does not match "
            f"period timestamps ({probs.shape[0]} != {w.period_timestamp_ns.shape[0]})"
        )

    epoch_probs = _aggregate_period_probs_to_epochs(
        probs,
        period_start_sample=w.period_start_sample,
        n_samples_per_period=int(metadata.n_samples_per_period),
    )
    probs_sample_rate_hz = metadata.sample_rate_hz / metadata.n_samples_per_period
    period_length_ns = int(round(1e9 / probs_sample_rate_hz))

    if output == "probs_timeseries":
        epoch_timestamps = (
            w.period_timestamp_ns[0]
            + np.arange(epoch_probs.shape[0], dtype=np.int64) * period_length_ns
        )
        return TimeSeries(
            values=np.asarray(epoch_probs, dtype=np.float64),
            timestamps=epoch_timestamps,
            channel_names=list(metadata.class_labels),
            units=["1"] * len(metadata.class_labels),
            sample_rate=probs_sample_rate_hz,
        )

    onset = int(ts.timestamps[0])
    epoch_indices = np.argmax(epoch_probs, axis=1).astype(np.int64)

    if output == "indices_epochs":
        return Epochs(labels=epoch_indices, period_length=period_length_ns, onset=onset)

    epoch_labels = np.asarray(
        [metadata.class_labels[int(i)] for i in epoch_indices], dtype=object
    )
    return Epochs(labels=epoch_labels, period_length=period_length_ns, onset=onset)
