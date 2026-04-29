"""Sliding-window batching for sleep-scoring (NPTC layout).

Periods are cut with ``period_stride_samples`` (default: non-overlapping epochs when
equal to ``n_samples_per_period``). **Trailing samples that do not fill a full
period are not discarded**: the last period is **zero- or edge-padded** to
``n_samples_per_period`` so the tail of the recording is still scored.

Batch slots that exist only to round up ``n_periods`` to a multiple of
``n_periods_per_window`` are filled by repeating the last **real** period tensor
and flagged in ``batch_slot_is_real_period``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np

from somnio.data import TimeSeries
from somnio.tasks.sleep_scoring.schema import ModelMetadata


class PeriodTimestampAlignment(str, Enum):
    """Reference instant for each period when mapping to output time."""

    PERIOD_START = "period_start"
    """Timestamp of the first sample in the period (observed, not padded)."""

    PERIOD_CENTER = "period_center"
    """Mean timestamp over observed samples in the period (padding excluded)."""


@dataclass(frozen=True, slots=True)
class WindowingResult:
    """Batches and per-period timing for NPTC inference."""

    batches: np.ndarray
    """Shape ``(n_batch, n_periods_per_window, n_samples_per_period, n_channels)``, float32."""

    period_start_sample: np.ndarray
    """Index into the source series where each period starts; shape ``(n_periods,)``."""

    period_timestamp_ns: np.ndarray
    """Anchor time per period (see ``PeriodTimestampAlignment``); shape ``(n_periods,)``, int64."""

    period_is_fully_observed: np.ndarray
    """``True`` if that period used only real samples (no tail padding); shape ``(n_periods,)``."""

    batch_slot_is_real_period: np.ndarray
    """``True`` if batch slot maps to a real period; ``False`` for batch tail padding; shape ``(n_batch, n_periods_per_window)``."""

    sample_rate_hz: float
    """Nominal sample rate used for validation / diagnostics."""


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _period_anchor_ns(
    timestamps: np.ndarray,
    start: int,
    end_excl: int,
    alignment: PeriodTimestampAlignment,
) -> int:
    real_end = min(end_excl, len(timestamps))
    if real_end <= start:
        return int(timestamps[-1])
    seg = timestamps[start:real_end]
    if alignment == PeriodTimestampAlignment.PERIOD_START:
        return int(seg[0])
    return int(np.rint(np.mean(seg.astype(np.float64))))


def _pad_period_tail(
    chunk: np.ndarray,
    n_samples_per_period: int,
    mode: Literal["edge", "constant"],
    constant_value: float,
) -> tuple[np.ndarray, bool]:
    """Return ``(T, C)`` and whether tail padding was applied."""
    t, c = chunk.shape
    _require_positive("n_samples_per_period", n_samples_per_period)
    if t == n_samples_per_period:
        return chunk, False
    if t > n_samples_per_period:
        raise RuntimeError(
            f"internal: chunk length {t} > n_samples_per_period {n_samples_per_period}"
        )
    pad_rows = n_samples_per_period - t
    pad_width = ((0, pad_rows), (0, 0))
    if mode == "edge":
        out = np.pad(chunk, pad_width, mode="edge")
    else:
        out = np.pad(
            chunk,
            pad_width,
            mode="constant",
            constant_values=constant_value,
        )
    return out, True


def build_nptc_batches(
    time_series: TimeSeries,
    *,
    n_periods_per_window: int,
    n_samples_per_period: int,
    n_channels: int,
    sample_rate_hz: float,
    period_stride_samples: int | None = None,
    partial_period_padding: Literal["edge", "constant"] = "edge",
    partial_pad_value: float = 0.0,
    batch_tail_padding: Literal["edge", "constant"] = "edge",
    batch_tail_pad_value: float = 0.0,
    timestamp_alignment: PeriodTimestampAlignment = PeriodTimestampAlignment.PERIOD_START,
    sample_rate_tolerance_hz: float = 1e-3,
) -> WindowingResult:
    """Slice ``TimeSeries.values`` into NPTC batches.

    Partial final periods are **padded**, not dropped. Incomplete **batches**
    are padded by repeating the last real period (``edge``) or zeros (``constant``).
    """
    _require_positive("n_periods_per_window", n_periods_per_window)
    _require_positive("n_samples_per_period", n_samples_per_period)
    _require_positive("n_channels", n_channels)
    _require_positive("sample_rate_hz", sample_rate_hz)

    stride = (
        period_stride_samples
        if period_stride_samples is not None
        else n_samples_per_period
    )
    _require_positive("period_stride_samples", stride)

    if (
        time_series.sample_rate is not None
        and abs(time_series.sample_rate - sample_rate_hz) > sample_rate_tolerance_hz
    ):
        raise ValueError(
            f"time_series.sample_rate={time_series.sample_rate} does not match "
            f"expected sample_rate_hz={sample_rate_hz} (tol={sample_rate_tolerance_hz})"
        )

    values = time_series.values
    timestamps = time_series.timestamps
    n_samples = time_series.n_samples
    if time_series.n_channels != n_channels:
        raise ValueError(
            f"time_series has n_channels={time_series.n_channels}, expected metadata n_channels={n_channels}"
        )
    if time_series.n_samples == 0:
        raise ValueError("time_series has no samples")

    starts = np.arange(0, n_samples, stride, dtype=np.int64)
    n_periods = int(len(starts))
    periods = np.empty((n_periods, n_samples_per_period, n_channels), dtype=np.float64)
    fully_obs = np.empty(n_periods, dtype=bool)
    ts_out = np.empty(n_periods, dtype=np.int64)

    for i, s in enumerate(starts):
        e = min(int(s) + n_samples_per_period, n_samples)
        chunk = values[s:e, :].copy()
        padded, had_tail_pad = _pad_period_tail(
            chunk,
            n_samples_per_period,
            partial_period_padding,
            partial_pad_value,
        )
        periods[i] = padded
        fully_obs[i] = not had_tail_pad
        ts_out[i] = _period_anchor_ns(timestamps, int(s), e, timestamp_alignment)

    p = n_periods_per_window
    n_batch = (n_periods + p - 1) // p
    total_slots = n_batch * p
    batched = np.zeros(
        (total_slots, n_samples_per_period, n_channels), dtype=np.float32
    )
    batched[:n_periods] = periods.astype(np.float32, copy=False)

    slot_real = np.zeros((n_batch, p), dtype=bool)
    slot_real.flat[:n_periods] = True

    if n_periods < total_slots:
        tail = (
            batched[n_periods - 1].copy()
            if batch_tail_padding == "edge"
            else np.full(
                (n_samples_per_period, n_channels),
                batch_tail_pad_value,
                dtype=np.float32,
            )
        )
        batched[n_periods:] = tail

    out_batches = batched.reshape(n_batch, p, n_samples_per_period, n_channels)
    return WindowingResult(
        batches=out_batches,
        period_start_sample=starts,
        period_timestamp_ns=ts_out,
        period_is_fully_observed=fully_obs,
        batch_slot_is_real_period=slot_real,
        sample_rate_hz=float(sample_rate_hz),
    )


def build_nptc_batches_from_metadata(
    time_series: TimeSeries,
    metadata: ModelMetadata,
    *,
    period_stride_samples: int | None = None,
    **kwargs: Any,
) -> WindowingResult:
    """Same as :func:`build_nptc_batches` with shape fields taken from ``metadata``."""
    return build_nptc_batches(
        time_series,
        n_periods_per_window=metadata.n_periods_per_window,
        n_samples_per_period=metadata.n_samples_per_period,
        n_channels=metadata.n_channels,
        sample_rate_hz=metadata.sample_rate_hz,
        period_stride_samples=period_stride_samples,
        **kwargs,
    )
