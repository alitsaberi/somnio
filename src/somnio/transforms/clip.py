"""Clipping and basic signal sanitation transforms for TimeSeries.

This module provides:
- Non-finite handling (NaN/Inf)
- Hard clipping to fixed bounds
- Robust IQR-based clipping
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from somnio.data.timeseries import TimeSeries


NonFiniteStrategy = Literal["error", "clip", "replace"]


def apply_clip(
    ts: TimeSeries,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> TimeSeries:
    """Clamp values to [lower, upper] (applied elementwise across all channels)."""
    if lower is None and upper is None:
        return ts
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"lower must be <= upper, got lower={lower}, upper={upper}")

    out = np.asarray(ts.values, dtype=np.float64)
    if lower is not None:
        out = np.maximum(out, float(lower))
    if upper is not None:
        out = np.minimum(out, float(upper))
    out = np.asarray(out, dtype=np.float64, order="C")

    return TimeSeries(
        values=out,
        timestamps=ts.timestamps.copy(),
        channel_names=list(ts.channel_names),
        units=list(ts.units),
        sample_rate=ts.sample_rate,
    )


def apply_clip_iqr(
    ts: TimeSeries,
    *,
    iqr_factor: float = 20.0,
    q_low: float = 0.25,
    q_high: float = 0.75,
) -> TimeSeries:
    """Robust outlier clipping using per-channel IQR bounds."""
    if iqr_factor < 0:
        raise ValueError(f"iqr_factor must be >= 0, got {iqr_factor}")
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(f"Require 0 <= q_low < q_high <= 1, got {q_low}, {q_high}")

    x = np.asarray(ts.values, dtype=np.float64)
    ql = np.quantile(x, q_low, axis=0)
    qh = np.quantile(x, q_high, axis=0)
    iqr = qh - ql
    lower = ql - (iqr_factor * iqr)
    upper = qh + (iqr_factor * iqr)

    out = np.clip(x, lower.reshape(1, -1), upper.reshape(1, -1))
    return TimeSeries(
        values=np.asarray(out, dtype=np.float64),
        timestamps=ts.timestamps.copy(),
        channel_names=list(ts.channel_names),
        units=list(ts.units),
        sample_rate=ts.sample_rate,
    )


def apply_non_finite(
    ts: TimeSeries,
    *,
    strategy: NonFiniteStrategy = "error",
    replace_with: float = 0.0,
    clip_lower: float | None = None,
    clip_upper: float | None = None,
) -> TimeSeries:
    """Handle NaN/Inf in `TimeSeries.values`."""
    x = np.asarray(ts.values, dtype=np.float64)
    mask = ~np.isfinite(x)
    if not mask.any():
        return ts

    if strategy == "error":
        raise ValueError("TimeSeries contains non-finite values (NaN/Inf).")

    out = x.copy()

    if strategy == "replace":
        out[mask] = float(replace_with)
    elif strategy == "clip":
        if clip_lower is None and clip_upper is None:
            raise ValueError("clip strategy requires clip_lower and/or clip_upper")

        nan_mask = np.isnan(out)
        if nan_mask.any():
            if clip_lower is not None and clip_upper is not None:
                out[nan_mask] = 0.5 * (float(clip_lower) + float(clip_upper))
            elif clip_lower is not None:
                out[nan_mask] = float(clip_lower)
            else:
                out[nan_mask] = float(clip_upper)  # type: ignore[arg-type]

        if clip_lower is not None:
            out[np.isneginf(out)] = float(clip_lower)
        if clip_upper is not None:
            out[np.isposinf(out)] = float(clip_upper)
    else:  # pragma: no cover
        raise ValueError(f"Unknown strategy {strategy!r}")

    return TimeSeries(
        values=out,
        timestamps=ts.timestamps.copy(),
        channel_names=list(ts.channel_names),
        units=list(ts.units),
        sample_rate=ts.sample_rate,
    )
