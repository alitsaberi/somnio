"""Scaling transforms for :class:`~somnio.data.timeseries.TimeSeries`."""

from __future__ import annotations

from typing import Literal

import numpy as np

from somnio.data.timeseries import TimeSeries


ScaleMethod = Literal["zscore", "standard", "robust"]


def _as_per_channel_param(
    x: float | list[float] | tuple[float, ...] | np.ndarray | None,
    *,
    n_channels: int,
    name: str,
) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return np.full((n_channels,), float(x), dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != (n_channels,):
        raise ValueError(f"{name} must have shape (n_channels,), got {arr.shape}")
    return arr


def _iqr(x: np.ndarray, *, axis: int) -> np.ndarray:
    q75 = np.quantile(x, 0.75, axis=axis)
    q25 = np.quantile(x, 0.25, axis=axis)
    return q75 - q25


def apply_scale(
    ts: TimeSeries,
    *,
    method: ScaleMethod = "zscore",
    center: float | list[float] | tuple[float, ...] | np.ndarray | None = None,
    scale: float | list[float] | tuple[float, ...] | np.ndarray | None = None,
    eps: float = 1e-12,
) -> TimeSeries:
    """Scale a `TimeSeries` per channel.

    Args:
        ts: Input time-series.
        method: Scaling strategy when `center`/`scale` are not explicitly provided.
            - `"zscore"` / `"standard"`: center=mean, scale=std
            - `"robust"`: center=median, scale=IQR (Q75-Q25)
        center: Optional per-channel center. If provided, overrides `method`'s center.
        scale: Optional per-channel scale. If provided, overrides `method`'s scale.
        eps: Lower bound for the effective scale to avoid division by ~0.

    Returns:
        New `TimeSeries` with scaled values. Timestamps and channel metadata are preserved.
    """
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    values = np.asarray(ts.values, dtype=np.float64)
    n_channels = ts.n_channels
    center_arr = _as_per_channel_param(center, n_channels=n_channels, name="center")
    scale_arr = _as_per_channel_param(scale, n_channels=n_channels, name="scale")

    if center_arr is None or scale_arr is None:
        axis = 0  # time axis
        if method in ("zscore", "standard"):
            if center_arr is None:
                center_arr = np.mean(values, axis=axis)
            if scale_arr is None:
                scale_arr = np.std(values, axis=axis, ddof=0)
        elif method == "robust":
            if center_arr is None:
                center_arr = np.median(values, axis=axis)
            if scale_arr is None:
                scale_arr = _iqr(values, axis=axis)
        else:  # pragma: no cover
            raise ValueError(f"Unknown scaling method {method!r}")

    assert center_arr is not None
    assert scale_arr is not None

    denom = np.maximum(scale_arr, eps)
    out = (values - center_arr.reshape(1, -1)) / denom.reshape(1, -1)

    return TimeSeries(
        values=out,
        timestamps=ts.timestamps.copy(),
        channel_names=list(ts.channel_names),
        units=list(ts.units),
        sample_rate=ts.sample_rate,
    )
