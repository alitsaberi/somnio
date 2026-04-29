"""Resample :class:`~somnio.data.timeseries.TimeSeries` to a target rate (SciPy).

Requires the ``signal`` extra::

    pip install 'somnio[signal]'
"""

from __future__ import annotations

import numpy as np

from somnio.data.timeseries import TimeSeries
from somnio.utils.imports import MissingOptionalDependency

try:
    from scipy.signal import resample as scipy_resample
except ModuleNotFoundError as exc:
    raise MissingOptionalDependency(
        "scipy", extra="signal", purpose="Resampling transforms"
    ) from exc


def apply_resample(ts: TimeSeries, target_sample_rate_hz: float) -> TimeSeries:
    """FFT resampling along time (uniform grid). Preserves first timestamp; spacing updates.

    Args:
        ts: Series with ``sample_rate`` set (required).
        target_sample_rate_hz: Desired sample rate in Hz.

    Returns:
        New ``TimeSeries`` at ``target_sample_rate_hz`` with the same channel metadata,
        or ``ts`` unchanged if rates are equal within ``1e-6`` Hz.

    Raises:
        ValueError: If ``target_sample_rate_hz <= 0`` or ``ts.sample_rate`` is ``None``.
    """
    if ts.sample_rate is None:
        raise ValueError(
            "TimeSeries.sample_rate must not be None; resampling needs a source rate."
        )

    if target_sample_rate_hz <= 0:
        raise ValueError(
            f"target_sample_rate_hz must be > 0, got {target_sample_rate_hz}"
        )

    if np.isclose(ts.sample_rate, target_sample_rate_hz, rtol=0.0, atol=1e-6):
        return ts

    n_in = ts.n_samples
    if n_in < 2:
        raise ValueError(f"need at least 2 samples to resample, got {n_in}")

    n_out = max(2, int(round(n_in * target_sample_rate_hz / ts.sample_rate)))
    new_values = scipy_resample(ts.values, n_out, axis=0)
    step_ns = int(round(1e9 / target_sample_rate_hz))
    new_timestamps = ts.timestamps[0] + np.arange(n_out, dtype=np.int64) * step_ns

    return TimeSeries(
        values=np.asarray(new_values, dtype=np.float64),
        timestamps=new_timestamps,
        channel_names=list(ts.channel_names),
        units=list(ts.units),
        sample_rate=target_sample_rate_hz,
    )
