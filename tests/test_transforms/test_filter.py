"""Tests for somnio.transforms.filter."""

from __future__ import annotations

import numpy as np
import pytest

from somnio.data import TimeSeries
from somnio.transforms.filter import (
    _classify_filter_type,
    apply_butterworth_filter,
    apply_filter,
    apply_fir_filter,
)


FS = 128.0
DURATION_S = 180.0  # 3 minutes — long enough for a 0.3 Hz FIR


def _make_ts(
    values: np.ndarray,
    sample_rate: float = FS,
) -> TimeSeries:
    n = values.shape[0]
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    timestamps = (np.arange(n, dtype=np.int64) * (1e9 / sample_rate)).astype(np.int64)
    n_ch = values.shape[1]
    return TimeSeries(
        values=values.astype(np.float64),
        timestamps=timestamps,
        channel_names=tuple(f"ch{i}" for i in range(n_ch)),
        units=tuple("V" for _ in range(n_ch)),
        sample_rate=sample_rate,
    )


def _drift_plus_tone(
    sample_rate: float = FS,
    duration_s: float = DURATION_S,
    drift_hz: float = 0.05,
    drift_amp: float = 1000.0,
    tone_hz: float = 10.0,
    tone_amp: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    drift = drift_amp * np.sin(2 * np.pi * drift_hz * t)
    tone = tone_amp * np.sin(2 * np.pi * tone_hz * t)
    return drift + tone, t


def _band_power(x: np.ndarray, sample_rate: float, f_lo: float, f_hi: float) -> float:
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    psd = np.abs(np.fft.rfft(x)) ** 2
    mask = (freqs >= f_lo) & (freqs < f_hi)
    return float(psd[mask].sum())


def _attenuation_db(
    before: np.ndarray,
    after: np.ndarray,
    sample_rate: float,
    f_lo: float,
    f_hi: float,
) -> float:
    p0 = _band_power(before, sample_rate, f_lo, f_hi)
    p1 = _band_power(after, sample_rate, f_lo, f_hi)
    assert p0 > 0
    return float(10.0 * np.log10(p1 / p0))


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (0.3, None, "highpass"),
        (None, 35.0, "lowpass"),
        (0.3, 35.0, "bandpass"),
        (35.0, 0.3, "bandstop"),
    ],
)
def test_classify_filter_type(low, high, expected):
    assert _classify_filter_type(low, high) == expected


def test_noop_both_none():
    ts = _make_ts(np.ones(100))
    out = apply_filter(ts, None, None)
    assert out is ts


def test_apply_fir_filter_delegates_and_rejects_drift():
    signal, _ = _drift_plus_tone()
    ts = _make_ts(signal)
    out = apply_fir_filter(ts, low_cutoff=0.3, high_cutoff=35.0)

    x = ts.values[:, 0]
    y = out.values[:, 0]
    atten = _attenuation_db(x, y, FS, 0.0, 0.2)
    assert atten < -20.0, f"expected ≫20 dB drift attenuation, got {atten:.1f} dB"
    assert abs(float(np.mean(y))) < 5.0

    # Passband tone (~10 Hz) should remain.
    tone_before = _band_power(x, FS, 9.0, 11.0)
    tone_after = _band_power(y, FS, 9.0, 11.0)
    assert tone_after > 0.1 * tone_before


def test_apply_filter_fir_highpass_rejects_drift():
    signal, _ = _drift_plus_tone()
    ts = _make_ts(signal)
    out = apply_filter(ts, low_cutoff=0.3, method="fir")
    atten = _attenuation_db(ts.values[:, 0], out.values[:, 0], FS, 0.0, 0.2)
    assert atten < -20.0


def test_apply_filter_iir_rejects_drift():
    signal, _ = _drift_plus_tone()
    ts = _make_ts(signal)
    out = apply_filter(ts, low_cutoff=0.3, high_cutoff=35.0, method="iir")

    x = ts.values[:, 0]
    y = out.values[:, 0]
    atten = _attenuation_db(x, y, FS, 0.0, 0.2)
    assert atten < -20.0, f"expected ≫20 dB drift attenuation, got {atten:.1f} dB"
    assert abs(float(np.mean(y))) < 5.0

    tone_before = _band_power(x, FS, 9.0, 11.0)
    tone_after = _band_power(y, FS, 9.0, 11.0)
    assert tone_after > 0.1 * tone_before


def test_apply_butterworth_filter_alias():
    signal, _ = _drift_plus_tone()
    ts = _make_ts(signal)
    out = apply_butterworth_filter(ts, low_cutoff=0.3, high_cutoff=35.0, order=4)
    atten = _attenuation_db(ts.values[:, 0], out.values[:, 0], FS, 0.0, 0.2)
    assert atten < -20.0


@pytest.mark.parametrize("method", ["fir", "iir"])
def test_lowpass_and_bandstop_run(method):
    signal, _ = _drift_plus_tone()
    ts = _make_ts(signal)
    lp = apply_filter(ts, high_cutoff=20.0, method=method)
    assert lp.values.shape == ts.values.shape
    # bandstop: stop between 8 and 12 Hz (API: low > high)
    bs = apply_filter(ts, low_cutoff=12.0, high_cutoff=8.0, method=method)
    assert bs.values.shape == ts.values.shape


def test_short_signal_raises_for_aggressive_fir_highpass():
    # A few seconds is far too short for an auto 0.3 Hz FIR at 128 Hz.
    n = int(FS * 5)
    ts = _make_ts(np.random.default_rng(0).standard_normal(n))
    with pytest.raises(ValueError, match="too short"):
        apply_filter(ts, low_cutoff=0.3, method="fir")


def test_short_signal_iir_still_works():
    n = int(FS * 5)
    t = np.arange(n) / FS
    signal = 1000.0 * np.sin(2 * np.pi * 0.05 * t) + 10.0 * np.sin(2 * np.pi * 10.0 * t)
    ts = _make_ts(signal)
    out = apply_filter(ts, low_cutoff=0.3, method="iir")
    assert out.values.shape == ts.values.shape
    assert abs(float(np.mean(out.values[:, 0]))) < abs(float(np.mean(ts.values[:, 0])))


def test_cutoff_at_or_above_nyquist_raises():
    ts = _make_ts(np.ones(1000))
    with pytest.raises(ValueError, match="Nyquist"):
        apply_filter(ts, high_cutoff=FS / 2.0)


def test_equal_cutoffs_raise():
    ts = _make_ts(np.ones(1000))
    with pytest.raises(ValueError, match="must differ"):
        apply_filter(ts, low_cutoff=10.0, high_cutoff=10.0)


def test_missing_sample_rate_raises():
    ts = TimeSeries(
        values=np.ones((100, 1)),
        timestamps=np.arange(100, dtype=np.int64),
        channel_names=("ch0",),
        units=("V",),
        sample_rate=None,
    )
    with pytest.raises(ValueError, match="sample_rate"):
        apply_filter(ts, low_cutoff=1.0)


def test_unknown_method_raises():
    ts = _make_ts(np.ones(1000))
    with pytest.raises(ValueError, match="method"):
        apply_filter(ts, low_cutoff=1.0, method="cheby")  # type: ignore[arg-type]


def test_explicit_numtaps_too_long_raises():
    ts = _make_ts(np.ones(100))
    with pytest.raises(ValueError, match="numtaps"):
        apply_filter(ts, high_cutoff=20.0, method="fir", numtaps=99)
