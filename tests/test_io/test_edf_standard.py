"""Tests for multiplexed standard EDF I/O."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from somnio.data.units import V

pytest.importorskip("mne")
pytest.importorskip("edfio")

from somnio.data.timeseries import TimeSeries  # noqa: E402
from somnio.io.edf import standard  # noqa: E402


def _ts(n: int = 200, channels: int = 2, sample_rate: float = 100.0) -> TimeSeries:
    step = int(round(1e9 / sample_rate))
    # EDF header requires recording dates in 1985–2084 (edfio).
    base = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    ts = np.arange(n, dtype=np.int64) * step + base
    vals = np.random.RandomState(0).randn(n, channels).astype(np.float64)
    names = tuple(f"CH_{i}" for i in range(channels))
    units = tuple(V for _ in range(channels))
    return TimeSeries(
        values=vals,
        timestamps=ts,
        channel_names=names,
        units=units,
        sample_rate=sample_rate,
    )


def test_standard_edf_roundtrip(tmp_path: Path) -> None:
    pytest.skip("EDF export temporarily disabled (to_mne_raw NotImplemented)")
    data = _ts()
    path = tmp_path / "rec.edf"
    standard.write(path, data, overwrite=True)
    got = standard.read(path, preload=True, verbose="ERROR")
    np.testing.assert_allclose(got.values, data.values, rtol=0, atol=5e-5)
    np.testing.assert_array_equal(got.timestamps, data.timestamps)
    assert got.channel_names == data.channel_names
    assert got.units == data.units
    assert got.sample_rate == data.sample_rate


def test_standard_edf_rejects_pre_edf_epoch_timestamps() -> None:
    pytest.skip("EDF export temporarily disabled (to_mne_raw NotImplemented)")
    step = 10_000_000
    ts_bad = TimeSeries(
        values=np.zeros((4, 1)),
        timestamps=np.arange(4, dtype=np.int64) * step,
        channel_names=("a",),
        units=(V,),
        sample_rate=100.0,
    )
    with pytest.raises(ValueError, match="1985"):
        standard.write(Path("unused.edf"), ts_bad, overwrite=True)


def test_standard_edf_reader_writer_protocol(tmp_path: Path) -> None:
    pytest.skip("EDF export temporarily disabled (to_mne_raw NotImplemented)")
    data = _ts()
    path = tmp_path / "r.edf"
    standard.StandardEDFWriter().write(path, data, overwrite=True)
    got = standard.StandardEDFReader().read(path, verbose="ERROR")
    np.testing.assert_allclose(got.values, data.values, rtol=0, atol=5e-5)
