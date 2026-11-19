"""Tests for EDF/BDF channel subset loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mne")

import mne  # noqa: E402

from somnio.io.edf import read_standard  # noqa: E402


def _write_bdf(path: Path, *, n: int = 100, sfreq: float = 100.0) -> None:
    data = np.random.default_rng(0).standard_normal((3, n)) * 1e-6
    info = mne.create_info(
        ["EEG1", "EEG2", "ACC"],
        sfreq,
        ch_types=["eeg", "eeg", "misc"],
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    mne.export.export_raw(str(path), raw, fmt="bdf", overwrite=True, verbose="ERROR")


def test_read_standard_bdf_loads_all_channels(tmp_path: Path) -> None:
    path = tmp_path / "rec.bdf"
    _write_bdf(path)
    ts = read_standard(path, verbose="ERROR")
    assert ts.channel_names == ("EEG1", "EEG2", "ACC")
    assert ts.n_samples == 100
    assert ts.sample_rate == 100.0


def test_read_standard_bdf_channel_subset(tmp_path: Path) -> None:
    path = tmp_path / "rec.bdf"
    _write_bdf(path)
    ts = read_standard(path, channels=["EEG1", "ACC"], verbose="ERROR")
    assert ts.channel_names == ("EEG1", "ACC")
    assert ts.n_channels == 2


def test_read_standard_bdf_missing_channel_raises(tmp_path: Path) -> None:
    path = tmp_path / "rec.bdf"
    _write_bdf(path)
    with pytest.raises(ValueError, match="not found"):
        read_standard(path, channels=["NOPE"], verbose="ERROR")
