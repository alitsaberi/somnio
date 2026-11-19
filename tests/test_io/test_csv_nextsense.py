"""Tests for NextSense CSV reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from somnio.data.units import UV
from somnio.io.csv import read_nextsense
from somnio.io.csv.nextsense import list_channels, read_metadata


def _write_nextsense(
    path: Path,
    *,
    agent: str = "Left",
    eeg_col: str = "ELW-ELC",
    eeg_values: list[float] | None = None,
    acc_prefix: str = "ACC_L",
) -> None:
    if eeg_values is None:
        eeg_values = [1.0, 2.0, 3.0, 4.0]
    lines = [
        "#Header version 1.0",
        "#eegStreamingRate: 100.0",
        "#accelerationStreamingRate: 25.0",
        f"#Agent device: {agent}",
        (
            f"#SAMPLE_NUMBER,{eeg_col},{eeg_col}-FILTERED,ERW-ERC,ERW-ERC-FILTERED,"
            f"{acc_prefix}_X,{acc_prefix}_Y,{acc_prefix}_Z,"
            "ACC_R_X,ACC_R_Y,ACC_R_Z,GYRO_L_X,GYRO_L_Y,GYRO_L_Z,"
            "GYRO_R_X,GYRO_R_Y,GYRO_R_Z,SAMPLING_TIMESTAMP,RECEPTION_TIME"
        ),
        "#========== Start Data ==========",
    ]
    base_ms = 1_700_000_000_000
    for i, value in enumerate(eeg_values):
        acc = (10.0, 20.0, 30.0) if i % 4 == 0 else (0.0, 0.0, 0.0)
        row = [
            str(i + 1),
            str(value),
            "0.0",
            "0.0",
            "0.0",
            str(acc[0]),
            str(acc[1]),
            str(acc[2]),
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            str(base_ms + i * 10),
            str(base_ms + i * 10),
        ]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_nextsense_single_file(tmp_path: Path) -> None:
    path = tmp_path / "left.csv"
    _write_nextsense(path)
    ts = read_nextsense(path)
    assert ts.sample_rate == 100.0
    assert ts.n_samples == 4
    assert "ELW_ELC" in ts.channel_names
    assert ts.units[ts.channel_index_map["ELW_ELC"]] == UV
    np.testing.assert_allclose(
        ts.values[:, ts.channel_index_map["ELW_ELC"]], [1, 2, 3, 4]
    )


def test_read_nextsense_channel_subset(tmp_path: Path) -> None:
    path = tmp_path / "left.csv"
    _write_nextsense(path)
    ts = read_nextsense(path, channels=["ELW_ELC", "ACC_L_X"])
    assert ts.channel_names == ("ELW_ELC", "ACC_L_X")


def test_list_channels_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "left.csv"
    _write_nextsense(path)
    assert "ELW_ELC" in list_channels(path)
    meta = read_metadata(path)
    assert meta["eegStreamingRate"] == "100.0"


def test_read_nextsense_pair_merges_sides(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_nextsense(left, agent="Left", eeg_values=[1.0, 1.0, 1.0, 1.0])
    # Right file: fill ERW-ERC (column index 3 in header)
    lines = [
        "#Header version 1.0",
        "#eegStreamingRate: 100.0",
        "#accelerationStreamingRate: 25.0",
        "#Agent device: Right",
        (
            "#SAMPLE_NUMBER,ELW-ELC,ELW-ELC-FILTERED,ERW-ERC,ERW-ERC-FILTERED,"
            "ACC_L_X,ACC_L_Y,ACC_L_Z,ACC_R_X,ACC_R_Y,ACC_R_Z,"
            "GYRO_L_X,GYRO_L_Y,GYRO_L_Z,GYRO_R_X,GYRO_R_Y,GYRO_R_Z,"
            "SAMPLING_TIMESTAMP,RECEPTION_TIME"
        ),
        "#========== Start Data ==========",
    ]
    base_ms = 1_700_000_000_000
    for i, value in enumerate([5.0, 6.0, 7.0, 8.0]):
        row = [
            str(i + 1),
            "0.0",
            "0.0",
            str(value),
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "1.0",
            "2.0",
            "3.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            str(base_ms + i * 10),
            str(base_ms + i * 10),
        ]
        lines.append(",".join(row))
    right.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ts = read_nextsense(
        [left, right], channels=["ELW_ELC", "ERW_ERC", "ACC_L_X", "ACC_R_X"]
    )
    assert "ELW_ELC" in ts.channel_names
    assert "ERW_ERC" in ts.channel_names
    np.testing.assert_allclose(
        ts.values[:, ts.channel_index_map["ELW_ELC"]], [1, 1, 1, 1]
    )
    np.testing.assert_allclose(
        ts.values[:, ts.channel_index_map["ERW_ERC"]], [5, 6, 7, 8]
    )


def test_read_nextsense_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_nextsense(tmp_path / "missing.csv")
