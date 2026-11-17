"""Tests for model.yaml discovery."""

from pathlib import Path

import pytest

from somnio.tasks.sleep_scoring.models.metadata import discover_metadata_path


def test_discover_stem_yaml(tmp_path: Path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    meta = tmp_path / "model.yaml"
    meta.write_text("sample_rate_hz: 1\n")
    assert discover_metadata_path(model) == meta


def test_discover_inside_directory(tmp_path: Path):
    saved = tmp_path / "tf"
    saved.mkdir()
    meta = saved / "model.yaml"
    meta.write_text("sample_rate_hz: 1\n")
    assert discover_metadata_path(saved) == meta


def test_discover_missing_raises(tmp_path: Path):
    model = tmp_path / "weights.onnx"
    model.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="sidecar"):
        discover_metadata_path(model)
