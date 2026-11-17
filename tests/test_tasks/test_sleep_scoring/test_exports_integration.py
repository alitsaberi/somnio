"""Optional integration tests against real U-Time export artifacts.

Default root (override with ``SOMNIO_SLEEP_SCORING_EXPORTS``):

``/project/4180000.62/uz-sleep/models/u-sleep-nsrr-2024_eeg_finetune/repeated_training/depth_12/rep_05/exports``
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from somnio.data import TimeSeries
from somnio.tasks.sleep_scoring.score import score_sleep_stages

_DEFAULT_EXPORTS = Path(
    "/project/4180000.62/uz-sleep/models/u-sleep-nsrr-2024_eeg_finetune/"
    "repeated_training/depth_12/rep_05/exports"
)


def _exports_root() -> Path:
    override = os.environ.get("SOMNIO_SLEEP_SCORING_EXPORTS")
    return Path(override) if override else _DEFAULT_EXPORTS


EXPORTS = _exports_root()
_HAVE_EXPORTS = EXPORTS.is_dir()

pytestmark = pytest.mark.skipif(
    not _HAVE_EXPORTS,
    reason=f"U-Time export fixtures not found at {EXPORTS}",
)


def _synthetic_batch(*, p: int, t: int, c: int) -> np.ndarray:
    return np.zeros((1, p, t, c), dtype=np.float32)


def _synthetic_ts(*, n_periods: int, t: int, c: int, sample_rate: float) -> TimeSeries:
    n_samples = n_periods * t
    values = np.zeros((n_samples, c), dtype=np.float64)
    step = int(round(1e9 / sample_rate))
    timestamps = np.arange(n_samples, dtype=np.int64) * step
    names = tuple(f"ch{i}" for i in range(c))
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=names,
        units=tuple("V" for _ in range(c)),
        sample_rate=sample_rate,
    )


def test_load_and_predict_onnx_default():
    pytest.importorskip("onnxruntime")
    from somnio.tasks.sleep_scoring.models.onnx import OnnxSleepScoringModel

    model = OnnxSleepScoringModel.load(EXPORTS / "onnx_default" / "model.onnx")
    md = model.metadata
    assert md.n_periods_per_window == 35
    batch = _synthetic_batch(
        p=35, t=md.n_samples_per_period, c=md.n_channels
    )
    out = model.predict(batch)
    assert out.shape[-1] == len(md.class_labels)
    assert out.shape[0] == 1
    assert out.shape[1] == 35


def test_load_and_predict_onnx_dynamic():
    pytest.importorskip("onnxruntime")
    from somnio.tasks.sleep_scoring.models.onnx import OnnxSleepScoringModel

    model = OnnxSleepScoringModel.load(EXPORTS / "onnx_dynamic" / "model.onnx")
    md = model.metadata
    assert md.n_periods_per_window is None
    p = 3
    batch = _synthetic_batch(p=p, t=md.n_samples_per_period, c=md.n_channels)
    out = model.predict(batch)
    assert out.shape[-1] == len(md.class_labels)
    assert out.shape[0] == 1
    assert out.shape[1] == p


def test_load_and_predict_tf_default():
    pytest.importorskip("tensorflow")
    from somnio.tasks.sleep_scoring.models.tensorflow import (
        TensorflowSleepScoringModel,
    )

    model = TensorflowSleepScoringModel.load(EXPORTS / "tf_default")
    md = model.metadata
    assert md.n_periods_per_window == 35
    batch = _synthetic_batch(
        p=35, t=md.n_samples_per_period, c=md.n_channels
    )
    out = model.predict(batch)
    assert out.shape[-1] == len(md.class_labels)
    assert out.shape[0] == 1
    assert out.shape[1] == 35


def test_load_and_predict_tf_dynamic():
    pytest.importorskip("tensorflow")
    from somnio.tasks.sleep_scoring.models.tensorflow import (
        TensorflowSleepScoringModel,
    )

    model = TensorflowSleepScoringModel.load(EXPORTS / "tf_dynamic")
    md = model.metadata
    assert md.n_periods_per_window is None
    p = 3
    batch = _synthetic_batch(p=p, t=md.n_samples_per_period, c=md.n_channels)
    out = model.predict(batch)
    assert out.shape[-1] == len(md.class_labels)
    assert out.shape[0] == 1
    assert out.shape[1] == p


def test_score_smoke_onnx_fixed_and_dynamic():
    pytest.importorskip("onnxruntime")
    from somnio.tasks.sleep_scoring.models.onnx import OnnxSleepScoringModel

    fixed = OnnxSleepScoringModel.load(EXPORTS / "onnx_default" / "model.onnx")
    md = fixed.metadata
    ts = _synthetic_ts(
        n_periods=md.n_periods_per_window or 2,
        t=md.n_samples_per_period,
        c=md.n_channels,
        sample_rate=md.sample_rate_hz,
    )
    probs = score_sleep_stages(ts, backend=fixed, metadata=md)
    assert probs.values.shape[1] == len(md.class_labels)
    assert probs.values.shape[0] == (md.n_periods_per_window or 2)

    dyn = OnnxSleepScoringModel.load(EXPORTS / "onnx_dynamic" / "model.onnx")
    md_d = dyn.metadata
    n_periods = 4
    ts_d = _synthetic_ts(
        n_periods=n_periods,
        t=md_d.n_samples_per_period,
        c=md_d.n_channels,
        sample_rate=md_d.sample_rate_hz,
    )
    probs_d = score_sleep_stages(ts_d, backend=dyn, metadata=md_d)
    assert probs_d.values.shape == (n_periods, len(md_d.class_labels))
