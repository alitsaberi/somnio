"""Tests for sleep-scoring ModelMetadata."""

import pytest

from somnio.tasks.sleep_scoring.schema import ModelMetadata


def _base(**kwargs):
    data = dict(
        sample_rate_hz=128.0,
        n_periods_per_window=35,
        n_samples_per_period=3840,
        n_samples_per_prediction=3840,
        n_channels=1,
        class_labels=["W", "N1", "N2", "N3", "REM"],
    )
    data.update(kwargs)
    return data


def test_fixed_window_metadata():
    md = ModelMetadata.model_validate(_base())
    assert md.n_periods_per_window == 35
    assert md.n_predictions_per_period == 1


def test_dynamic_null_n_periods():
    md = ModelMetadata.model_validate(_base(n_periods_per_window=None))
    assert md.n_periods_per_window is None


def test_rejects_zero_n_periods():
    with pytest.raises(ValueError, match="n_periods_per_window"):
        ModelMetadata.model_validate(_base(n_periods_per_window=0))


def test_rejects_duplicate_labels():
    with pytest.raises(ValueError, match="unique"):
        ModelMetadata.model_validate(_base(class_labels=["W", "W"]))


def test_period_not_divisible_by_prediction():
    with pytest.raises(ValueError, match="divisible"):
        ModelMetadata.model_validate(
            _base(n_samples_per_period=100, n_samples_per_prediction=30)
        )
