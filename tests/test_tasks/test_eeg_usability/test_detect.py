"""Tests for somnio.tasks.eeg_usability.detect.

Requires the ``eeg-usability`` extra.  All tests are skipped when optional
dependencies are not installed so the default test suite stays light.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("requests")

from somnio.data.timeseries import TimeSeries
from somnio.tasks.eeg_usability import detect
from somnio.tasks.eeg_usability.defaults import (
    EPOCH_DURATION_S,
    MODEL_NAMES,
    N_FEATURES_LITE,
    OUTPUT_CHANNEL_NAMES,
    OUTPUT_SAMPLE_RATE_HZ,
    SAMPLE_RATE_HZ,
)
from somnio.tasks.eeg_usability.detect import (
    get_usability_score,
    get_usability_scores,
    load_model,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_T0_NS = 1_700_000_000_000_000_000
_STEP_NS = int(1e9 / SAMPLE_RATE_HZ)
_EPOCH_LENGTH = int(EPOCH_DURATION_S * SAMPLE_RATE_HZ)


def _make_timestamps(n: int) -> np.ndarray:
    return (_T0_NS + np.arange(n, dtype=np.int64) * _STEP_NS).astype(np.int64)


def _make_eeg_ts(
    n_samples: int,
    *,
    channel_names: tuple[str, ...] = ("EEG_L", "EEG_R", "MOVEMENT"),
    units: tuple[str, ...] | None = None,
    sample_rate: float = SAMPLE_RATE_HZ,
) -> TimeSeries:
    if units is None:
        units = tuple("V" if name.startswith("EEG") else "1" for name in channel_names)
    values = np.zeros((n_samples, len(channel_names)), dtype=np.float64)
    return TimeSeries(
        values=values,
        timestamps=_make_timestamps(n_samples),
        channel_names=channel_names,
        units=units,
        sample_rate=sample_rate,
    )


class StubModel:
    """Minimal classifier stub with configurable feature count and labels."""

    def __init__(
        self,
        num_features: int = N_FEATURES_LITE,
        *,
        left_label: int = 0,
        right_label: int = 1,
    ) -> None:
        self._num_features = num_features
        self._left_label = left_label
        self._right_label = right_label
        self.predict_calls: list[np.ndarray] = []

    def num_feature(self) -> int:
        return self._num_features

    def predict(self, samples: np.ndarray) -> np.ndarray:
        self.predict_calls.append(samples.copy())
        n = samples.shape[0]
        label = (
            self._left_label if len(self.predict_calls) % 2 == 1 else self._right_label
        )
        probs = np.zeros((n, 2), dtype=np.float64)
        probs[np.arange(n), label] = 1.0
        return probs


def _stub_spectrogram_features(data: np.ndarray, sample_rate: float) -> np.ndarray:
    del sample_rate
    num_epochs, num_channels, _ = data.shape
    return np.zeros((num_epochs, num_channels, 2, 2), dtype=np.float32)


@pytest.fixture(autouse=True)
def _patch_feature_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid slow spectrogram work in every test."""
    monkeypatch.setattr(
        detect, "_extract_spectrogram_features", _stub_spectrogram_features
    )


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_raises_when_sample_rate_is_none(self):
        n = _EPOCH_LENGTH
        ts = TimeSeries(
            values=np.zeros((n, 3)),
            timestamps=_make_timestamps(n),
            channel_names=("EEG_L", "EEG_R", "MOVEMENT"),
            units=("V", "V", "1"),
            sample_rate=None,
        )
        model = StubModel()
        with pytest.raises(ValueError, match="sample_rate"):
            get_usability_scores(
                ts, model, eeg_left="EEG_L", eeg_right="EEG_R", movement="MOVEMENT"
            )

    def test_raises_when_sample_rate_is_not_256_hz(self):
        ts = _make_eeg_ts(_EPOCH_LENGTH, sample_rate=128.0)
        model = StubModel()
        with pytest.raises(ValueError, match="256"):
            get_usability_scores(
                ts, model, eeg_left="EEG_L", eeg_right="EEG_R", movement="MOVEMENT"
            )

    def test_raises_when_required_channels_missing(self):
        ts = _make_eeg_ts(_EPOCH_LENGTH)
        model = StubModel()
        with pytest.raises(ValueError, match="channels"):
            get_usability_scores(
                ts,
                model,
                eeg_left="EEG_L",
                eeg_right="MISSING",
                movement="MOVEMENT",
            )

    def test_raises_when_recording_shorter_than_one_epoch(self):
        ts = _make_eeg_ts(_EPOCH_LENGTH - 1)
        model = StubModel()
        with pytest.raises(ValueError, match="No epochs"):
            get_usability_scores(
                ts, model, eeg_left="EEG_L", eeg_right="EEG_R", movement="MOVEMENT"
            )

    def test_rejects_non_lite_model(self):
        ts = _make_eeg_ts(_EPOCH_LENGTH)
        model = StubModel(N_FEATURES_LITE + 1)
        with pytest.raises(ValueError, match="Only lite"):
            get_usability_scores(
                ts, model, eeg_left="EEG_L", eeg_right="EEG_R", movement="MOVEMENT"
            )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_get_usability_scores_shape_and_metadata(self):
        ts = _make_eeg_ts(_EPOCH_LENGTH * 2)
        model = StubModel()

        scores, samples_to_keep, epoch_length = get_usability_scores(
            ts, model, eeg_left="EEG_L", eeg_right="EEG_R", movement="MOVEMENT"
        )

        assert scores.channel_names == OUTPUT_CHANNEL_NAMES
        assert scores.sample_rate == OUTPUT_SAMPLE_RATE_HZ
        assert samples_to_keep == _EPOCH_LENGTH * 2
        assert epoch_length == _EPOCH_LENGTH
        assert scores.n_samples == 2
        assert scores.values.shape == (2, 2)

    def test_get_usability_score_single_channel(self):
        ts = _make_eeg_ts(_EPOCH_LENGTH, channel_names=("EEG_L", "MOVEMENT"))
        model = StubModel()

        scores, samples_to_keep, epoch_length = get_usability_score(
            ts, model, eeg="EEG_L", movement="MOVEMENT", output_channel="usability"
        )

        assert scores.channel_names == ("usability",)
        assert tuple(u.symbol for u in scores.units) == ("1",)
        assert scores.sample_rate == OUTPUT_SAMPLE_RATE_HZ
        assert samples_to_keep == _EPOCH_LENGTH
        assert epoch_length == _EPOCH_LENGTH
        assert scores.n_samples == 1

    def test_predictions_use_stub_model_labels(self):
        ts = _make_eeg_ts(_EPOCH_LENGTH * 2)
        model = StubModel(left_label=1, right_label=0)

        scores, _, _ = get_usability_scores(
            ts, model, eeg_left="EEG_L", eeg_right="EEG_R", movement="MOVEMENT"
        )

        assert scores.values.shape == (2, 2)
        np.testing.assert_array_equal(scores.values[:, 0], [1.0, 1.0])
        np.testing.assert_array_equal(scores.values[:, 1], [0.0, 0.0])
        assert len(model.predict_calls) == 2

    def test_lite_sample_feature_count(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.undo()

        n_samples = int(SAMPLE_RATE_HZ * EPOCH_DURATION_S)
        data = (
            np.random.default_rng(0)
            .standard_normal((2, 2, n_samples))
            .astype(np.float32)
        )
        spec = detect._extract_spectrogram_features(data, SAMPLE_RATE_HZ)
        samples = detect._create_lite_sample(spec, eeg_idx=0, movement_idx=1)
        assert samples.shape == (2, N_FEATURES_LITE)


# ---------------------------------------------------------------------------
# Model catalog and cache
# ---------------------------------------------------------------------------


class TestModelCatalog:
    def test_only_lite_models_are_exported(self):
        assert set(MODEL_NAMES) == {"lite", "lite_binary"}


class TestLoadModelCache:
    def test_download_writes_via_temp_file(self, tmp_path: Path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_name = MODEL_NAMES["lite"]
        model_path = models_dir / model_name
        payload = {"model": StubModel()}

        monkeypatch.setattr(detect, "get_models_dir", lambda: models_dir, raising=False)
        monkeypatch.setattr(
            detect,
            "_get_available_models",
            lambda: {model_name: "https://example.com/model.pkl"},
        )

        class FakeResponse:
            content = pickle.dumps(payload)

            @staticmethod
            def raise_for_status() -> None:
                return None

        monkeypatch.setattr(
            detect.requests, "get", lambda *args, **kwargs: FakeResponse()
        )

        load_model("lite")

        assert model_path.exists()
        assert not list(models_dir.glob(f".{model_name}.*.tmp"))

    def test_corrupt_cache_is_refetched(self, tmp_path: Path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_name = MODEL_NAMES["lite"]
        model_path = models_dir / model_name
        model_path.write_bytes(b"not a pickle")
        payload = {"model": StubModel()}
        download_calls: list[str] = []

        monkeypatch.setattr(detect, "get_models_dir", lambda: models_dir, raising=False)
        monkeypatch.setattr(
            detect,
            "_get_available_models",
            lambda: {model_name: "https://example.com/model.pkl"},
        )

        class FakeResponse:
            content = pickle.dumps(payload)

            @staticmethod
            def raise_for_status() -> None:
                return None

        def fake_get(*args, **kwargs):
            download_calls.append(model_name)
            return FakeResponse()

        monkeypatch.setattr(detect.requests, "get", fake_get)

        model = load_model("lite")

        assert download_calls == [model_name]
        assert isinstance(model, StubModel)
        loaded_payload = pickle.loads(model_path.read_bytes())
        assert isinstance(loaded_payload, dict)
        assert "model" in loaded_payload

    def test_failed_download_does_not_leave_final_path(
        self, tmp_path: Path, monkeypatch
    ):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_name = MODEL_NAMES["lite"]
        model_path = models_dir / model_name

        monkeypatch.setattr(detect, "get_models_dir", lambda: models_dir, raising=False)
        monkeypatch.setattr(
            detect,
            "_get_available_models",
            lambda: {model_name: "https://example.com/model.pkl"},
        )

        class FakeResponse:
            content = b"partial"

            @staticmethod
            def raise_for_status() -> None:
                return None

        original_replace = Path.replace

        def failing_replace(self, target):
            if str(self).endswith(".tmp"):
                raise OSError("disk full")
            return original_replace(self, target)

        monkeypatch.setattr(
            detect.requests, "get", lambda *args, **kwargs: FakeResponse()
        )
        monkeypatch.setattr(Path, "replace", failing_replace)

        with pytest.raises(OSError, match="disk full"):
            load_model("lite")

        assert not model_path.exists()
        assert not list(models_dir.glob(f".{model_name}.*.tmp"))

    def test_unknown_version_raises(self):
        with pytest.raises(ValueError, match="not found"):
            load_model("default")
