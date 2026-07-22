"""Default parameters for EEG usability scoring."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Input signal requirements
SAMPLE_RATE_HZ: float = 256.0
EPOCH_DURATION_S: float = 10.0

# Output score metadata
OUTPUT_CHANNEL_NAMES: tuple[str, str] = ("usability_left", "usability_right")
OUTPUT_UNITS: tuple[str, str] = ("1", "1")
OUTPUT_SAMPLE_RATE_HZ: float = 1.0 / EPOCH_DURATION_S

# Lite models use spectrogram features only (no TSFEL).
N_FEATURES_LITE: int = 2838

# Pre-trained lite model artifacts from eegFloss
MODEL_NAMES: dict[str, str] = {
    "lite": "eegUsability_model_v0.7_lite.pkl",
    "lite_binary": "eegUsability_model_v0.7.3_lite_binary.pkl",
}
MODELS_INFO_URL: str = (
    "https://drive.usercontent.google.com/download"
    "?id=1f55ko0vH8BUYO9HqAQAVH4iGTu3Xdyf8&export=download&authuser=0"
)


@lru_cache(maxsize=1)
def get_models_dir() -> Path:
    """Return the user-writable directory for cached usability models."""
    try:
        from platformdirs import user_cache_path
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "platformdirs is required to resolve the EEG usability model cache "
            "directory. Install the 'eeg-usability' extra: "
            "`pip install 'somnio[eeg-usability]'`."
        ) from exc

    return user_cache_path("somnio", appauthor=False) / "models" / "eeg_usability"
