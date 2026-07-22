"""EEG usability scoring task.

Predicts left/right EEG usability labels from 256 Hz EEG and movement data
using eegFloss *lite* (spectrogram-only) models.

Derived from the eegUsability models in `eegFloss
<https://github.com/Niloy333/eegFloss>`_.
"""

from somnio.tasks.eeg_usability.defaults import (
    EPOCH_DURATION_S,
    MODEL_NAMES,
    MODELS_INFO_URL,
    N_FEATURES_LITE,
    OUTPUT_CHANNEL_NAMES,
    OUTPUT_SAMPLE_RATE_HZ,
    OUTPUT_UNITS,
    SAMPLE_RATE_HZ,
    get_models_dir,
)
from somnio.tasks.eeg_usability.detect import (
    get_usability_score,
    get_usability_scores,
    load_model,
)

__all__ = [
    "EPOCH_DURATION_S",
    "MODEL_NAMES",
    "MODELS_INFO_URL",
    "N_FEATURES_LITE",
    "OUTPUT_CHANNEL_NAMES",
    "OUTPUT_SAMPLE_RATE_HZ",
    "OUTPUT_UNITS",
    "SAMPLE_RATE_HZ",
    "get_models_dir",
    "get_usability_score",
    "get_usability_scores",
    "load_model",
]
