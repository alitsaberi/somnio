"""EEG usability feature extraction, model loading, and scoring.

This module implements the eegUsability pipeline from `eegFloss
<https://github.com/Niloy333/eegFloss>`_ and loads pre-trained models
distributed with that project. If you use this functionality in research,
please cite Sikder et al. (2025), https://doi.org/10.48550/arXiv.2507.06433,
and the eegFloss software release, https://doi.org/10.5281/zenodo.15823969.
See also https://alitsaberi.github.io/somnio/user-guide/eeg-usability/.

Requires the ``eeg-usability`` extra::

    pip install 'somnio[eeg-usability]'
"""

from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from somnio.data.timeseries import TimeSeries
from somnio.data.units import UV, convert_values
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
from somnio.utils.imports import MissingOptionalDependency

try:
    import requests
except ModuleNotFoundError as exc:
    raise MissingOptionalDependency(
        "requests", extra="eeg-usability", purpose="EEG usability model download"
    ) from exc

try:
    import tsfel
except ModuleNotFoundError as exc:
    raise MissingOptionalDependency(
        "tsfel", extra="eeg-usability", purpose="EEG usability feature extraction"
    ) from exc

try:
    from joblib import Parallel, delayed
except ModuleNotFoundError as exc:
    raise MissingOptionalDependency(
        "joblib", extra="eeg-usability", purpose="EEG usability feature extraction"
    ) from exc

try:
    from scipy.signal import spectrogram
except ModuleNotFoundError as exc:
    raise MissingOptionalDependency(
        "scipy", extra="eeg-usability", purpose="EEG usability spectrogram features"
    ) from exc

logger = logging.getLogger(__name__)

_CGF_STATISTICAL: dict[str, Any] | None = None
_CGF_TEMPORAL: dict[str, Any] | None = None
_CGF_SPECTRAL: dict[str, Any] | None = None


def _get_available_models() -> dict[str, str]:
    try:
        response = requests.get(MODELS_INFO_URL, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to fetch EEG usability model metadata from {MODELS_INFO_URL!r}: "
            f"{exc}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "EEG usability model metadata response was not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            "EEG usability model metadata must be a JSON object mapping "
            f"model names to download URLs; got {type(payload).__name__}."
        )

    return payload


def _download_model(model_name: str) -> None:
    available_models = _get_available_models()
    if model_name not in available_models:
        raise ValueError(
            f"Model {model_name!r} not found in available models. "
            f"Available models: {sorted(available_models)}"
        )

    model_url = available_models[model_name]
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / model_name

    try:
        response = requests.get(model_url, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to download EEG usability model {model_name!r} from "
            f"{model_url!r}: {exc}"
        ) from exc

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=models_dir,
            prefix=f".{model_name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(response.content)
            tmp_path = Path(tmp.name)
        tmp_path.replace(model_path)
    except OSError:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _read_model_pickle(model_path: Path) -> object:
    with model_path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(
            f"EEG usability model pickle at {model_path!r} must contain a "
            f"'model' entry; got {type(payload).__name__}."
        )

    return payload["model"]


def _extract_spectrogram_features(data: np.ndarray, sample_rate: float) -> np.ndarray:
    num_epochs, num_channels, _ = data.shape

    data_reshaped = data.reshape(-1, data.shape[-1])

    _, _, sxx = spectrogram(data_reshaped, sample_rate, axis=-1)

    spectrogram_features = sxx.reshape(num_epochs, num_channels, *sxx.shape[1:])
    spectrogram_features = np.transpose(spectrogram_features, (0, 1, 3, 2))
    logger.debug("Spectrogram features shape: %s", spectrogram_features.shape)

    return spectrogram_features.astype(np.float32)


def _get_tsfel_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    global _CGF_STATISTICAL, _CGF_TEMPORAL, _CGF_SPECTRAL
    if _CGF_STATISTICAL is None:
        _CGF_STATISTICAL = tsfel.get_features_by_domain("statistical")
        _CGF_TEMPORAL = tsfel.get_features_by_domain("temporal")
        _CGF_SPECTRAL = tsfel.get_features_by_domain("spectral")
    return _CGF_STATISTICAL, _CGF_TEMPORAL, _CGF_SPECTRAL


def _extract_tsfel_features_per_channel(
    data: np.ndarray, sample_rate: float
) -> np.ndarray:
    cgf_statistical, cgf_temporal, cgf_spectral = _get_tsfel_configs()

    all_features: dict[str, Any] = {}
    all_features.update(
        tsfel.time_series_features_extractor(
            cgf_statistical, data, fs=sample_rate, verbose=0
        )
    )
    all_features.update(
        tsfel.time_series_features_extractor(
            cgf_temporal, data, fs=sample_rate, verbose=0
        )
    )
    all_features.update(
        tsfel.time_series_features_extractor(
            cgf_spectral, data, fs=sample_rate, verbose=0
        )
    )

    return np.array(list(all_features.values()), dtype=np.float32).T


def _extract_tsfel_features(data: np.ndarray, sample_rate: float) -> np.ndarray:
    num_epochs, num_channels, _ = data.shape

    reshaped_data = data.reshape(-1, data.shape[-1])

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_extract_tsfel_features_per_channel)(epoch_channel_data, sample_rate)
        for epoch_channel_data in reshaped_data
    )

    tsfel_features = np.array(results).reshape(num_epochs, num_channels, -1)
    logger.debug("TSFEL features shape: %s", tsfel_features.shape)

    return tsfel_features


def _create_lite_sample(
    spectrogram_features: np.ndarray,
    *,
    eeg_idx: int,
    movement_idx: int,
) -> np.ndarray:
    eeg = spectrogram_features[:, eeg_idx, :, :].reshape(
        spectrogram_features.shape[0], -1
    )
    movement = spectrogram_features[:, movement_idx, :, :].reshape(
        spectrogram_features.shape[0], -1
    )
    return np.concatenate([eeg, movement], axis=1)


def _create_sample(
    spectrogram_features: np.ndarray,
    statistical_features: np.ndarray,
    *,
    eeg_idx: int,
    movement_idx: int,
) -> np.ndarray:
    spec_feats = spectrogram_features[:, [eeg_idx, movement_idx], :, :]
    spec_feats = np.hstack((spec_feats[:, 0, :, :], spec_feats[:, 1, :, :]))
    spec_feats = np.hstack([spec_feats[:, i, :] for i in range(spec_feats.shape[1])])

    stat_feats = statistical_features[:, [eeg_idx, movement_idx], :]
    stat_feats = np.hstack((stat_feats[:, 0, :], stat_feats[:, 1, :]))

    return np.hstack((spec_feats, stat_feats))


def _build_model_samples(
    spectrogram_features: np.ndarray,
    model: object,
    *,
    eeg_idx: int,
    movement_idx: int,
    tsfel_features: np.ndarray | None = None,
) -> np.ndarray:
    num_features = model.num_feature()  # type: ignore[attr-defined]
    if num_features == N_FEATURES_LITE:
        logger.info("Using lite set of features: %s", num_features)
        return _create_lite_sample(
            spectrogram_features, eeg_idx=eeg_idx, movement_idx=movement_idx
        )

    logger.info("Using full set of features: %s", num_features)
    if tsfel_features is None:
        raise ValueError("TSFEL features are required for the full usability model.")
    return _create_sample(
        spectrogram_features,
        tsfel_features,
        eeg_idx=eeg_idx,
        movement_idx=movement_idx,
    )


def _predict_usability_labels(model: object, samples: np.ndarray) -> np.ndarray:
    predictions = model.predict(samples)  # type: ignore[attr-defined]
    return np.argmax(predictions, axis=1).astype(np.float64)


def _epoch_timestamps(
    timestamps: np.ndarray, epoch_length: int, n_epochs: int
) -> np.ndarray:
    return np.array(
        [
            int(np.mean(timestamps[i * epoch_length : (i + 1) * epoch_length]))
            for i in range(n_epochs)
        ],
        dtype=np.int64,
    )


def _prepare_epoch_array(
    ts: TimeSeries,
    channels: list[str],
    *,
    eeg_channels: set[str],
) -> tuple[TimeSeries, int, int, np.ndarray]:
    if ts.sample_rate is None:
        raise ValueError(
            "TimeSeries.sample_rate must not be None; "
            "EEG usability scoring requires a known sample rate."
        )

    if ts.sample_rate != SAMPLE_RATE_HZ:
        raise ValueError(f"Data must have a sample rate of {SAMPLE_RATE_HZ}")

    missing = set(channels) - set(ts.channel_names)
    if missing:
        raise ValueError(f"Data must have the following channels: {channels}")

    selected = ts.select_channels(channels)

    epoch_length = int(EPOCH_DURATION_S * selected.sample_rate)  # type: ignore[operator]
    n_epochs = selected.n_samples // epoch_length
    samples_to_keep = n_epochs * epoch_length
    logger.info(
        "Number of samples: %d, number of epochs: %d, samples to keep: %d",
        selected.n_samples,
        n_epochs,
        samples_to_keep,
    )

    if n_epochs == 0:
        raise ValueError(
            f"No epochs found in the data (epoch_length={epoch_length}, "
            f"data_length={selected.n_samples})."
        )

    if samples_to_keep < selected.n_samples:
        logger.info(
            "Dropping %d samples from the end of the data.",
            selected.n_samples - samples_to_keep,
        )
        selected = selected[:samples_to_keep]

    values = selected.values.copy()
    for channel_index, channel_name in enumerate(selected.channel_names):
        if channel_name in eeg_channels:
            values[:, channel_index] = convert_values(
                values[:, channel_index],
                selected.units[channel_index],
                UV,
            )

    array = values.reshape(n_epochs, epoch_length, selected.n_channels).transpose(
        0, 2, 1
    )
    return selected, samples_to_keep, epoch_length, array


def load_model(version: str = "default") -> object:
    """Load a pre-trained EEG usability model, downloading it on first use.

    Args:
        version: Model version key in :data:`~somnio.tasks.eeg_usability.defaults.MODEL_NAMES`.

    Returns:
        Fitted classifier with a ``predict`` method and ``num_feature()`` accessor.

    Raises:
        ValueError: If *version* is unknown.
        RuntimeError: If model metadata cannot be fetched or the pickle cannot be read.
        MissingOptionalDependency: If optional packages are not installed.
    """
    model_name = MODEL_NAMES.get(version)

    if model_name is None:
        raise ValueError(
            f"Model {version!r} not found in available models. "
            f"Available models: {sorted(MODEL_NAMES)}"
        )

    model_path = get_models_dir() / model_name

    if not model_path.exists():
        logger.info(
            "Model %s not found in %s. Downloading...",
            model_name,
            get_models_dir(),
        )
        _download_model(model_name)
        logger.info("Model %s downloaded to %s", model_name, model_path)

    logger.info("Loading model %s from %s", model_name, model_path)
    try:
        return _read_model_pickle(model_path)
    except (OSError, pickle.UnpicklingError) as exc:
        logger.warning(
            "Failed to load EEG usability model pickle at %s (%s); "
            "removing cache and re-downloading.",
            model_path,
            exc,
        )
        model_path.unlink(missing_ok=True)
        _download_model(model_name)
        try:
            return _read_model_pickle(model_path)
        except (OSError, pickle.UnpicklingError) as retry_exc:
            raise RuntimeError(
                f"Failed to load EEG usability model pickle at {model_path!r}: "
                f"{retry_exc}"
            ) from retry_exc


def get_usability_score(
    ts: TimeSeries,
    model: object,
    eeg: str,
    movement: str,
    *,
    output_channel: str = "usability",
) -> tuple[TimeSeries, int, int]:
    """Predict EEG usability labels for one electrode from 10-second epochs.

    Args:
        ts: Input time-series at 256 Hz containing one EEG channel and a
            movement channel. The EEG channel is converted to microvolts (µV)
            internally before feature extraction.
        model: Classifier returned by :func:`load_model`.
        eeg: Channel name for the EEG electrode.
        movement: Channel name for the movement signal.
        output_channel: Name of the returned score channel.

    Returns:
        ``(scores, samples_to_keep, epoch_length)`` where

        * ``scores`` — :class:`~somnio.data.timeseries.TimeSeries` of integer
          usability labels per 10-second epoch (sample rate 0.1 Hz).
        * ``samples_to_keep`` — number of leading input samples that fit whole
          epochs (trailing partial epoch is excluded).
        * ``epoch_length`` — epoch size in input samples.

    Raises:
        ValueError: If ``ts.sample_rate`` is not 256 Hz, required channels are
            missing, or the recording is shorter than one epoch.
        MissingOptionalDependency: If optional packages are not installed.
    """
    channels = [eeg, movement]
    selected, samples_to_keep, epoch_length, array = _prepare_epoch_array(
        ts,
        channels,
        eeg_channels={eeg},
    )

    sample_rate = selected.sample_rate  # type: ignore[assignment]
    spectrogram_features = _extract_spectrogram_features(array, sample_rate)
    tsfel_features = None
    if model.num_feature() != N_FEATURES_LITE:  # type: ignore[attr-defined]
        tsfel_features = _extract_tsfel_features(array, sample_rate)

    samples = _build_model_samples(
        spectrogram_features,
        model,
        eeg_idx=0,
        movement_idx=1,
        tsfel_features=tsfel_features,
    )
    logger.debug("Samples shape: %s", samples.shape)

    labels = _predict_usability_labels(model, samples)
    n_epochs = labels.shape[0]
    scores_ts = TimeSeries(
        values=labels.reshape(n_epochs, 1),
        timestamps=_epoch_timestamps(selected.timestamps, epoch_length, n_epochs),
        channel_names=[output_channel],
        units=["1"],
        sample_rate=OUTPUT_SAMPLE_RATE_HZ,
    )
    return scores_ts, samples_to_keep, epoch_length


def get_usability_scores(
    ts: TimeSeries,
    model: object,
    eeg_left: str,
    eeg_right: str,
    movement: str,
) -> tuple[TimeSeries, int, int]:
    """Predict left/right EEG usability labels from 10-second epochs.

    This is a convenience wrapper that scores both electrodes in one pass over
    the input channels. For a single EEG electrode, use
    :func:`get_usability_score`.

    Args:
        ts: Input time-series at 256 Hz containing left EEG, right EEG, and
            movement channels. EEG channels are converted to microvolts (µV)
            internally before feature extraction.
        model: Classifier returned by :func:`load_model`.
        eeg_left: Channel name for the left EEG electrode.
        eeg_right: Channel name for the right EEG electrode.
        movement: Channel name for the movement signal.

    Returns:
        ``(scores, samples_to_keep, epoch_length)`` where

        * ``scores`` — :class:`~somnio.data.timeseries.TimeSeries` of integer
          usability labels per 10-second epoch (channels
          ``usability_left`` / ``usability_right``, sample rate 0.1 Hz).
        * ``samples_to_keep`` — number of leading input samples that fit whole
          epochs (trailing partial epoch is excluded).
        * ``epoch_length`` — epoch size in input samples.

    Raises:
        ValueError: If ``ts.sample_rate`` is not 256 Hz, required channels are
            missing, or the recording is shorter than one epoch.
        MissingOptionalDependency: If optional packages are not installed.
    """
    channels = [eeg_left, eeg_right, movement]
    selected, samples_to_keep, epoch_length, array = _prepare_epoch_array(
        ts,
        channels,
        eeg_channels={eeg_left, eeg_right},
    )

    sample_rate = selected.sample_rate  # type: ignore[assignment]
    spectrogram_features = _extract_spectrogram_features(array, sample_rate)
    tsfel_features = None
    if model.num_feature() != N_FEATURES_LITE:  # type: ignore[attr-defined]
        tsfel_features = _extract_tsfel_features(array, sample_rate)

    movement_idx = channels.index(movement)
    samples_left = _build_model_samples(
        spectrogram_features,
        model,
        eeg_idx=0,
        movement_idx=movement_idx,
        tsfel_features=tsfel_features,
    )
    samples_right = _build_model_samples(
        spectrogram_features,
        model,
        eeg_idx=1,
        movement_idx=movement_idx,
        tsfel_features=tsfel_features,
    )
    logger.debug("Samples left shape: %s", samples_left.shape)
    logger.debug("Samples right shape: %s", samples_right.shape)

    usability_scores = np.column_stack(
        (
            _predict_usability_labels(model, samples_left),
            _predict_usability_labels(model, samples_right),
        )
    )
    logger.debug("Usability scores shape: %s", usability_scores.shape)

    n_epochs = usability_scores.shape[0]
    scores_ts = TimeSeries(
        values=usability_scores,
        timestamps=_epoch_timestamps(selected.timestamps, epoch_length, n_epochs),
        channel_names=list(OUTPUT_CHANNEL_NAMES),
        units=list(OUTPUT_UNITS),
        sample_rate=OUTPUT_SAMPLE_RATE_HZ,
    )

    return scores_ts, samples_to_keep, epoch_length
