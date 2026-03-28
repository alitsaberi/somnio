"""ZMax-style directory of per-channel EDF files (optional ``edf`` extra)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from somnio.data.adapters.mne import from_mne_raw, import_mne, to_mne_raw
from somnio.data.timeseries import TimeSeries

from somnio.io.edf.utils import (
    ensure_export_edf_deps,
    require_edf_compatible_timestamps,
)


def _discover_edf_stems(root: Path) -> list[str]:
    """Sorted stems of ``*.edf`` in ``root`` (deterministic column order)."""
    paths = sorted(root.glob("*.edf"))
    return [p.stem for p in paths]


def _read_raw_edf(
    path: Path,
    *,
    preload: bool,
    verbose: str | bool | None,
    units: dict[str, str] | str | None,
) -> Any:
    from mne.io import read_raw_edf

    kwargs: dict[str, Any] = {"preload": preload, "verbose": verbose}
    if units is not None:
        kwargs["units"] = units
    return read_raw_edf(path, **kwargs)


def read(
    path: Path | str,
    *,
    stems: Sequence[str] | None = None,
    stem_aliases: Mapping[str, str] | None = None,
    preload: bool = True,
    verbose: str | bool | None = "ERROR",
    units: dict[str, str] | str | None = None,
) -> TimeSeries:
    """Load per-channel EDF files from a directory into one :class:`~somnio.data.timeseries.TimeSeries`.

    Discovers ``*.edf`` files under ``path`` (sorted by stem) when ``stems`` is
    omitted. When ``stems`` is set, loads ``{stem}.edf`` for each entry in that
    order; every file must exist.

    By default the :class:`~somnio.data.timeseries.TimeSeries` channel name for each
    file is the filename stem (verbatim). Use ``stem_aliases`` to rename: keys are
    stems as on disk, values are the channel names to use.

    All selected files must share the same sample rate, length, and timestamp grid.

    Args:
        path: Directory containing per-channel ``.edf`` files.
        stems: Filename stems without ``.edf``, in load order; ``None`` loads every
            ``*.edf`` in the directory (sorted lexicographically by path).
        stem_aliases: Optional map ``disk stem -> channel name`` for columns that
            should not use the stem verbatim.
        preload: Passed through to MNE.
        verbose: MNE verbosity.
        units: Optional channel units hint forwarded to :func:`mne.io.read_raw_edf`.
    """
    import_mne()
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(root)

    if stems is None:
        stem_list = _discover_edf_stems(root)
        used_explicit_stems = False
    else:
        stem_list = [str(s).strip() for s in stems]
        used_explicit_stems = True

    aliases = dict(stem_aliases) if stem_aliases is not None else {}

    slices: list[TimeSeries] = []
    for stem in stem_list:
        fp = root / f"{stem}.edf"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing EDF for stem {stem!r}: {fp}")

        raw = _read_raw_edf(fp, preload=preload, verbose=verbose, units=units)
        ts = from_mne_raw(raw)
        if ts.n_channels != 1:
            raise ValueError(f"{fp} expected exactly one channel, got {ts.n_channels}")
        name = aliases.get(stem, stem)
        ts = TimeSeries(
            values=ts.values,
            timestamps=ts.timestamps,
            channel_names=(name,),
            units=ts.units,
            sample_rate=ts.sample_rate,
        )
        slices.append(ts)

    if not slices:
        if used_explicit_stems:
            raise FileNotFoundError(f"No stems to load (empty ``stems``) under {root}")
        raise FileNotFoundError(f"No *.edf files under {root}")

    ref = slices[0]
    for ts in slices[1:]:
        if ts.sample_rate != ref.sample_rate:
            raise ValueError(
                f"Sample rate mismatch: {ref.sample_rate} vs {ts.sample_rate}"
            )
        if ts.n_samples != ref.n_samples:
            raise ValueError(f"Length mismatch: {ref.n_samples} vs {ts.n_samples}")
        if not np.array_equal(ts.timestamps, ref.timestamps):
            raise ValueError("Timestamp grid mismatch between per-channel EDF files")

    values = np.column_stack([ts.values[:, 0] for ts in slices])
    channel_names = tuple(ts.channel_names[0] for ts in slices)
    units = tuple(ts.units[0] for ts in slices)
    return TimeSeries(
        values=values,
        timestamps=ref.timestamps.copy(),
        channel_names=channel_names,
        units=units,
        sample_rate=ref.sample_rate,
    )


def write(
    path: Path | str,
    data: TimeSeries,
    *,
    channel_to_stem: Mapping[str, str] | None = None,
    overwrite: bool = False,
    verbose: str | bool | None = None,
) -> None:
    """Write one single-channel EDF per :class:`~somnio.data.timeseries.TimeSeries` column.

    Each file is named ``{stem}.edf`` where ``stem`` defaults to the channel name
    (verbatim). Use ``channel_to_stem`` to override: keys are ``TimeSeries``
    channel names, values are filename stems (no ``.edf``).
    """
    ensure_export_edf_deps()
    require_edf_compatible_timestamps(data)
    mne = import_mne()
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)

    stem_map = dict(channel_to_stem) if channel_to_stem is not None else {}

    for name in data.channel_names:
        stem = stem_map.get(name, name)
        fp = root / f"{stem}.edf"
        if fp.exists() and not overwrite:
            raise FileExistsError(f"{fp} exists (pass overwrite=True)")
        sub = data.select_channels([name])
        raw = to_mne_raw(sub)
        mne.export.export_raw(fp, raw, overwrite=True, verbose=verbose)


class ZMaxMultiEDFReader:
    """Stateless reader for a directory of per-channel EDF files (ZMax-style layout)."""

    def read(self, path: Path, **kwargs: Any) -> TimeSeries:
        """Read from directory ``path``; optional ``stems``, ``stem_aliases``, etc."""
        return read(path, **kwargs)


class ZMaxMultiEDFWriter:
    """Stateless writer for a directory of per-channel EDF files (ZMax-style layout)."""

    def write(self, path: Path, data: TimeSeries, **kwargs: Any) -> None:
        """Write ``data`` into ``path``; supports ``overwrite``, ``verbose``, ``channel_to_stem``."""
        overwrite = bool(kwargs.get("overwrite", False))
        verbose = kwargs.get("verbose", None)
        channel_to_stem = kwargs.get("channel_to_stem", None)
        write(
            path,
            data,
            channel_to_stem=channel_to_stem,
            overwrite=overwrite,
            verbose=verbose,
        )
