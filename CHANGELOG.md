# Changelog

All notable changes to this project are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (`MAJOR.MINOR.PATCH`, with optional pre-release suffixes such as `1.0.0b1`).

## [Unreleased]

## [0.2.0b1] - 2026-06-02

Beta pre-release after 0.1.0. New pipeline, transforms, and task modules; annotation time bases and extras layout changed. Documentation and test coverage for sleep scoring are still incomplete.

### Added

- **`somnio.pipeline`**: YAML/JSON-driven execution engine with serial and parallel steps, bundle I/O, transform chains, and validation (optional extra `schemas`: `pydantic`, `pyyaml`).
- **`somnio.transforms`**: `clip`, `filter`, `resample`, and `scale` helpers (optional extra `signal`: SciPy).
- **`somnio.tasks.eye_movement`**: Eye-movement detection and event types (optional extra `signal`).
- **`somnio.tasks.sleep_scoring`**: ONNX-based sleep scoring, windowing, and model metadata (optional extra `onnx`: `onnxruntime`).
- **`somnio.data.units`**: Unit conversion utilities for signal workflows.
- **Optional extras**: `signal`, `mne`, `nsrr`, `schemas`, and `onnx`; `nsrr` dependencies moved out of `cli` (install `somnio[cli,nsrr]` for `download-nsrr`).

### Changed

- **`Event` / `Epochs`**: Onset, duration, and epoch period use **int nanoseconds** (Unix epoch) instead of seconds as `float`. `Event.description` is replaced by **`type`** and optional **`label`**.
- **`somnio.devices.zmax`**: Stimulation parameters use seconds; protocol and client refactored.
- **Tests**: Pipeline, transforms, eye-movement, and sleep-scoring windowing coverage added.

### Documentation

- Getting started extras matrix updated; pipeline execution notes expanded in-repo (full user-guide coverage for new modules is still pending).

## [0.1.0] - 2026-03-28

First public release on PyPI (`somnio` 0.1.0, Python ≥ 3.10). Core dependency: NumPy. Optional extras: `hdf5`, `cli`, `edf`.

### Added

- **`somnio.data`**: In-memory types for biosignal workflows—`TimeSeries` and `Sample`, annotation helpers (`Event`, `Epochs`, epoch ↔ event conversions), and `concat` / `collect_samples` utilities.
- **`somnio.io`**: Abstract `SignalReader` / `SignalWriter` and `AnnotationReader` / `AnnotationWriter` APIs.
- **`somnio.io.hdf5`** (extra `hdf5`): Native HDF5 read/write helpers plus USleep-oriented layouts (`USleepHDF5Reader` / `USleepHDF5Writer`, timestamp alignment).
- **`somnio.io.edf`** (extra `edf`): Standard EDF readers/writers and ZMax multi-EDF readers/writers (via MNE + edfio).
- **`somnio.devices.zmax`**: TCP client for the ZMax headband live stream (Hypnodyne / ZMax USB server), including connection helpers, sampling constants, and LED / stimulation enums.
- **CLI** (extra `cli`): `somnio` Typer entry point with `download-nsrr` and optional file logging (`-l` / `-f`).
- **Packaging**: `pyproject.toml` metadata (README, license, keywords, classifiers, project URLs), `uv_build` backend, and optional dependency groups for docs, lint, and tests.
- **Automation**: GitHub Actions workflow to build, test, and publish to PyPI with **Trusted Publishing** (OIDC) on version tags `v*`, with tag/version consistency checks and wheel/sdist smoke tests.

### Documentation

- MkDocs (Material) site, including user guide, reference material, and CLI documentation; published at the URL in `project.urls` → Documentation.

[Unreleased]: https://github.com/alitsaberi/somnio/compare/v0.2.0b1...HEAD
[0.2.0b1]: https://github.com/alitsaberi/somnio/compare/v0.1.0...v0.2.0b1
[0.1.0]: https://github.com/alitsaberi/somnio/releases/tag/v0.1.0
