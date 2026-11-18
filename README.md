# Somnio

Python library and CLI for sleep and physiological data workflows.

## Install

```bash
pip install somnio
```

Optional extras (install as `somnio[extra]` or `somnio[a,b]`):

- `somnio[cli]` — command-line tools
- `somnio[nsrr]` — NSRR download helpers
- `somnio[schemas]` — YAML schemas
- `somnio[signal]` — SciPy-based signal processing (`resample`, etc.)
- `somnio[mne]` — MNE-based processing (`apply_filter` via `mne.filter.filter_data`, adapters)
- `somnio[eeg-usability]` — EEG usability scoring
- `somnio[edf]` — EDF I/O
- `somnio[hdf5]` — HDF5 layouts
- `somnio[onnx]` — ONNX Runtime sleep scoring 
- `somnio[tensorflow]` — TensorFlow sleep scoring

With **uv**: `uv add somnio`, `uv add somnio --extra cli`, or `uv add somnio --extra hdf5`.

## Documentation

[Documentation](https://alitsaberi.github.io/somnio/)

## Contributing

See [Contributing](https://github.com/alitsaberi/somnio/blob/master/docs/contributing.md).

## Changelog

See [CHANGELOG.md](https://github.com/alitsaberi/somnio/blob/master/CHANGELOG.md).

## Credits

The [`somnio.tasks.eeg_usability`](https://alitsaberi.github.io/somnio/user-guide/eeg-usability/) module implements the eegUsability pipeline from [eegFloss](https://github.com/Niloy333/eegFloss) (MIT License, © 2025 Niloy Sikder). See [NOTICE](NOTICE) for the full third-party notice and citation details.

## License

[MIT](https://github.com/alitsaberi/somnio/blob/master/LICENSE)
