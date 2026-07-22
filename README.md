# Somnio

Python library and CLI for sleep and physiological data workflows.

## Install

```bash
pip install somnio
```

Optional extras (install as `somnio[extra]` or `somnio[a,b]`):

- `somnio[cli]` — command-line tools
- `somnio[nsrr]` — NSRR download helpers (pair with `cli` for the `download-nsrr` command)
- `somnio[schemas]` — YAML schemas (`pydantic`, `pyyaml`)
- `somnio[signal]` — SciPy-based signal processing utilities
- `somnio[eeg-usability]` — EEG usability scoring (`somnio.tasks.eeg_usability`; `requests`, `scipy`, `tsfel`, `joblib`, `platformdirs`)
- `somnio[mne]` — MNE-based processing utilities
- `somnio[edf]` — EDF I/O (`edfio`, plus `mne`)
- `somnio[hdf5]` — `somnio.io.hdf5` layouts (requires [h5py](https://www.h5py.org/))

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
