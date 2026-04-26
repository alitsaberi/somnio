# Getting started

## Installation

Somnio requires **Python 3.10+**.

### With uv

```bash
uv add somnio
```

To use the [CLI](user-guide/cli.md), install with the optional `cli` extra:

```bash
uv add somnio --extra cli
```

The `download-nsrr` command also needs the `nsrr` extra:

```bash
uv add somnio --extra cli --extra nsrr
```

### Extras matrix

Somnio ships small optional extras so you can install only what you use:

- `cli`: command-line interface (`typer`, `loguru`)
- `nsrr`: NSRR download helpers (use with `cli`)
- `schemas`: pipeline YAML/JSON schemas (`pydantic`, `pyyaml`)
- `signal`: signal processing utilities used by detectors (SciPy)
- `mne`: MNE-based processing utilities
- `edf`: EDF I/O (`edfio`, plus `mne`)
- `hdf5`: HDF5 layouts (`h5py`)

### With pip

```bash
pip install somnio
```

For CLI support:

```bash
pip install somnio[cli]
```

For `download-nsrr`:

```bash
pip install 'somnio[cli,nsrr]'
```

For signal-processing detectors that require SciPy:

```bash
pip install somnio[signal]
```

For HDF5 signal layouts (`somnio.io.hdf5`, native and USleep formats):

```bash
uv add somnio --extra hdf5
# or
pip install somnio[hdf5]
```

### Verify installation

If you installed the CLI extra, confirm it works:

```bash
somnio --help
```

You should see the list of commands (e.g. `download-nsrr`).
