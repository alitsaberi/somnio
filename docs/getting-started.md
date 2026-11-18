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

- `cli`: command-line interface
- `nsrr`: NSRR download helpers
- `schemas`: pipeline YAML schemas
- `signal`: SciPy-based signal processing
- `mne`: MNE-based processing
- `edf`: EDF I/O
- `hdf5`: HDF5 layouts
- `onnx`: ONNX Runtime sleep scoring backends; see [Sleep scoring](user-guide/sleep-scoring.md)
- `tensorflow`: TensorFlow sleep scoring backends; see [Sleep scoring](user-guide/sleep-scoring.md)
- `eeg-usability`: EEG usability scoring from eegFloss *lite* models; see [EEG usability](user-guide/eeg-usability.md)

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

For SciPy-based signal processing (`resample`, and other SciPy transforms):

```bash
pip install somnio[signal]
```

For MNE-based filtering (`apply_filter`) and the MNE ↔ TimeSeries adapter:

```bash
pip install somnio[mne]
```

Eye-movement detection with default preprocessing needs both:

```bash
pip install 'somnio[signal,mne]'
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
