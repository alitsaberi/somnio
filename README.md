# Somnio

Python library and CLI for sleep and physiological data workflows.

## Install

```bash
pip install somnio
```

Optional extras:

- `somnio[cli]` — command-line tools
- `somnio[hdf5]` — `somnio.io.hdf5` layouts (requires [h5py](https://www.h5py.org/))

With **uv**: `uv add somnio`, `uv add somnio[cli]`, or `uv add somnio[hdf5]`.

## Documentation

Site source lives under [`docs/`](docs/). Build or serve locally (with the `docs` dependency group installed):

```bash
uv sync --group docs
uv run mkdocs serve
```

Reference material includes [data types](docs/reference/data.md) and [I/O / HDF5](docs/reference/io.md).
