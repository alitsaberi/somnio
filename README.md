# ZUtils

Python library and CLI for sleep and physiological data workflows.

## Install

```bash
pip install zutils
```

Optional extras:

- `zutils[cli]` — command-line tools
- `zutils[hdf5]` — `zutils.io.hdf5` layouts (requires [h5py](https://www.h5py.org/))

With **uv**: `uv add zutils`, `uv add zutils[cli]`, or `uv add zutils[hdf5]`.

## Documentation

Site source lives under [`docs/`](docs/). Build or serve locally (with the `docs` dependency group installed):

```bash
uv sync --group docs
uv run mkdocs serve
```

Reference material includes [data types](docs/reference/data.md) and [I/O / HDF5](docs/reference/io.md).
