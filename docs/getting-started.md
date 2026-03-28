# Getting started

## Installation

ZUtils requires **Python 3.10+**.

### With uv

```bash
uv add zutils
```

To use the [CLI](user-guide/cli.md), install with the optional `cli` extra:

```bash
uv add zutils[cli]
```

### With pip

```bash
pip install zutils
```

For CLI support:

```bash
pip install zutils[cli]
```

For HDF5 signal layouts (`zutils.io.hdf5`, native and USleep formats):

```bash
uv add zutils[hdf5]
# or
pip install zutils[hdf5]
```

### Verify installation

If you installed the CLI extra, confirm it works:

```bash
zutils --help
```

You should see the list of commands (e.g. `download-nsrr`).
