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
