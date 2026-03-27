# HDF5 Reference

API reference for `zutils.io.hdf5`.

Requires the `hdf5` extra (`pip install zutils[hdf5]`).

---

## `HDF5Format`

```python
class HDF5Format:
    def __init__(self, compression: str | None = "gzip") -> None: ...
```

Default zutils HDF5 layout: one top-level group per named signal.
Implements [`FileFormat`](io-base.md).

Subclass and override `write` / `read` to implement an alternative HDF5
layout (e.g. uSleep-style `channels/` group with a global `sample_rate`
attribute) while keeping the same interface.

**Args:**

| Parameter | Default | Description |
|---|---|---|
| `compression` | `"gzip"` | HDF5 compression filter. `None` disables compression. |

### `write(path, data)`

```python
def write(
    self,
    path: Path,
    data: dict[str, Sample | TimeSeries],
) -> None
```

Write named data objects to an HDF5 file. Overwrites any existing file.

### `read(path)`

```python
def read(self, path: Path) -> dict[str, TimeSeries]
```

Load all top-level groups from an HDF5 file as `TimeSeries` objects.
`Sample` objects written via `write` are restored as single-row `TimeSeries`.

---

## `serialize`

```python
def serialize(
    obj: Sample | TimeSeries,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]
```

Convert a data object to `(datasets_dict, attributes_dict)` for HDF5 storage.

| Key | Type | Shape / notes |
|---|---|---|
| `datasets["values"]` | `float64 ndarray` | `(n_samples, n_channels)`; `(1, n_channels)` for `Sample` |
| `datasets["timestamps"]` | `int64 ndarray` | `(n_samples,)`; `(1,)` for `Sample` |
| `attrs["channel_names"]` | `list[str]` | |
| `attrs["units"]` | `list[str]` | |
| `attrs["sample_rate"]` | `float` | Present only when not `None` |

Values are stored in SI base units without scaling.

---

## `deserialize`

```python
def deserialize(datasets: dict[str, Any], attrs: dict[str, Any]) -> TimeSeries
```

Reconstruct a `TimeSeries` from an HDF5 group's datasets and attributes.
`datasets` values may be `h5py.Dataset` objects or numpy arrays.
`sample_rate` is `None` when absent from `attrs`.

---

## `write_hdf5`

```python
def write_hdf5(path: Path, data: dict[str, Sample | TimeSeries]) -> None
```

Convenience wrapper for `HDF5Format().write(path, data)`.
Overwrites any existing file at `path`.

---

## `read_hdf5`

```python
def read_hdf5(path: Path) -> dict[str, TimeSeries]
```

Convenience wrapper for `HDF5Format().read(path)`.
