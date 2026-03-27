# HDF5 Reference

API reference for `zutils.io.hdf5`.

Requires the `hdf5` extra (`pip install zutils[hdf5]`).

---

## Exceptions

### `GroupDoesNotExistError`

Raised when a requested HDF5 group is not present in the file.
Subclass of `ValueError`.

### `DatasetDoesNotExistError`

Raised when a requested HDF5 dataset is not present in a group.
Subclass of `ValueError`.

---

## `HDF5Manager`

```python
class HDF5Manager:
    def __init__(
        self,
        file_path: Path,
        compression: str = "gzip",
        max_retries: int = 5,
        retry_delay: float = 0.1,
    ) -> None: ...
```

Low-level HDF5 file manager. Opens the file in append mode (`"a"`), creating
it if absent.  Implements the context-manager protocol — prefer `with HDF5Manager(...)`.

### Properties

| Property | Type | Description |
|---|---|---|
| `groups` | `list[str]` | Top-level group names in the file |

### Methods

#### `close()`

```python
def close(self) -> None
```

Close the underlying HDF5 file. Safe to call multiple times.

#### `create_group(group_name, **attributes)`

```python
def create_group(self, group_name: str, **attributes: Any) -> h5py.Group
```

Create a new top-level group and attach keyword arguments as HDF5 attributes.

#### `create_dataset(group_name, dataset_name, ...)`

```python
def create_dataset(
    self,
    group_name: str,
    dataset_name: str,
    data: np.ndarray | None = None,
    shape: tuple[int, ...] | None = None,
    dtype: Any = None,
    max_shape: tuple[int | None, ...] | None = None,
    **attributes: Any,
) -> h5py.Dataset
```

Create a dataset inside an existing group.  Either `data` or `shape` must be
provided.  `max_shape` defaults to all-`None` (unlimited) to allow later
`append` calls.  Uses `self.compression` for the new dataset.

Raises `GroupDoesNotExistError` if `group_name` is absent.

#### `get_dataset(group_name, dataset_name)`

```python
def get_dataset(self, group_name: str, dataset_name: str) -> h5py.Dataset
```

Return a dataset by name.

Raises `GroupDoesNotExistError` or `DatasetDoesNotExistError` if absent.

#### `append(group_name, dataset_name, data)`

```python
def append(self, group_name: str, dataset_name: str, data: np.ndarray) -> None
```

Extend a resizable dataset along axis 0.  The dataset must have been created
with `max_shape=(None, ...)`.  For 2-D datasets the column count must match.

---

## `serialize`

```python
def serialize(
    obj: Sample | TimeSeries,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]
```

Convert a data object to `(datasets_dict, attributes_dict)` for HDF5 storage.

| Key | Type | Shape |
|---|---|---|
| `datasets["values"]` | `float64 ndarray` | `(n_samples, n_channels)` |
| `datasets["timestamps"]` | `int64 ndarray` | `(n_samples,)` |
| `attrs["channel_names"]` | `list[str]` | — |
| `attrs["units"]` | `list[str]` | — |
| `attrs["sample_rate"]` | `float` | present only when not `None` |

A `Sample` is serialized as a single-row TimeSeries: `values` shape `(1, n_channels)`,
`timestamps` shape `(1,)`.  Values are stored in SI base units without scaling.

---

## `deserialize`

```python
def deserialize(datasets: dict[str, Any], attrs: dict[str, Any]) -> TimeSeries
```

Reconstruct a `TimeSeries` from HDF5 group data.  `datasets` values may be
`h5py.Dataset` objects or numpy arrays.  `sample_rate` is `None` when the key
is absent from `attrs`.

---

## `write_hdf5`

```python
def write_hdf5(path: Path, data: dict[str, Sample | TimeSeries]) -> None
```

Write a mapping of named data objects to an HDF5 file.
**Overwrites any existing file at `path`.**

---

## `read_hdf5`

```python
def read_hdf5(path: Path) -> dict[str, TimeSeries]
```

Load all top-level groups from an HDF5 file as `TimeSeries` objects.
`Sample` objects written via `write_hdf5` are restored as single-row `TimeSeries`.
