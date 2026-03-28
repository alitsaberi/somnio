# I/O reference (`zutils.io`)

Signal formats are organized under `zutils.io` using small **layout modules**. Optional backends (HDF5 today) require extra dependencies.

---

## Optional dependency: HDF5

Install **h5py** via the project extra:

```bash
uv add zutils[hdf5]
# or
pip install zutils[hdf5]
```

Import layout helpers from the package (not the top-level `zutils` namespace):

```python
from zutils.io.hdf5 import (
    read,
    read_all,
    write,
    deserialize,
    serialize,
    usleep_read,
    usleep_write,
    align_timestamps_to_usleep_grid,
)
```

Protocols `SignalReader`, `SignalWriter`, and annotation I/O protocols live in `zutils.io.base`.

---

## Native zutils HDF5 (`zutils.io.hdf5.native`)

**Use case:** one HDF5 file with one or more groups, each holding a full `TimeSeries` (values + per-sample timestamps).

### Layout

For group `{name}`:

| Path | Kind | Description |
|------|------|-------------|
| `/{name}/data` | Dataset | `float64`, shape `(n_samples, n_channels)` |
| `/{name}/timestamp` | Dataset | `int64`, shape `(n_samples,)` |
| `/{name}` attrs | Attributes | `channel_names`, `units` (string arrays); optional `sample_rate` |

### Module functions

| Function | Purpose |
|----------|---------|
| `read(path, group=None)` | Load one group; if `group` is omitted, the file must contain exactly one native-layout top-level group |
| `read_all(path)` | `dict[str, TimeSeries]` for every top-level native group |
| `write(path, data, group_name, append=False)` | Write one group; `append=True` opens the file in append mode |
| `serialize(ts)` / `deserialize(datasets, attrs)` | Encode/decode for streaming append (e.g. Slumber `HDF5Manager` building datasets incrementally) |

### Protocol wrappers

- `NativeHDF5Reader().read(path, **kwargs)` — pass `group=...` in `kwargs` if needed
- `NativeHDF5Writer().write(path, data, **kwargs)` — requires `group_name=...`; optional `append=`

---

## USleep / zmax-datasets HDF5 (`zutils.io.hdf5.usleep`)

**Use case:** interchange with USleep-style exports: one sample rate for the whole file, no per-sample timestamps stored.

### Layout

| Location | Content |
|----------|---------|
| File attrs | `sample_rate` (Hz, required); optional `start_timestamp_ns` (first sample time, ns since Unix epoch) |
| `/channels/{channel_name}` | 1-D `float64` samples per channel |
| Per-channel attrs | `channel_index` (int, column order, zmax-datasets style); optional `unit` |

On **read**, timestamps are reconstructed as:

```text
t[i] = start_timestamp_ns + i * round(1e9 / sample_rate)
```

### Writing requirements

- `TimeSeries.sample_rate` must be set.
- `TimeSeries.timestamps` must match that grid exactly (lossless round-trip). If you only have small jitter, call **`align_timestamps_to_usleep_grid(ts)`** before `usleep_write`.

Per-channel **`sample_rate`** on legacy or third-party files is **ignored**; only the file-level attribute is used.

### Channel order

- If **every** channel dataset has **`channel_index`**, columns follow those indices (same idea as zmax-datasets).
- If **none** have `channel_index`, channel names are sorted lexicographically (backward compatibility).

### Module functions

| Function | Purpose |
|----------|---------|
| `read(path)` | Build a `TimeSeries` from the file |
| `write(path, data)` | Overwrite `path` with one USleep-style file |
| `align_timestamps_to_usleep_grid(data)` | Copy of `data` with canonical timestamps for the current `sample_rate` |

### Protocol wrappers

`USleepHDF5Reader`, `USleepHDF5Writer` implement the same `read` / `write` signatures as other layout modules.

---

## See also

- [Data reference](data.md) — `TimeSeries`, `Sample`, `sample_rate` semantics
- [User guide: data types](../user-guide/data.md)
