# I/O reference (`somnio.io`)

Signal formats are organized under `somnio.io` using small **layout modules**. Optional backends (**HDF5**, **EDF**) require extra dependencies.

---

## Optional dependency: HDF5

Install **h5py** via the project extra:

```bash
uv add somnio[hdf5]
# or
pip install somnio[hdf5]
```

Import layout helpers from the package (not the top-level `somnio` namespace):

```python
from somnio.io.hdf5 import (
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

Protocols `SignalReader`, `SignalWriter`, and annotation I/O protocols live in `somnio.io.base`.

---

## Native somnio HDF5 (`somnio.io.hdf5.native`)

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

## USleep / zmax-datasets HDF5 (`somnio.io.hdf5.usleep`)

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

## Optional dependency: EDF

Install **MNE** and **edfio** via the project extra:

```bash
uv add somnio[edf]
# or
pip install somnio[edf]
```

Reading uses MNE; writing multiplexed or per-channel EDF uses MNE’s exporter (`edfio` for `.edf`). Conversion between MNE `Raw` and `TimeSeries` is centralized in [`somnio.data.adapters.mne`](data.md#mne-adapter-optional).

Import layout entry points from `somnio.io.edf`:

```python
from somnio.io.edf import (
    read_standard,
    write_standard,
    read_zmax_multi,
    write_zmax_multi,
    StandardEDFReader,
    StandardEDFWriter,
    ZMaxMultiEDFReader,
    ZMaxMultiEDFWriter,
)
```

### EDF export constraints

- **`TimeSeries.sample_rate`** must be set for any EDF write handled here.
- The **first sample’s wall time** must map to a calendar year **1985–2084** (EDF header limit enforced by edfio). Otherwise you get a clear `ValueError` suggesting another format (e.g. HDF5) or shifting timestamps.
- MNE may **append** samples at the end of a file to satisfy EDF block length; compare only the intended prefix when testing round-trips with short signals.

---

## Standard multiplexed EDF (`somnio.io.edf.standard`)

**Use case:** one file containing multiple channels (classic EDF/BDF as loaded by MNE).

| Entry | Role |
|-------|------|
| `read_standard(path, preload=..., verbose=..., units=...)` | Load file → `TimeSeries` via `mne.io.read_raw_edf` and `from_mne_raw` |
| `write_standard(path, data, overwrite=..., verbose=...)` | `TimeSeries` → EDF via `to_mne_raw` and `mne.export.export_raw` |

Channel labels from the file are normalized for somnio (**spaces → underscores**) inside the MNE adapter. Physical units follow MNE (e.g. EEG in **volts**). Optional `units=` is forwarded to `read_raw_edf` when channel units are missing in the file.

**Protocol wrappers:** `StandardEDFReader`, `StandardEDFWriter`.

---

## Per-channel directory EDF — ZMax-style (`somnio.io.edf.zmax`)

**Use case:** a **directory** of single-channel EDF files, e.g. `EEG L.edf`, `dX.edf`, one channel per file (common for some wearable / export pipelines).

| Entry | Role |
|-------|------|
| `read_zmax_multi(path, stems=..., stem_aliases=..., ...)` | Merge selected files under `path` into one `TimeSeries` |
| `write_zmax_multi(path, data, channel_to_stem=..., overwrite=..., ...)` | One `{stem}.edf` per column |

**Discovery:** with `stems=None`, every `*.edf` in the directory is loaded; stems are taken in **lexicographic sort order** of paths for stable column order.

**Explicit list:** `stems=["EEG L", "dX"]` loads those files **in that order**; each must exist.

**Verbatim naming:** by default, each **channel name** in `TimeSeries` equals the filename **stem** (no automatic space/underscore rewriting). Optional maps:

- **`stem_aliases`**: `{disk_stem: channel_name}` on read when on-disk names should not become column names verbatim.
- **`channel_to_stem`**: `{channel_name: disk_stem}` on write when filenames should differ from `channel_names`.

All merged files must share the same sample rate, length, and timestamp grid.

**Protocol wrappers:** `ZMaxMultiEDFReader`, `ZMaxMultiEDFWriter` (pass `stems`, `stem_aliases`, `channel_to_stem`, etc. through `**kwargs` where applicable).

---

## See also

- [Data reference](data.md) — `TimeSeries`, `Sample`, `sample_rate` semantics; MNE adapter
- [User guide: data types](../user-guide/data.md)
