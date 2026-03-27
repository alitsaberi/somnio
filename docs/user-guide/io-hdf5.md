# HDF5 I/O

`zutils.io.hdf5` reads and writes `Sample` and `TimeSeries` objects to HDF5 files.
It owns all serialization logic — data types themselves carry no format knowledge.

## Installation

HDF5 support requires the optional `hdf5` extra:

```bash
uv add zutils[hdf5]
# or
pip install zutils[hdf5]
```

## High-level API

For most use cases `write_hdf5` and `read_hdf5` are all you need.

### Writing

```python
from pathlib import Path
from zutils.data import TimeSeries
from zutils.io.hdf5 import write_hdf5
import numpy as np

eeg = TimeSeries(
    values=np.random.randn(256, 2),
    timestamps=np.arange(256, dtype=np.int64) * int(1e9 / 256),
    channel_names=["EEG_L", "EEG_R"],
    units=["V", "V"],
    sample_rate=256.0,
)

write_hdf5(Path("recording.h5"), {"eeg": eeg})
```

Each key in the dict becomes a top-level group in the file.
**An existing file is overwritten.**

### Reading

```python
from zutils.io.hdf5 import read_hdf5

data = read_hdf5(Path("recording.h5"))
eeg = data["eeg"]          # TimeSeries
print(eeg.n_samples)       # 256
print(eeg.sample_rate)     # 256.0
```

`read_hdf5` returns a `dict[str, TimeSeries]`.  `Sample` objects written via
`write_hdf5` are restored as single-row `TimeSeries` (shape `(1, n_channels)`).

## Storage layout

Each group stores two datasets and three attributes:

```
<group_name>/
    values      float64  (n_samples, n_channels)
    timestamps  int64    (n_samples,)

<group_name>.attrs:
    channel_names   list[str]
    units           list[str]
    sample_rate     float          (absent when None / irregular)
```

Values are stored in SI base units exactly — no scaling on read or write.

## `serialize` / `deserialize`

Use these when you need to integrate with custom HDF5 layouts.

```python
from zutils.io.hdf5 import serialize, deserialize

datasets, attrs = serialize(ts)
# datasets: {"values": ndarray, "timestamps": ndarray}
# attrs:    {"channel_names": [...], "units": [...], "sample_rate": 256.0}

restored = deserialize(datasets, attrs)
```

## Low-level: `HDF5Manager`

`HDF5Manager` provides direct control over HDF5 file operations with
retry/backoff for file-locking issues (common on Windows network drives).

```python
from zutils.io.hdf5 import HDF5Manager
import numpy as np

path = Path("stream.h5")

with HDF5Manager(path, compression="gzip") as mgr:
    mgr.create_group("eeg", channel_names=["EEG_L", "EEG_R"], units=["V", "V"])
    mgr.create_dataset(
        "eeg", "values",
        data=np.zeros((0, 2), dtype=np.float64),
        max_shape=(None, 2),
    )
    mgr.create_dataset(
        "eeg", "timestamps",
        data=np.zeros(0, dtype=np.int64),
        max_shape=(None,),
    )

# Stream: append each incoming chunk
with HDF5Manager(path) as mgr:
    mgr.append("eeg", "values", chunk_values)       # (n, 2) float64
    mgr.append("eeg", "timestamps", chunk_stamps)   # (n,)   int64
```

### Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `file_path` | — | Path to HDF5 file (created if absent) |
| `compression` | `"gzip"` | Compression filter for new datasets |
| `max_retries` | `5` | Open attempts before raising `OSError` |
| `retry_delay` | `0.1` | Initial retry wait in seconds (doubles each attempt) |
