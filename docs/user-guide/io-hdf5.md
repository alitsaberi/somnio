# HDF5 I/O

`zutils.io.hdf5` reads and writes `Sample` and `TimeSeries` objects to HDF5 files.
All serialization logic lives here — data types carry no format knowledge.

## Installation

HDF5 support requires the optional `hdf5` extra:

```bash
uv add zutils[hdf5]
# or
pip install zutils[hdf5]
```

## High-level API

`write_hdf5` and `read_hdf5` cover most use cases.

```python
from pathlib import Path
import numpy as np
from zutils.data import TimeSeries
from zutils.io.hdf5 import write_hdf5, read_hdf5

eeg = TimeSeries(
    values=np.random.randn(256, 2),
    timestamps=np.arange(256, dtype=np.int64) * int(1e9 / 256),
    channel_names=["EEG_L", "EEG_R"],
    units=["V", "V"],
    sample_rate=256.0,
)

write_hdf5(Path("recording.h5"), {"eeg": eeg})   # overwrites if exists

data = read_hdf5(Path("recording.h5"))
print(data["eeg"].n_samples)    # 256
print(data["eeg"].sample_rate)  # 256.0
```

Each key in the dict becomes a top-level group. `read_hdf5` returns
`dict[str, TimeSeries]`. `Sample` objects are restored as single-row
`TimeSeries` (shape `(1, n_channels)`).

## Storage layout

```
<group_name>/
    values      float64  (n_samples, n_channels)
    timestamps  int64    (n_samples,)

<group_name>.attrs:
    channel_names   list[str]
    units           list[str]
    sample_rate     float          (absent when None / irregular)
```

Values are stored in SI base units — no scaling on read or write.

## `HDF5Format` — format instance

`write_hdf5` / `read_hdf5` are convenience wrappers around `HDF5Format`.
Use the class directly when you need to configure compression, pass the
format as a dependency, or subclass for a different HDF5 layout.

```python
from zutils.io.hdf5 import HDF5Format

fmt = HDF5Format(compression=None)   # disable compression
fmt.write(path, {"eeg": ts})
result = fmt.read(path)
```

`HDF5Format` implements the [`FileFormat`](../reference/io-base.md) protocol,
so it can be passed anywhere a `FileFormat` is expected:

```python
from zutils.io import FileFormat

def save(fmt: FileFormat, path: Path, data: dict) -> None:
    fmt.write(path, data)

save(HDF5Format(), path, {"eeg": ts})
```

### Custom HDF5 layouts

Subclass `HDF5Format` to implement a different on-disk layout while keeping
the same interface. For example, a uSleep-style layout stores all channels
under a single `channels/` group with a global `sample_rate` attribute:

```python
class USleepHDF5Format(HDF5Format):
    """uSleep-style HDF5: channels/ group + global sample_rate attribute."""

    def write(self, path: Path, data: dict[str, TimeSeries]) -> None:
        import h5py
        path = Path(path)
        if path.exists():
            path.unlink()
        # Validate uniform sample rate
        rates = {ts.sample_rate for ts in data.values()}
        if len(rates) != 1 or None in rates:
            raise ValueError("uSleep format requires a single sample_rate for all channels")
        sample_rate = rates.pop()
        with h5py.File(path, "w") as f:
            f.attrs["sample_rate"] = sample_rate
            channels = f.create_group("channels")
            for name, ts in data.items():
                channels.create_dataset(name, data=ts.values, compression="gzip")
                channels[name].attrs["units"] = list(ts.units)

    def read(self, path: Path) -> dict[str, TimeSeries]:
        # TODO: implement read for uSleep layout
        raise NotImplementedError
```

## `serialize` / `deserialize`

Use these when integrating with custom HDF5 files one group at a time.

```python
from zutils.io.hdf5 import serialize, deserialize

datasets, attrs = serialize(ts)
# datasets: {"values": ndarray, "timestamps": ndarray}
# attrs:    {"channel_names": [...], "units": [...], "sample_rate": 256.0}

restored = deserialize(datasets, attrs)
```

> **Note:** `HDF5Manager` (real-time append-while-recording) is a slumber
> concern and will live in `slumber.ext.units.storage`, not in zutils.
