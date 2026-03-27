# Data Types

`zutils.data` provides two pure in-memory containers for sleep and physiological signals.
They carry no I/O or serialization logic — that belongs in [`zutils.io`](../reference/data.md).

## Core types

| Type | Shape | Use case |
|---|---|---|
| `Sample` | `(n_channels,)` | Single time-point; streaming through ezmsg DAGs |
| `TimeSeries` | `(n_samples, n_channels)` | Multi-sample block; storage, processing, windowing |

## Conventions

All values and metadata follow these rules throughout zutils:

- **Timestamps** — `int64` nanoseconds since Unix epoch (`time.time_ns()`).
- **Values** — always `float64`; integer sensors are cast on construction.
- **Physical units** — SI base units, tracked per-channel in the `units` field.
  Use `"V"` (not `"uV"`), `"m/s^2"` (not `"g"`), `"degC"` for temperature.
  The I/O layer handles format-specific scaling (e.g. EDF stores µV → `read_edf` converts to V).
- **`sample_rate`** — `float` Hz or `None` for irregular data. Timestamps are always
  authoritative; `sample_rate` is metadata, not a replacement.
- **`channel_names`** — unique, underscore-separated strings (e.g. `"EEG_L"`, `"ACC_X"`).

## Creating a `Sample`

```python
import numpy as np
from zutils.data import Sample

sample = Sample(
    values=np.array([0.000123, -0.000045, 9.81, 0.0, 0.12, 36.5]),
    timestamp=1_700_000_000_000_000_000,  # ns since Unix epoch
    channel_names=["EEG_L", "EEG_R", "ACC_X", "ACC_Y", "ACC_Z", "TEMP"],
    units=["V", "V", "m/s^2", "m/s^2", "m/s^2", "degC"],
)
```

`values` is coerced to `float64` on construction. Passing integer arrays is safe.

## Creating a `TimeSeries`

```python
import numpy as np
from zutils.data import TimeSeries

n_samples = 256  # 1 second at 256 Hz
step_ns = int(1e9 / 256)

ts = TimeSeries(
    values=np.zeros((n_samples, 2), dtype=np.float64),
    timestamps=np.arange(n_samples, dtype=np.int64) * step_ns,
    channel_names=["EEG_L", "EEG_R"],
    units=["V", "V"],
    sample_rate=256.0,
)

print(ts.n_samples, ts.n_channels)  # 256 2
print(ts.duration)                  # 0:00:00.996093
print(ts.is_regular)                # True
```

## Selecting channels

```python
eeg = ts.select_channels(["EEG_L"])
```

## Slicing by time

```python
# Keep samples between t=0 and t=0.5 s (in nanoseconds)
half = ts.select_time(start=0, end=int(0.5e9))
```

## Integer / slice indexing

```python
first_100 = ts[:100]   # TimeSeries with 100 samples
single    = ts[42]     # TimeSeries with 1 sample
```

`channel_names`, `units`, and `sample_rate` are always preserved.

## Concatenating `TimeSeries` objects

```python
from zutils.data import concat

combined = concat([ts_a, ts_b, ts_c])
```

`sample_rate` is propagated only when all inputs share the same value; otherwise it is set to `None`.
`channel_names` and `units` must match across all inputs.

## Building a `TimeSeries` from `Sample` objects

```python
from zutils.data import collect_samples

samples = [
    Sample(values=np.array([v, -v]), timestamp=i * step_ns,
           channel_names=["EEG_L", "EEG_R"], units=["V", "V"])
    for i, v in enumerate([0.001, 0.002, 0.003])
]

ts = collect_samples(samples)
# ts.sample_rate is None — caller can set it if the source is known to be regular
ts.sample_rate = 256.0
```
