# Data Reference

API reference for `zutils.data`.

---

## `Sample`

```python
@dataclass
class Sample:
    values: np.ndarray          # (n_channels,) float64
    timestamp: int              # ns since Unix epoch
    channel_names: tuple[str, ...]  # unique, length == n_channels
    units: tuple[str, ...]      # SI symbols, length == n_channels
```

Single time-point, multi-channel measurement. Lightweight type for streaming through ezmsg DAGs (one sample per message).

**Validation** (`__post_init__`):

- `values` is coerced to `float64`.
- `values.ndim` must be 1.
- `len(channel_names) == len(units) == n_channels`.
- `channel_names` must be unique.

---

## `TimeSeries`

```python
@dataclass
class TimeSeries:
    values: np.ndarray              # (n_samples, n_channels) float64
    timestamps: np.ndarray          # (n_samples,) int64 ns since Unix epoch
    channel_names: tuple[str, ...]  # unique, length == n_channels
    units: tuple[str, ...]          # SI symbols, length == n_channels
    sample_rate: float | None = None  # nominal Hz when regular; None = irregular/unknown
```

Timestamp-first multi-channel timeseries. Supports irregular sampling natively. When `sample_rate` is set, it denotes **nominal** regular sampling; timestamps are **not** auto-adjusted to match. Strict grids for a given file format are enforced in that format’s writer (see [I/O reference](io.md)).

**Validation** (`__post_init__`):

- `values` is coerced to `float64`; `timestamps` to `int64`.
- `values.ndim` must be 2.
- `timestamps.shape == (n_samples,)`.
- `len(channel_names) == len(units) == n_channels`.
- `channel_names` must be unique.
- `sample_rate > 0` when not `None`.

### Properties

| Property | Type | Description |
|---|---|---|
| `n_samples` | `int` | Number of rows in `values` |
| `n_channels` | `int` | Number of columns in `values` |
| `shape` | `tuple[int, int]` | `(n_samples, n_channels)` |
| `duration` | `timedelta` | Wall-clock span from first to last timestamp |
| `channel_index_map` | `dict[str, int]` | Channel name → column index (cached) |
| `is_regular` | `bool` | `True` when `sample_rate is not None` |

### Methods

#### `select_channels(names)`

```python
def select_channels(self, names: list[str]) -> TimeSeries
```

Return a new `TimeSeries` restricted to the given channel names (in the given order). Raises `KeyError` for unknown names.

#### `select_time(start, end)`

```python
def select_time(
    self,
    start: int | None = None,
    end: int | None = None,
) -> TimeSeries
```

Return samples within `[start, end]` nanosecond timestamps (inclusive). `None` means no bound.

#### `__getitem__(key)`

```python
def __getitem__(self, key: int | slice) -> TimeSeries
```

Integer or slice indexing along the sample axis. An integer index returns a single-sample `TimeSeries`. Preserves `channel_names`, `units`, and `sample_rate`.

---

## `concat`

```python
def concat(series: Sequence[TimeSeries]) -> TimeSeries
```

Concatenate `TimeSeries` objects along the time axis.

- `series` must be non-empty.
- All inputs must share `channel_names` and `units`.
- `sample_rate` is propagated only when all inputs share the same value; otherwise `None`.

---

## `collect_samples`

```python
def collect_samples(samples: Sequence[Sample]) -> TimeSeries
```

Batch a sequence of `Sample` objects into a `TimeSeries`.

- `samples` must be non-empty.
- All inputs must share `channel_names` and `units`.
- Returns `sample_rate=None` — caller may set it afterward if the source is known to be regular.

---

## MNE adapter (optional)

Module: `zutils.data.adapters.mne` (requires **`zutils[edf]`** or **`mne`** installed).

| Function | Purpose |
|----------|---------|
| `from_mne_raw(raw)` | Build `TimeSeries` from a loaded MNE `Raw` (SI scaling and timestamps follow MNE; channel names normalized with spaces → underscores). |
| `to_mne_raw(data)` | Build `RawArray` from `TimeSeries` (`sample_rate` required; `meas_date` from the first timestamp). |
| `import_mne()` | Return the `mne` module or raise with an install hint. |

The [EDF I/O reference](io.md#optional-dependency-edf) uses these helpers for all EDF layouts. They are the single implementation for MNE ↔ `TimeSeries` conversion in zutils.
