# FileFormat Protocol

API reference for `zutils.io.base`.

---

## `FileFormat`

```python
@runtime_checkable
class FileFormat(Protocol):
    def write(self, path: Path, data: dict[str, TimeSeries]) -> None: ...
    def read(self, path: Path) -> dict[str, TimeSeries]: ...
```

Protocol defining the read/write contract for all zutils file format handlers.

Each concrete format (e.g. `HDF5Format`) implements this protocol. Within a
single file format, multiple layout strategies are possible by providing
different implementations (e.g. the default zutils HDF5 layout vs. a uSleep
HDF5 layout).

`FileFormat` is decorated with `@runtime_checkable`, so `isinstance` checks
work at runtime.

### `write(path, data)`

```python
def write(self, path: Path, data: dict[str, TimeSeries]) -> None
```

Write named `TimeSeries` objects to `path`.

### `read(path)`

```python
def read(self, path: Path) -> dict[str, TimeSeries]
```

Read named `TimeSeries` objects from `path`. Returns `dict[str, TimeSeries]`.

---

## Implementing a custom format

```python
from pathlib import Path
from zutils.data import TimeSeries
from zutils.io import FileFormat

class MyFormat:
    """Custom file format implementing the FileFormat protocol."""

    def write(self, path: Path, data: dict[str, TimeSeries]) -> None:
        ...

    def read(self, path: Path) -> dict[str, TimeSeries]:
        ...

assert isinstance(MyFormat(), FileFormat)  # True
```
