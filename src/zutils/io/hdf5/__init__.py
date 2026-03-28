"""HDF5 signal I/O layouts (requires optional ``hdf5`` dependency)."""

from zutils.io.hdf5.native import (
    NativeHDF5Reader,
    NativeHDF5Writer,
    deserialize,
    read,
    read_all,
    serialize,
    write,
)
from zutils.io.hdf5.usleep import (
    USleepHDF5Reader,
    USleepHDF5Writer,
    align_timestamps_to_usleep_grid,
    read as usleep_read,
    write as usleep_write,
)

__all__ = [
    "NativeHDF5Reader",
    "NativeHDF5Writer",
    "USleepHDF5Reader",
    "USleepHDF5Writer",
    "align_timestamps_to_usleep_grid",
    "deserialize",
    "read",
    "read_all",
    "serialize",
    "usleep_read",
    "usleep_write",
    "write",
]
