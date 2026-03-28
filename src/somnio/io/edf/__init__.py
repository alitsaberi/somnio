"""EDF signal I/O layouts (optional ``edf`` extra: MNE + edfio)."""

from somnio.io.edf.standard import (
    StandardEDFReader,
    StandardEDFWriter,
    read as read_standard,
    write as write_standard,
)
from somnio.io.edf.zmax import (
    ZMaxMultiEDFReader,
    ZMaxMultiEDFWriter,
    read as read_zmax_multi,
    write as write_zmax_multi,
)

__all__ = [
    "StandardEDFReader",
    "StandardEDFWriter",
    "ZMaxMultiEDFReader",
    "ZMaxMultiEDFWriter",
    "read_standard",
    "read_zmax_multi",
    "write_standard",
    "write_zmax_multi",
]
