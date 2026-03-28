"""Binary parsing helpers for ZMax live-mode hex data packets."""


def get_byte_at(buffer: str, index: int) -> int:
    """Read one byte from a space-separated hex buffer.

    The buffer is encoded as hex pairs separated by spaces:
    ``"00 1A FF ..."`` where each pair is one byte and ``index`` is
    the zero-based byte position.

    Args:
        buffer: Space-separated hex string (2 hex chars + 1 space per byte,
            no trailing space).
        index: Zero-based byte index.

    Returns:
        Integer value of the byte at *index*.
    """
    hex_str = buffer[index * 3 : index * 3 + 2]
    return int(hex_str, 16)


def get_word_at(buffer: str, index: int) -> int:
    """Read a big-endian 16-bit word from a space-separated hex buffer.

    Combines the byte at *index* (high byte) and *index + 1* (low byte)
    into a single 16-bit integer.

    Args:
        buffer: Space-separated hex string.
        index: Zero-based byte index of the high byte.

    Returns:
        16-bit integer value.
    """
    return get_byte_at(buffer, index) * 256 + get_byte_at(buffer, index + 1)


def dec2hex(decimal: int, pad: int = 2) -> str:
    """Format an integer as an uppercase hex string with zero-padding.

    Args:
        decimal: Non-negative integer to convert.
        pad: Minimum number of hex digits (zero-padded on the left).

    Returns:
        Uppercase hex string of at least *pad* characters.
    """
    return format(decimal, f"0{pad}x").upper()
