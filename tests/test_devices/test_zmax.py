"""Tests for ZMax TCP client (mocked socket)."""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest

from somnio.data.timeseries import Sample
from somnio.devices.zmax import (
    ConnectionClosedError,
    DataType,
    ZMax,
)
from somnio.devices.zmax.enums import DongleStatus, LEDColor, scale_eeg
from somnio.devices.zmax.protocol import dec2hex, get_byte_at, get_word_at


# ---------------------------------------------------------------------------
# Helpers for building fake data buffers
# ---------------------------------------------------------------------------


def _make_data_buffer(n_bytes: int = 40, packet_type: int = 1) -> str:
    """Return a space-separated hex buffer of *n_bytes* bytes.

    The first byte is set to *packet_type*; all others default to ``0x00``.
    """
    raw = [0] * n_bytes
    raw[0] = packet_type
    return " ".join(f"{b:02X}" for b in raw)


def _make_data_line(buffer: str, prefix: str = "D0") -> bytes:
    """Wrap a hex buffer in a ZMax live-mode line ``{prefix}.{buffer}\\r\\n``."""
    return f"{prefix}.{buffer}\r\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_socket():
    """Patch ``socket.socket`` for the duration of the test."""
    with patch("socket.socket") as mock:
        yield mock


@pytest.fixture
def zmax_device(mock_socket):
    """Return an unconnected ZMax instance backed by a mock socket."""
    mock_socket.return_value.getpeername.side_effect = OSError()
    return ZMax("127.0.0.1", 8080)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_zmax_initialization(mock_socket):
    mock_socket.return_value.getpeername.side_effect = OSError()
    zmax = ZMax("127.0.0.1", 8080)

    assert zmax._ip == "127.0.0.1"
    assert zmax._port == 8080
    assert zmax._dongle_status == DongleStatus.UNKNOWN
    assert not zmax.is_connected()


def test_zmax_repr(mock_socket):
    zmax = ZMax("192.168.1.1", 9000)
    assert repr(zmax) == "ZMax(ip='192.168.1.1', port=9000)"


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


def test_zmax_context_manager(mock_socket):
    """Context manager should connect on enter and disconnect on exit."""
    # Initially disconnected; after connect() the socket reports a peer.
    mock_socket.return_value.getpeername.side_effect = [
        OSError(),  # checked inside connect() guard
        ("127.0.0.1", 8080),  # checked after entering the context
    ]

    with ZMax("127.0.0.1", 8080) as zmax:
        mock_socket.return_value.connect.assert_called_once()
        assert zmax.is_connected()

    mock_socket.return_value.close.assert_called_once()


def test_zmax_connection_error(mock_socket):
    """connect() should raise ConnectionError when the TCP connect fails."""
    mock_socket.return_value.getpeername.side_effect = OSError()
    mock_socket.return_value.connect.side_effect = OSError("refused")
    zmax = ZMax("127.0.0.1", 8080)

    with pytest.raises(ConnectionError):
        zmax.connect()

    mock_socket.return_value.connect.assert_called_once()


def test_zmax_already_connected_skips_reconnect(mock_socket, caplog):
    """Calling connect() on an already-connected ZMax logs a warning."""
    mock_socket.return_value.getpeername.return_value = ("127.0.0.1", 8080)
    zmax = ZMax("127.0.0.1", 8080)

    with caplog.at_level(logging.WARNING):
        zmax.connect()

    assert "Already connected" in caplog.text
    mock_socket.return_value.connect.assert_not_called()


def test_zmax_is_connected(mock_socket):
    """is_connected() uses getpeername to detect active connections."""
    mock_socket.return_value.getpeername.side_effect = OSError()
    zmax = ZMax("127.0.0.1", 8080)
    assert not zmax.is_connected()

    mock_socket.return_value.getpeername.side_effect = None
    mock_socket.return_value.getpeername.return_value = ("127.0.0.1", 8080)
    assert zmax.is_connected()


def test_zmax_disconnect_closes_socket(mock_socket):
    """disconnect() closes the current socket and reinitialises it."""
    mock_socket.return_value.getpeername.return_value = ("127.0.0.1", 8080)
    zmax = ZMax("127.0.0.1", 8080)

    zmax.disconnect()

    mock_socket.return_value.close.assert_called_once()


# ---------------------------------------------------------------------------
# Data reading
# ---------------------------------------------------------------------------


def _make_recv_responses(data_line: bytes):
    """Split a data line into per-byte recv() return values."""
    return [bytes([b]) for b in data_line] + [b""]


def test_zmax_read(mock_socket):
    """read() should return a Sample with correctly scaled SI values."""
    buf = _make_data_buffer(40)
    data_line = _make_data_line(buf)
    mock_socket.return_value.recv.side_effect = _make_recv_responses(data_line)
    mock_socket.return_value.getpeername.return_value = ("127.0.0.1", 8080)

    zmax = ZMax("127.0.0.1", 8080)
    data_types = [DataType.EEG_RIGHT, DataType.EEG_LEFT]
    sample = zmax.read(data_types=data_types)

    assert isinstance(sample, Sample)
    assert sample.channel_names == ("EEG_RIGHT", "EEG_LEFT")
    assert sample.units == ("V", "V")
    assert sample.values.shape == (2,)
    assert sample.values.dtype == np.float64


def test_zmax_read_numpy(mock_socket):
    """read_numpy() should return a 1-D float64 array."""
    buf = _make_data_buffer(40)
    data_line = _make_data_line(buf)
    mock_socket.return_value.recv.side_effect = _make_recv_responses(data_line)
    mock_socket.return_value.getpeername.return_value = ("127.0.0.1", 8080)

    zmax = ZMax("127.0.0.1", 8080)
    values = zmax.read_numpy(data_types=[DataType.EEG_RIGHT])

    assert isinstance(values, np.ndarray)
    assert values.ndim == 1
    assert values.dtype == np.float64


def test_zmax_read_skips_debug_messages(mock_socket):
    """read() silently discards DEBUG lines and reads the next valid packet."""
    buf = _make_data_buffer(40)
    debug_line = b"DEBUG some debug info\r\n"
    data_line = _make_data_line(buf)
    bytes_sequence = (
        _make_recv_responses(debug_line)[:-1]  # drop the trailing b""
        + _make_recv_responses(data_line)
    )
    mock_socket.return_value.recv.side_effect = bytes_sequence

    zmax = ZMax("127.0.0.1", 8080)
    sample = zmax.read(data_types=[DataType.EEG_RIGHT])
    assert isinstance(sample, Sample)


def test_zmax_connection_lost(mock_socket):
    """read() raises ConnectionClosedError when the socket is closed by the server."""
    mock_socket.return_value.recv.return_value = b""

    zmax = ZMax("127.0.0.1", 8080)
    with pytest.raises(ConnectionClosedError):
        zmax.read(data_types=[DataType.EEG_RIGHT])


def test_zmax_read_skips_invalid_data(mock_socket):
    """Invalid packet type in first position causes the packet to be discarded."""
    bad_buf = _make_data_buffer(
        40, packet_type=0
    )  # type 0 is not in VALID_PACKET_TYPES
    good_buf = _make_data_buffer(40, packet_type=1)
    bad_line = _make_data_line(bad_buf)
    good_line = _make_data_line(good_buf)
    bytes_sequence = _make_recv_responses(bad_line)[:-1] + _make_recv_responses(
        good_line
    )
    mock_socket.return_value.recv.side_effect = bytes_sequence

    zmax = ZMax("127.0.0.1", 8080)
    sample = zmax.read(data_types=[DataType.EEG_RIGHT])
    assert isinstance(sample, Sample)


def test_zmax_read_skips_short_data(mock_socket):
    """Data shorter than MIN_DATA_LENGTH is discarded; next valid packet is used."""
    short_buf = "01 " * 10  # only 10 bytes — too short
    short_buf = short_buf.rstrip()
    good_buf = _make_data_buffer(40, packet_type=1)
    short_line = _make_data_line(short_buf)
    good_line = _make_data_line(good_buf)
    bytes_sequence = _make_recv_responses(short_line)[:-1] + _make_recv_responses(
        good_line
    )
    mock_socket.return_value.recv.side_effect = bytes_sequence

    zmax = ZMax("127.0.0.1", 8080)
    sample = zmax.read(data_types=[DataType.EEG_RIGHT])
    assert isinstance(sample, Sample)


# ---------------------------------------------------------------------------
# Dongle status
# ---------------------------------------------------------------------------


def test_dongle_status_updated_from_message(mock_socket):
    """_handle_dongle_message parses the status token from the message."""
    zmax = ZMax("127.0.0.1", 8080)
    zmax._handle_dongle_message("_DONGLE_INSERTED")
    assert zmax._dongle_status == DongleStatus.INSERTED
    assert zmax.dongle_inserted is True


def test_dongle_removed(mock_socket):
    zmax = ZMax("127.0.0.1", 8080)
    zmax._handle_dongle_message("_DONGLE_REMOVED")
    assert zmax._dongle_status == DongleStatus.REMOVED
    assert zmax.dongle_inserted is False


def test_dongle_message_skipped_during_read(mock_socket):
    """Dongle messages encountered during read() are consumed transparently."""
    buf = _make_data_buffer(40)
    dongle_line = b"_DONGLE_INSERTED\r\n"
    data_line = _make_data_line(buf)
    bytes_sequence = _make_recv_responses(dongle_line)[:-1] + _make_recv_responses(
        data_line
    )
    mock_socket.return_value.recv.side_effect = bytes_sequence

    zmax = ZMax("127.0.0.1", 8080)
    sample = zmax.read(data_types=[DataType.EEG_RIGHT])
    assert isinstance(sample, Sample)
    assert zmax._dongle_status == DongleStatus.INSERTED


# ---------------------------------------------------------------------------
# Stimulation / vibration validation
# ---------------------------------------------------------------------------


def test_stimulate_validates_repetitions(mock_socket):
    zmax = ZMax("127.0.0.1", 8080)
    with pytest.raises(ValueError, match="Repetitions"):
        zmax.stimulate(LEDColor.RED, 10, 10, 0, False)

    with pytest.raises(ValueError, match="Repetitions"):
        zmax.stimulate(LEDColor.RED, 10, 10, 200, False)


def test_stimulate_validates_on_duration(mock_socket):
    zmax = ZMax("127.0.0.1", 8080)
    with pytest.raises(ValueError, match="On duration"):
        zmax.stimulate(LEDColor.RED, 0, 10, 1, False)

    with pytest.raises(ValueError, match="On duration"):
        zmax.stimulate(LEDColor.RED, 300, 10, 1, False)


def test_stimulate_validates_off_duration(mock_socket):
    zmax = ZMax("127.0.0.1", 8080)
    with pytest.raises(ValueError, match="Off duration"):
        zmax.stimulate(LEDColor.RED, 10, 0, 1, False)


def test_stimulate_validates_led_intensity(mock_socket):
    zmax = ZMax("127.0.0.1", 8080)
    with pytest.raises(ValueError, match="LED intensity"):
        zmax.stimulate(LEDColor.RED, 10, 10, 1, False, led_intensity=0)

    with pytest.raises(ValueError, match="LED intensity"):
        zmax.stimulate(LEDColor.RED, 10, 10, 1, False, led_intensity=101)


def test_stimulate_sends_correct_command(mock_socket):
    """stimulate() encodes the packet and calls send_string."""
    zmax = ZMax("127.0.0.1", 8080)

    zmax.stimulate(
        led_color=LEDColor.RED,
        on_duration=10,
        off_duration=10,
        repetitions=1,
        vibration=False,
        led_intensity=100,
        alternate_eyes=False,
    )

    mock_socket.return_value.sendall.assert_called_once()
    sent_bytes = mock_socket.return_value.sendall.call_args[0][0]
    sent_str = sent_bytes.decode("utf-8")
    assert sent_str.startswith("LIVEMODE_SENDBYTES")
    assert sent_str.endswith("\r\n")


def test_stimulate_sequential_validates_off_duration_before_sleep(mock_socket):
    """Invalid ``off_duration`` must fail with ValueError before ``sleep`` runs."""
    zmax = ZMax("127.0.0.1", 8080)
    with patch("somnio.devices.zmax.client.sleep") as mock_sleep:
        with pytest.raises(ValueError, match="Off duration"):
            zmax.stimulate_sequential(
                LEDColor.RED,
                on_duration=10,
                off_duration=0,
                repetitions=1,
                vibration=False,
            )
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# DataType / protocol utilities
# ---------------------------------------------------------------------------


def test_data_type_scaling_si():
    """SI scale functions produce values in the correct unit and magnitude."""
    # EEG midpoint (32768) → 0 V
    assert scale_eeg(32768) == pytest.approx(0.0)

    # EEG at full-scale positive (65535) → ≈ 1952 µV = 1.952e-3 V
    assert scale_eeg(65535) == pytest.approx(3952 / 2 * 1e-6, rel=1e-3)

    # All DataType members should have non-empty file_name and unit
    for dt in DataType:
        assert dt.value.file_name
        assert dt.value.unit


def test_data_type_category():
    assert DataType.EEG_RIGHT.category == "EEG"
    assert DataType.ACCELEROMETER_X.category == "ACCELEROMETER"
    assert DataType.OXIMETER_INFRARED_AC.category == "OXIMETER"


def test_get_by_category():
    eeg = DataType.get_by_category("EEG")
    assert DataType.EEG_RIGHT in eeg
    assert DataType.EEG_LEFT in eeg
    assert DataType.ACCELEROMETER_X not in eeg


def test_data_type_str():
    assert str(DataType.EEG_RIGHT) == "EEG_RIGHT"


def test_protocol_get_byte_at():
    buf = "00 01 FF 10"
    assert get_byte_at(buf, 0) == 0x00
    assert get_byte_at(buf, 1) == 0x01
    assert get_byte_at(buf, 2) == 0xFF
    assert get_byte_at(buf, 3) == 0x10


def test_protocol_get_word_at():
    buf = "01 02 03 04"
    assert get_word_at(buf, 0) == 0x0102
    assert get_word_at(buf, 2) == 0x0304


def test_protocol_dec2hex():
    assert dec2hex(0) == "00"
    assert dec2hex(255) == "FF"
    assert dec2hex(1, pad=4) == "0001"


def test_expected_data_length_matches_min():
    """EXPECTED_DATA_LENGTH is an alias for MIN_DATA_LENGTH."""
    from somnio.devices.zmax.constants import EXPECTED_DATA_LENGTH, MIN_DATA_LENGTH

    assert EXPECTED_DATA_LENGTH == MIN_DATA_LENGTH


def test_ensure_connected_raises_when_disconnected(mock_socket):
    """ensure_connected raises ValueError when the device is not connected."""
    from somnio.devices.zmax import ensure_connected

    mock_socket.return_value.getpeername.side_effect = OSError()
    zmax = ZMax("127.0.0.1", 8080)

    with pytest.raises(ValueError, match="is not connected"):
        ensure_connected(zmax)


def test_ensure_connected_returns_zmax_when_connected(mock_socket):
    """ensure_connected returns the ZMax instance when connected."""
    from somnio.devices.zmax import ensure_connected

    mock_socket.return_value.getpeername.return_value = ("127.0.0.1", 8080)
    zmax = ZMax("127.0.0.1", 8080)

    result = ensure_connected(zmax)
    assert result is zmax
