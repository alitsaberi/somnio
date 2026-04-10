"""ZMax headband TCP client — live mode stream and stimulation commands."""

from __future__ import annotations

import logging
import socket
import types
from time import sleep, time_ns

import numpy as np

from somnio.data.timeseries import Sample

from .constants import (
    DEFAULT_IP,
    DEFAULT_PORT,
    DONGLE_MESSAGE_PREFIX,
    LED_MAX_INTENSITY,
    LIVEMODE_SENDBYTES_COMMAND,
    MIN_DATA_LENGTH,
    PACKET_TYPE_POSITION,
    SENDBYTES_MAX_RETRIES,
    STIMULATION_FLASH_LED_COMMAND,
    STIMULATION_MIN_REPETITIONS,
    STIMULATION_RETRY_DELAY,
    STIMULATION_TIME_UNIT_S,
    VALID_PACKET_TYPES,
)
from .enums import DataType, DongleStatus, LEDColor
from .protocol import (
    dec2hex,
    get_byte_at,
    seconds_to_stimulation_units,
    stimulation_led_intensity_to_pwm,
    validate_stimulation_repetitions,
)


logger = logging.getLogger(__name__)


class ConnectionClosedError(ConnectionError):
    """Raised when the ZMax TCP server closes the connection."""


def _initialize_socket(socket_timeout: float | None = None) -> socket.socket:
    """Create and configure a new TCP socket.

    Args:
        socket_timeout: Timeout in seconds for socket operations,
            or ``None`` for blocking mode.

    Returns:
        New unconnected :class:`socket.socket`.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(socket_timeout)
    return sock


def _is_dongle_message(message: str) -> bool:
    """Return True if *message* is a dongle-status notification."""
    return message.startswith(DONGLE_MESSAGE_PREFIX)


class ZMax:
    """Hypnodyne ZMax TCP client (live mode stream + stimulation commands).

    Supports use as a context manager::

        with ZMax() as zmax:
            sample = zmax.read()

    Attributes:
        _ip: IP address of the ZMax USB server.
        _port: TCP port of the ZMax USB server.
        _socket_timeout: Per-operation socket timeout in seconds.
        _socket: Active TCP socket (replaced on disconnect).
        _live_sequence_number: Counter for ``LIVEMODE_SENDBYTES`` commands.
        _dongle_status: Last reported USB dongle presence.
    """

    def __init__(
        self,
        ip: str = DEFAULT_IP,
        port: int = DEFAULT_PORT,
        socket_timeout: float | None = None,
    ) -> None:
        self._ip = ip
        self._port = port
        self._socket_timeout = socket_timeout
        self._socket = _initialize_socket(self._socket_timeout)
        self._live_sequence_number = 1
        self._dongle_status = DongleStatus.UNKNOWN

    def __repr__(self) -> str:
        return f"ZMax(ip={self._ip!r}, port={self._port})"

    def __enter__(self) -> ZMax:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        """Close the current socket and reset to a fresh unconnected socket."""
        if self._socket:
            self._socket.close()
            logger.info("Closed connection to %s", self)
        self._socket = _initialize_socket(self._socket_timeout)

    def connect(self) -> None:
        """Establish a TCP connection to the ZMax USB server.

        Sends the ``HELLO\\n`` handshake after a successful connection.

        Raises:
            ConnectionError: If the TCP connection or handshake fails.
        """
        if self.is_connected():
            logger.warning("Already connected to %s", self)
            return

        try:
            self._socket.connect((self._ip, self._port))
            logger.info("Connected to %s", self)
            self.send_string("HELLO\n")
        except OSError as exc:
            raise ConnectionError(f"Failed to connect to {self!r}") from exc

    @property
    def dongle_inserted(self) -> bool:
        """True when the USB dongle is currently reported as inserted."""
        return self._dongle_status == DongleStatus.INSERTED

    def is_connected(self) -> bool:
        """Return True if the socket has an active peer connection."""
        try:
            self._socket.getpeername()
        except OSError:
            return False
        return True

    # ------------------------------------------------------------------
    # Internal message handling
    # ------------------------------------------------------------------

    def _handle_dongle_message(self, message: str) -> None:
        """Parse a dongle-status message and update internal state.

        The ZMax server sends messages like ``_DONGLE_INSERTED`` when the
        USB dongle is plugged or unplugged.  The status part (third
        underscore-separated token) maps directly to a :class:`DongleStatus`
        member name.

        Args:
            message: Raw dongle-status string from the server.
        """
        self._dongle_status = DongleStatus[message.split("_")[2]]
        logger.info("Dongle status: %s", self._dongle_status.value)

    # ------------------------------------------------------------------
    # Data reading
    # ------------------------------------------------------------------

    def read(self, data_types: list[DataType] | None = None) -> Sample:
        """Read one sample from the live stream.

        Blocks until a valid data packet is received.  Debug, dongle-status,
        and other non-data messages are silently consumed.

        Args:
            data_types: Channels to include.  Defaults to all
                :class:`~enums.DataType` members.

        Returns:
            :class:`~somnio.data.timeseries.Sample` with values in SI units,
            timestamp from :func:`time.time_ns`, channel names from
            :attr:`DataType.name`, and units from
            :attr:`DataTypeConfig.unit`.
        """
        data_types = list(data_types) if data_types is not None else list(DataType)
        values, ts = self._read_values_until_valid(data_types)
        channel_names = tuple(dt.name for dt in data_types)
        units = tuple(dt.value.unit for dt in data_types)
        return Sample(
            values=values,
            timestamp=ts,
            channel_names=channel_names,
            units=units,
        )

    def read_numpy(self, data_types: list[DataType] | None = None) -> np.ndarray:
        """Read one sample and return raw values as a NumPy array.

        Lower-overhead alternative to :meth:`read` when metadata is not needed.

        Args:
            data_types: Channels to include.  Defaults to all
                :class:`~enums.DataType` members.

        Returns:
            1-D float64 array of SI-scaled values, one entry per channel.
        """
        data_types = list(data_types) if data_types is not None else list(DataType)
        values, _ = self._read_values_until_valid(data_types)
        return values

    def _read_values_until_valid(
        self, data_types: list[DataType]
    ) -> tuple[np.ndarray, int]:
        """Loop until a valid data packet is decoded.

        Args:
            data_types: List of channels to extract from each packet.

        Returns:
            Tuple of ``(values, timestamp_ns)`` where *values* is a float64
            array of SI-scaled channel readings and *timestamp_ns* is the
            acquisition time from :func:`time.time_ns`.

        Raises:
            ConnectionClosedError: If the server drops the connection.
        """
        while True:
            message = self._receive_line()

            if message.startswith("DEBUG"):
                logger.info("Ignoring debug message: %s", message)
                continue

            if _is_dongle_message(message):
                logger.debug("Dongle message: %s", message)
                self._handle_dongle_message(message)
                continue

            if not message.startswith("D"):
                logger.warning("Ignoring non-data message: %s", message)
                continue

            try:
                _, data = message.split(".", 1)

                logger.debug("Data length: %s", len(data))
                if not self._is_valid_data(data):
                    continue

                values = np.array(
                    [dt.value.get_value(data) for dt in data_types],
                    dtype=np.float64,
                )
                return values, time_ns()

            except ValueError:
                logger.warning("Failed to extract data from data message %s", message)
                continue

    def _receive_line(self) -> str:
        """Read one CRLF-terminated line from the socket.

        Carriage-return characters (``\\r``) are discarded; the line is
        returned without the trailing newline.

        Returns:
            Decoded UTF-8 string.

        Raises:
            ConnectionClosedError: If ``recv`` returns an empty byte string
                (clean server shutdown).
        """
        line = bytearray()
        while True:
            char = self._socket.recv(1)
            if not char:
                raise ConnectionClosedError("Lost connection to ZMax device.")
            if char == b"\r":
                continue
            if char == b"\n":
                break
            line.extend(char)
        return line.decode("utf-8")

    def _is_valid_data(self, data: str) -> bool:
        """Return True if *data* has a valid packet type and sufficient length.

        Args:
            data: Hex data portion of a ZMax live-mode line (after the dot).

        Returns:
            True when the packet type byte is in :data:`VALID_PACKET_TYPES`
            and the string length is at least :data:`MIN_DATA_LENGTH`.
        """
        packet_type = get_byte_at(data, PACKET_TYPE_POSITION)
        if packet_type not in VALID_PACKET_TYPES:
            logger.warning("Invalid type: %s", packet_type)
            return False
        if len(data) < MIN_DATA_LENGTH:
            logger.warning("Ignoring invalid data length: %s", len(data))
            return False
        return True

    # ------------------------------------------------------------------
    # Stimulation
    # ------------------------------------------------------------------

    def vibrate(
        self,
        on_duration_s: float,
        off_duration_s: float,
        repetitions: int,
    ) -> None:
        """Trigger the vibration motor without LED flash.

        Convenience wrapper around :meth:`stimulate_sequential` with
        ``led_color=LEDColor.OFF`` and ``vibration=True``.

        Args:
            on_duration_s: Duration of each vibration pulse in seconds (quantized to
                0.1s; roughly 0.1–25.5 s).
            off_duration_s: Pause before each pulse in seconds (>=0.1 s).
            repetitions: Number of pulses (1–127).
        """
        self.stimulate_sequential(
            led_color=LEDColor.OFF,
            on_duration_s=on_duration_s,
            off_duration_s=off_duration_s,
            repetitions=repetitions,
            vibration=True,
        )

    def stimulate(
        self,
        led_color: LEDColor,
        on_duration_s: float,
        off_duration_s: float,
        repetitions: int,
        vibration: bool,
        led_intensity: int = LED_MAX_INTENSITY,
        alternate_eyes: bool = False,
    ) -> None:
        """Send one ``LIVEMODE_SENDBYTES`` stimulation command.

        Builds and transmits a single LED-flash / vibration command packet.
        For multi-pulse sequences, use :meth:`stimulate_sequential` instead.

        Durations are expressed in seconds quantized to 0.1s (100 ms).

        Args:
            led_color: RGB colour for the LED flash.
            on_duration_s: LED on-time in seconds (0.1–25.5 s).
            off_duration_s: Off-time between flashes in seconds (0.1–25.5 s).
            repetitions: Number of repeats encoded in the packet (1–127).
            vibration: ``True`` to also activate the vibration motor.
            led_intensity: LED brightness as a percentage (1–100).
            alternate_eyes: ``True`` to alternate left/right LEDs each pulse.

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        validate_stimulation_repetitions(repetitions)
        on_units = seconds_to_stimulation_units(on_duration_s, name="On duration")
        off_units = seconds_to_stimulation_units(off_duration_s, name="Off duration")

        pwm = stimulation_led_intensity_to_pwm(led_intensity)

        hex_values = [
            dec2hex(x)
            for x in [
                STIMULATION_FLASH_LED_COMMAND,
                *led_color.value,
                *led_color.value,  # repeated for left + right eye
                pwm,
                0,
                on_units,
                off_units,
                repetitions,
                int(vibration),
                int(alternate_eyes),
            ]
        ]

        command = (
            f"{LIVEMODE_SENDBYTES_COMMAND} {SENDBYTES_MAX_RETRIES}"
            f" {self._get_next_live_sequence_number()}"
            f" {STIMULATION_RETRY_DELAY}"
            f" {'-'.join(hex_values)}\r\n"
        )

        logger.debug("ZMax stimulation command: %s", command)
        self.send_string(command)

    def stimulate_sequential(
        self,
        led_color: LEDColor,
        on_duration_s: float,
        off_duration_s: float,
        repetitions: int,
        vibration: bool,
        led_intensity: int = LED_MAX_INTENSITY,
        alternate_eyes: bool = False,
    ) -> None:
        """Deliver *repetitions* stimulation pulses one-at-a-time.

        Unlike :meth:`stimulate` (which encodes all repetitions in a single
        packet), this method sends one packet per pulse and waits between
        pulses.  This gives finer timing control and avoids firmware limits
        on repetition count.

        Args:
            led_color: RGB colour for the LED flash.
            on_duration_s: Duration of each vibration pulse in seconds (quantized to
                0.1s; roughly 0.1–25.5 s).
            off_duration_s: Pause before each pulse in seconds (>=0.1 s).
            repetitions: Number of pulses to deliver.
            vibration: ``True`` to also activate the vibration motor.
            led_intensity: LED brightness as a percentage (1–100).
            alternate_eyes: ``True`` to alternate left/right LEDs each pulse.

        Raises:
            ValueError: If any parameter is outside its valid range (same as
                :meth:`stimulate`). Ensures the inter-pulse delay is never
                negative before :func:`time.sleep` is called.
        """

        sleep_duration_s = off_duration_s - STIMULATION_TIME_UNIT_S

        if sleep_duration_s < 0:
            raise ValueError("Off duration must be greater than 100 ms")

        if repetitions < STIMULATION_MIN_REPETITIONS:
            raise ValueError(
                f"Repetitions must be at least {STIMULATION_MIN_REPETITIONS}"
            )

        for i in range(repetitions):
            logger.info("Stimulating %s/%s", i + 1, repetitions)
            sleep(sleep_duration_s)
            self.stimulate(
                led_color=led_color,
                on_duration_s=on_duration_s,
                off_duration_s=STIMULATION_TIME_UNIT_S,
                repetitions=1,
                vibration=vibration,
                led_intensity=led_intensity,
                alternate_eyes=alternate_eyes,
            )

    # ------------------------------------------------------------------
    # Low-level socket helpers
    # ------------------------------------------------------------------

    def send_string(self, message: str) -> None:
        """Encode *message* as UTF-8 and send it over the socket.

        Args:
            message: String to transmit.
        """
        self._socket.sendall(message.encode("utf-8"))

    def _get_next_live_sequence_number(self) -> int:
        """Increment and return the live-sequence counter modulo 256."""
        self._live_sequence_number += 1
        return self._live_sequence_number % 256
