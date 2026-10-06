from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

from .ble_proto import (
    camera_control_pb2,
    cohn_pb2,
    network_management_pb2,
    response_generic_pb2,
    set_camera_control_status_pb2,
)

logger = logging.getLogger(__name__)

GOPRO_BLE_SERVICE_UUID = "0000fea6-0000-1000-8000-00805f9b34fb"
GOPRO_COMPANY_ID = 0x02F2
GOPRO_BASE_UUID = "b5f9{:04x}-aa8d-11e3-9046-0002a5d5c51b"

KEEP_ALIVE_COMMAND_ID = 0x5B
KEEP_ALIVE_VALUE = 0x42
KEEP_ALIVE_INTERVAL_SECONDS = 3.0
PAIRING_TRIGGER_TIMEOUT_SECONDS = 10.0
SET_THIRD_PARTY_CLIENT_INFO = 0x50
GET_OPEN_GOPRO_API_VERSION = 0x51
DEFAULT_PHONE_NAME = "HapticaProvisioner"
SLEEP_COMMAND_ID = 0x05
SET_AP_CONTROL_COMMAND_ID = 0x17

QUERY_GET_STATUS = 0x13
QUERY_REGISTER_STATUS = 0x53
QUERY_ASYNC_STATUS = 0x93

DEFAULT_STATUS_IDS = (8, 10)
SUPPORTED_OPEN_GOPRO_API_VERSION = "2.0"
PAIRING_STATE_STATUS_ID = 19
LAST_PAIRING_TYPE_STATUS_ID = 20
LAST_PAIRING_SUCCESS_STATUS_ID = 21
WIFI_PROVISIONING_STATE_STATUS_ID = 24
CONNECTED_WIFI_SSID_STATUS_ID = 29
WIFI_BARS_STATUS_ID = 56
WIFI_STATUS_IDS = (
    WIFI_PROVISIONING_STATE_STATUS_ID,
    CONNECTED_WIFI_SSID_STATUS_ID,
    WIFI_BARS_STATUS_ID,
)


def _new_bleak_client(
    device: BLEDevice | str,
    *,
    timeout: float,
    use_cached: Optional[bool],
) -> BleakClient:
    try:
        if use_cached is None:
            return BleakClient(device, timeout=timeout)
        return BleakClient(device, use_cached=use_cached, timeout=timeout)
    except TypeError:
        return BleakClient(device, timeout=timeout)


def gopro_uuid(short_id: int) -> str:
    return GOPRO_BASE_UUID.format(short_id).lower()


COMMAND_REQ_UUID = gopro_uuid(0x0072)
COMMAND_RSP_UUID = gopro_uuid(0x0073)
SETTINGS_REQ_UUID = gopro_uuid(0x0074)
SETTINGS_RSP_UUID = gopro_uuid(0x0075)
QUERY_REQ_UUID = gopro_uuid(0x0076)
QUERY_RSP_UUID = gopro_uuid(0x0077)
NETWORK_MGMT_REQ_UUID = gopro_uuid(0x0091)
NETWORK_MGMT_RSP_UUID = gopro_uuid(0x0092)
WIFI_AP_SSID_UUID = gopro_uuid(0x0002)
WIFI_AP_PASSWORD_UUID = gopro_uuid(0x0003)
NOTIFY_RESPONSE_UUIDS = {
    COMMAND_RSP_UUID,
    SETTINGS_RSP_UUID,
    QUERY_RSP_UUID,
    NETWORK_MGMT_RSP_UUID,
}

PROTOBUF_RESPONSE_MAP = {
    (0x03, 0x81): response_generic_pb2.ResponseGeneric,
    (0xF1, 0xE9): response_generic_pb2.ResponseGeneric,
    (0xF1, 0xE2): response_generic_pb2.ResponseGeneric,
    (0xF1, 0xF8): response_generic_pb2.ResponseGeneric,
    (0x02, 0x82): network_management_pb2.ResponseStartScanning,
    (0x02, 0x0B): network_management_pb2.NotifStartScanning,
    (0x02, 0x83): network_management_pb2.ResponseGetApEntries,
    (0x02, 0x84): network_management_pb2.ResponseConnect,
    (0x02, 0x85): network_management_pb2.ResponseConnectNew,
    (0x02, 0x0C): network_management_pb2.NotifProvisioningState,
    # COHN responses (Feature ID 0xF1 command / 0xF5 query)
    (0xF1, 0xE5): response_generic_pb2.ResponseGeneric,  # RESPONSE_COHN_SETTING
    (0xF1, 0xE6): response_generic_pb2.ResponseGeneric,  # RESPONSE_CLEAR_COHN_CERT
    (0xF1, 0xE7): response_generic_pb2.ResponseGeneric,  # RESPONSE_CREATE_COHN_CERT
    (0xF5, 0xEE): cohn_pb2.ResponseCOHNCert,  # RESPONSE_GET_COHN_CERT
    (0xF5, 0xEF): cohn_pb2.NotifyCOHNStatus,  # RESPONSE_GET_COHN_STATUS / NOTIF
}


def _decode_status_int(value: Optional[bytes]) -> Optional[int]:
    if not value:
        return None
    if len(value) == 1:
        return value[0]
    if len(value) in (2, 4, 8):
        return int.from_bytes(value, "big", signed=False)
    return int.from_bytes(value[:1], "big", signed=False)


def _decode_status_string(value: Optional[bytes]) -> Optional[str]:
    if not value:
        return None
    try:
        text = value.decode("utf-8", errors="ignore").strip("\x00").strip()
    except Exception:
        return None
    if not text:
        return None
    if all((31 < ord(c) < 127) or c in "\t\n\r" for c in text):
        return text
    return None


def interpret_wifi_status(values: dict[int, bytes], expected_ssid: Optional[str] = None) -> dict[str, Any]:
    provisioning_state = _decode_status_int(values.get(WIFI_PROVISIONING_STATE_STATUS_ID))
    connected_ssid = _decode_status_string(values.get(CONNECTED_WIFI_SSID_STATUS_ID))
    wifi_bars = _decode_status_int(values.get(WIFI_BARS_STATUS_ID))
    provisioning_complete = provisioning_state == 4
    verified = False
    if expected_ssid and connected_ssid == expected_ssid:
        verified = True
    elif connected_ssid:
        verified = True
    elif wifi_bars is not None and wifi_bars > 0:
        verified = True
    return {
        "provisioning_state": provisioning_state,
        "provisioning_complete": provisioning_complete,
        "connected_ssid": connected_ssid,
        "wifi_bars": wifi_bars,
        "verified": verified,
        "raw": {str(k): v.hex() for k, v in values.items()},
    }


@dataclass
class TlvResponse:
    uuid: str
    id: int
    status: int
    payload: bytes


@dataclass
class QueryResponse(TlvResponse):
    data: dict[int, bytes]


@dataclass
class ProtobufResponse:
    uuid: str
    feature_id: int
    action_id: int
    data: object


@dataclass
class _ResponseWaiter:
    predicate: Callable[[object], bool]
    future: asyncio.Future


class _ResponseAccumulator:
    def __init__(self, uuid: str) -> None:
        self.uuid = uuid
        self.bytes_remaining = 0
        self.raw_bytes = bytearray()

    @property
    def is_received(self) -> bool:
        return len(self.raw_bytes) > 0 and self.bytes_remaining == 0

    def reset(self) -> None:
        self.bytes_remaining = 0
        self.raw_bytes = bytearray()

    def accumulate(self, data: bytes) -> None:
        cont_mask = 0b10000000
        hdr_mask = 0b01100000
        gen_len_mask = 0b00011111
        ext_13_byte0_mask = 0b00011111

        buf = bytearray(data)
        if buf[0] & cont_mask:
            if self.bytes_remaining <= 0:
                logger.debug("Ignoring continuation packet without start header")
                return
            buf.pop(0)
        else:
            self.raw_bytes = bytearray()
            hdr = (buf[0] & hdr_mask) >> 5
            if hdr == 0b00:
                self.bytes_remaining = buf[0] & gen_len_mask
                buf = buf[1:]
            elif hdr == 0b01:
                if len(buf) < 2:
                    logger.debug("BLE header too short for extended length (len=%s)", len(buf))
                    return
                self.bytes_remaining = ((buf[0] & ext_13_byte0_mask) << 8) + buf[1]
                buf = buf[2:]
            elif hdr == 0b10:
                if len(buf) < 3:
                    logger.debug("BLE header too short for 16-bit length (len=%s)", len(buf))
                    return
                self.bytes_remaining = (buf[1] << 8) + buf[2]
                buf = buf[3:]
            else:
                raise ValueError("Unsupported BLE header type")

        self.raw_bytes.extend(buf)
        self.bytes_remaining -= len(buf)


def _parse_tlv(uuid: str, raw: bytes) -> TlvResponse:
    return TlvResponse(uuid=uuid, id=raw[0], status=raw[1], payload=bytes(raw[2:]))


def _parse_query(uuid: str, raw: bytes) -> QueryResponse:
    base = _parse_tlv(uuid, raw)
    buf = bytearray(base.payload)
    data: dict[int, bytes] = {}
    while len(buf) >= 2:
        param_id = buf[0]
        param_len = buf[1]
        value = buf[2 : 2 + param_len]
        data[param_id] = bytes(value)
        buf = buf[2 + param_len :]
    return QueryResponse(uuid=base.uuid, id=base.id, status=base.status, payload=base.payload, data=data)


def _parse_protobuf(uuid: str, raw: bytes, message_cls: type) -> ProtobufResponse:
    feature_id = raw[0]
    action_id = raw[1]
    data = message_cls.FromString(bytes(raw[2:]))
    return ProtobufResponse(uuid=uuid, feature_id=feature_id, action_id=action_id, data=data)


def _encode_tlv_params(params: Optional[bytes | Iterable[bytes]]) -> bytes:
    if params is None:
        return b""
    if isinstance(params, (bytes, bytearray)):
        return bytes([len(params)]) + bytes(params)
    buf = bytearray()
    for param in params:
        param_bytes = bytes(param)
        buf.append(len(param_bytes))
        buf.extend(param_bytes)
    return bytes(buf)


def _normalize_uuid(value: str) -> str:
    return value.lower()


def _serial_suffix_from_advertisement(advertisement: AdvertisementData | None) -> Optional[str]:
    if not advertisement:
        return None
    for uuid, data in advertisement.service_data.items():
        normalized = _normalize_uuid(uuid)
        if normalized == GOPRO_BLE_SERVICE_UUID or normalized.startswith("0000fea6") or normalized == "fea6":
            suffix = _serial_tail_from_service_data(data)
            if suffix:
                return suffix
    return None


def _ap_mac_suffix_from_advertisement(advertisement: AdvertisementData | None) -> Optional[str]:
    if not advertisement:
        return None
    for uuid, data in advertisement.service_data.items():
        normalized = _normalize_uuid(uuid)
        if normalized == GOPRO_BLE_SERVICE_UUID or normalized.startswith("0000fea6") or normalized == "fea6":
            if len(data) >= 4:
                return data[:4].hex()
    return None


def _has_gopro_service(advertisement: AdvertisementData | None) -> bool:
    if not advertisement:
        return False
    if advertisement.manufacturer_data.get(GOPRO_COMPANY_ID):
        return True
    for uuid in advertisement.service_data.keys():
        normalized = _normalize_uuid(uuid)
        if normalized == GOPRO_BLE_SERVICE_UUID or normalized.startswith("0000fea6") or normalized == "fea6":
            return True
    return False


def _serial_tail_from_service_data(data: bytes) -> Optional[str]:
    if not data:
        return None
    if len(data) >= 12:
        tail = data[-8:]
    elif len(data) >= 8:
        tail = data[-4:]
    else:
        return None
    suffix = bytes(tail).decode("ascii", errors="ignore")
    return suffix or None


@dataclass
class BleAdvertisementStatus:
    schema_version: int
    processor_awake: bool
    wifi_ap_on: bool
    pairing: bool
    central_role: bool
    new_media: bool


def _gopro_advertisement_status(advertisement: AdvertisementData | None) -> Optional[BleAdvertisementStatus]:
    if not advertisement:
        return None
    data = advertisement.manufacturer_data.get(GOPRO_COMPANY_ID)
    if not data:
        return None
    raw = bytes(data)
    if len(raw) < 2:
        return None
    schema_version = raw[0]
    if schema_version not in (2, 3):
        return None
    status = raw[1]
    return BleAdvertisementStatus(
        schema_version=schema_version,
        processor_awake=bool(status & 0x01),
        wifi_ap_on=bool(status & 0x02),
        pairing=bool(status & 0x04),
        central_role=bool(status & 0x08),
        new_media=bool(status & 0x10),
    )


@dataclass
class BleScanResult:
    name: str
    address: str
    serial_suffix: Optional[str]
    ap_mac_suffix: Optional[str]
    pairing: Optional[bool] = None
    awake: Optional[bool] = None
    wifi_ap_on: Optional[bool] = None
    schema_version: Optional[int] = None


async def scan_ble_devices(timeout: float = 5.0) -> list[BleScanResult]:
    devices: dict[str, BLEDevice] = {}
    advertisements: dict[str, AdvertisementData] = {}

    def _callback(device: BLEDevice, advertisement: AdvertisementData) -> None:
        address = str(device.address)
        devices[address] = device
        advertisements[address] = advertisement

    try:
        discovered = await BleakScanner.discover(timeout=timeout, detection_callback=_callback)
    except TypeError:
        discovered = await BleakScanner.discover(timeout=timeout)
    for device in discovered:
        devices[str(device.address)] = device

    results: list[BleScanResult] = []
    for address, device in devices.items():
        address_str = str(address)
        advertisement = advertisements.get(address_str)
        name = (advertisement.local_name if advertisement else None) or device.name or ""
        serial_suffix = _serial_suffix_from_advertisement(advertisement)
        ap_mac_suffix = _ap_mac_suffix_from_advertisement(advertisement)
        status = _gopro_advertisement_status(advertisement)
        results.append(
            BleScanResult(
                name=name,
                address=address_str,
                serial_suffix=serial_suffix,
                ap_mac_suffix=ap_mac_suffix,
                pairing=status.pairing if status else None,
                awake=status.processor_awake if status else None,
                wifi_ap_on=status.wifi_ap_on if status else None,
                schema_version=status.schema_version if status else None,
            )
        )
    return results



def _name_matches(identifier: Optional[str], name: Optional[str]) -> bool:
    if not identifier or not name:
        return False
    return identifier in name


def _matches_identifier(
    identifier: Optional[str],
    camera_name: Optional[str],
    device: BLEDevice,
    advertisement: AdvertisementData | None,
) -> bool:
    if identifier:
        suffix = _serial_suffix_from_advertisement(advertisement)
        if suffix:
            if suffix == identifier or suffix.endswith(identifier):
                return True
        name = (advertisement.local_name if advertisement else None) or device.name
        if name and _name_matches(identifier, name):
            return True
    if camera_name:
        name = (advertisement.local_name if advertisement else None) or device.name
        if name and camera_name == name:
            return True
    return False


def _payload_packets(payload: bytes) -> Iterable[bytes]:
    if len(payload) <= 19:
        yield bytes([len(payload)]) + payload
        return

    length = len(payload)
    if length >= (2**13 - 1):
        raise ValueError("Payload too large for BLE packetization")

    header = bytearray((length | 0x2000).to_bytes(2, "big", signed=False))
    max_packet = 20
    is_first = True
    idx = 0
    while idx < length:
        if is_first:
            packet = bytearray(header)
            is_first = False
        else:
            packet = bytearray([0x80])
        chunk_size = min(max_packet - len(packet), length - idx)
        packet.extend(payload[idx : idx + chunk_size])
        idx += chunk_size
        yield bytes(packet)


class GoProBleConnection:
    def __init__(
        self,
        serial: str,
        identifier: Optional[str] = None,
        camera_name: Optional[str] = None,
        address: Optional[str] = None,
        *,
        phone_name: str = DEFAULT_PHONE_NAME,
        auto_finish_pairing: bool = True,
        auto_third_party: bool = True,
        enforce_api_version: bool = True,
        use_cached: Optional[bool] = None,
    ) -> None:
        self.serial = serial
        self.identifier = identifier
        self.camera_name = camera_name
        self.address = address
        self.phone_name = phone_name
        self.auto_finish_pairing = auto_finish_pairing
        self.auto_third_party = auto_third_party
        self.enforce_api_version = enforce_api_version
        self.use_cached = use_cached
        self.client: Optional[BleakClient] = None
        self._accumulators: dict[str, _ResponseAccumulator] = {}
        self._waiters: list[_ResponseWaiter] = []
        self._lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._keep_alive_task: Optional[asyncio.Task] = None
        self.status_values: dict[int, bytes] = {}
        self.api_version: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        return bool(self.client and self.client.is_connected)

    async def connect(self, timeout: float = 30.0, retries: int = 2) -> None:
        async with self._lock:
            if self.is_connected:
                return
            last_exc: Exception | None = None
            for attempt in range(retries):
                client: BleakClient | None = None
                if self.address:
                    try:
                        device: BLEDevice | str = self.address
                        logger.info(
                            "Connecting to GoPro BLE device %s (attempt %s/%s)",
                            device,
                            attempt + 1,
                            retries,
                        )
                        client = _new_bleak_client(device, timeout=timeout, use_cached=self.use_cached)
                        await client.connect(timeout=timeout)
                        break
                    except Exception as exc:
                        last_exc = exc
                        try:
                            if client:
                                await client.disconnect()
                        except Exception:
                            pass
                        logger.debug("Direct BLE address connect failed: %s", exc)

                try:
                    device = await self._scan_for_device()
                    logger.info(
                        "Connecting to GoPro BLE device %s (attempt %s/%s)",
                        device,
                        attempt + 1,
                        retries,
                    )
                    client = _new_bleak_client(device, timeout=timeout, use_cached=self.use_cached)
                    await client.connect(timeout=timeout)
                    break
                except Exception as scan_exc:
                    last_exc = scan_exc
                    try:
                        if client:
                            await client.disconnect()
                    except Exception:
                        pass
                    if attempt + 1 == retries:
                        raise
                    await asyncio.sleep(1.0)
                    continue
            else:
                if last_exc:
                    raise last_exc
                raise RuntimeError("BLE connect failed without exception")
            paired: Optional[bool] = None
            try:
                paired = await client.pair()
            except NotImplementedError:
                paired = None
            except Exception as exc:
                logger.warning("BLE pairing request failed: %s", exc)
                paired = False

            await self._ensure_services(client)
            if not paired:
                await self._trigger_pairing_via_encrypted_read(client)
            await self._enable_notifications(client)
            self.client = client
            self._stop_event.clear()
            try:
                if self.auto_finish_pairing:
                    await self._maybe_finish_pairing()
                if self.auto_third_party:
                    try:
                        await self.set_third_party_client_info()
                    except Exception as exc:
                        logger.debug("BLE third-party info failed: %s", exc)
                await self.wait_ready()
                if self.enforce_api_version:
                    await self.ensure_open_gopro_api_version()
                await self.start_keep_alive()
                try:
                    await self.register_status_updates(list(DEFAULT_STATUS_IDS))
                except Exception as exc:
                    logger.debug("Status registration failed: %s", exc)
            except Exception:
                try:
                    await client.disconnect()
                finally:
                    self.client = None
                raise

    async def disconnect(self) -> None:
        async with self._lock:
            self._stop_event.set()
            if self._keep_alive_task:
                self._keep_alive_task.cancel()
                self._keep_alive_task = None
            if self.client:
                await self.client.disconnect()
                self.client = None

    async def finish_pairing(self, phone_name: str) -> bool:
        message = network_management_pb2.RequestPairingFinish(
            result=network_management_pb2.EnumPairingFinishState.SUCCESS,
            phoneName=phone_name,
        )
        response = await self._send_protobuf(
            NETWORK_MGMT_REQ_UUID,
            feature_id=0x03,
            action_id=0x01,
            message=message,
            expected_action_id=0x81,
        )
        return int(response.data.result) == int(response_generic_pb2.EnumResultGeneric.RESULT_SUCCESS)

    async def claim_control(self) -> bool:
        message = set_camera_control_status_pb2.RequestSetCameraControlStatus(
            camera_control_status=set_camera_control_status_pb2.EnumCameraControlStatus.CAMERA_EXTERNAL_CONTROL
        )
        response = await self._send_protobuf(
            COMMAND_REQ_UUID,
            feature_id=0xF1,
            action_id=0x69,
            message=message,
            expected_action_id=0xE9,
        )
        return int(response.data.result) == int(response_generic_pb2.EnumResultGeneric.RESULT_SUCCESS)

    async def set_camera_name(self, name: str) -> bool:
        message = camera_control_pb2.RequestSetCameraName(name=name)
        response = await self._send_protobuf(
            COMMAND_REQ_UUID,
            feature_id=0xF1,
            action_id=0x62,
            message=message,
            expected_action_id=0xE2,
        )
        return int(response.data.result) == int(response_generic_pb2.EnumResultGeneric.RESULT_SUCCESS)

    async def set_third_party_client_info(self) -> bool:
        response = await self._send_tlv_command(SET_THIRD_PARTY_CLIENT_INFO, None, use_settings=False)
        return response.status == 0x00

    async def sleep_camera(self) -> bool:
        response = await self._send_tlv_command(SLEEP_COMMAND_ID, None, use_settings=False, timeout=5.0)
        return response.status == 0x00

    async def set_ap_control(self, mode: int) -> bool:
        if mode not in (0, 1, 2):
            raise ValueError("mode must be 0 (disable), 1 (enable), or 2 (bounce)")
        response = await self._send_tlv_command(
            SET_AP_CONTROL_COMMAND_ID,
            bytes([mode]),
            use_settings=False,
            timeout=5.0,
        )
        return response.status == 0x00

    async def get_open_gopro_api_version(self) -> Optional[str]:
        response = await self._send_tlv_command(GET_OPEN_GOPRO_API_VERSION, None, use_settings=False)
        if response.status != 0x00 or len(response.payload) < 4:
            return None
        major = response.payload[1]
        minor = response.payload[3]
        return f"{major}.{minor}"

    async def ensure_open_gopro_api_version(
        self,
        expected_version: str = SUPPORTED_OPEN_GOPRO_API_VERSION,
        retries: int = 3,
    ) -> str:
        last_version: Optional[str] = None
        for attempt in range(retries):
            last_version = await self.get_open_gopro_api_version()
            if last_version:
                break
            await asyncio.sleep(0.5)
        if not last_version:
            raise RuntimeError("Open GoPro API version unavailable")
        if last_version != expected_version:
            raise RuntimeError(
                f"Unsupported Open GoPro API version {last_version} (expected {expected_version})"
            )
        self.api_version = last_version
        return last_version

    async def _maybe_finish_pairing(self) -> bool:
        try:
            status = await self.get_status_values([PAIRING_STATE_STATUS_ID])
        except Exception as exc:
            logger.debug("BLE pairing state read failed: %s", exc)
            try:
                return await self.finish_pairing(self.phone_name)
            except Exception as finish_exc:
                logger.debug("BLE pairing finish failed: %s", finish_exc)
                return False

        value = status.get(PAIRING_STATE_STATUS_ID)
        if value:
            pairing_state = value[0]
            if pairing_state in (1, 2, 3):
                return await self.finish_pairing(self.phone_name)
            return True

        return await self.finish_pairing(self.phone_name)
    async def wait_ready(self, timeout: float = 10.0) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            response = await self._send_tlv_command(0x3C, None, use_settings=False)
            if response.status == 0x00:
                return True
            await asyncio.sleep(0.5)
        return False

    async def start_keep_alive(self) -> None:
        if self._keep_alive_task and not self._keep_alive_task.done():
            return
        self._keep_alive_task = asyncio.create_task(self._keep_alive_loop())

    async def register_status_updates(self, status_ids: list[int]) -> bool:
        payload = bytes([QUERY_REGISTER_STATUS, *status_ids])
        waiter = self._register_waiter(
            lambda r: isinstance(r, QueryResponse) and r.uuid == QUERY_RSP_UUID and r.id == QUERY_REGISTER_STATUS
        )
        await self._write_payload(QUERY_REQ_UUID, payload)
        response = await self._await_waiter(waiter, 10.0)
        if isinstance(response, QueryResponse) and response.status == 0x00:
            self.status_values.update(response.data)
            return True
        return False

    async def get_status_values(self, status_ids: list[int]) -> dict[int, bytes]:
        payload = bytes([QUERY_GET_STATUS, *status_ids])
        waiter = self._register_waiter(
            lambda r: isinstance(r, QueryResponse) and r.uuid == QUERY_RSP_UUID and r.id == QUERY_GET_STATUS
        )
        await self._write_payload(QUERY_REQ_UUID, payload)
        response = await self._await_waiter(waiter, 10.0)
        if isinstance(response, QueryResponse):
            self.status_values.update(response.data)
            return response.data
        return {}

    async def connect_wifi(self, ssid: str, password: str, scan: bool = True) -> bool:
        if not ssid:
            return False
        if scan:
            scan_id = await self._scan_networks()
            entry = await self._find_scan_entry(scan_id, ssid)
            if not entry:
                logger.warning("SSID %s not found in scan results", ssid)
                return False
            if entry.scan_entry_flags & network_management_pb2.EnumScanEntryFlags.SCAN_FLAG_CONFIGURED:
                message = network_management_pb2.RequestConnect(ssid=ssid)
                action_id = 0x04
                expected_action = 0x84
            else:
                message = network_management_pb2.RequestConnectNew(
                    ssid=ssid,
                    password=password,
                    bypass_eula_check=True,
                )
                action_id = 0x05
                expected_action = 0x85
        else:
            message = network_management_pb2.RequestConnectNew(
                ssid=ssid,
                password=password,
                bypass_eula_check=True,
            )
            action_id = 0x05
            expected_action = 0x85

        provisioning_waiter = self._register_waiter(
            lambda r: isinstance(r, ProtobufResponse)
            and r.uuid == NETWORK_MGMT_RSP_UUID
            and r.feature_id == 0x02
            and r.action_id == 0x0C
        )
        response = await self._send_protobuf(
            NETWORK_MGMT_REQ_UUID,
            feature_id=0x02,
            action_id=action_id,
            message=message,
            expected_action_id=expected_action,
            timeout=10.0,
        )
        if int(response.data.result) != int(response_generic_pb2.EnumResultGeneric.RESULT_SUCCESS):
            if provisioning_waiter in self._waiters:
                self._waiters.remove(provisioning_waiter)
            return False

        timeout_seconds = getattr(response.data, "timeout_seconds", 30)
        deadline = asyncio.get_running_loop().time() + float(timeout_seconds)
        while asyncio.get_running_loop().time() < deadline:
            remaining = max(1.0, deadline - asyncio.get_running_loop().time())
            notification = await self._await_waiter(provisioning_waiter, remaining)
            if isinstance(notification, ProtobufResponse):
                provisioning_state = notification.data.provisioning_state
                if provisioning_state in (
                    network_management_pb2.EnumProvisioning.PROVISIONING_SUCCESS_NEW_AP,
                    network_management_pb2.EnumProvisioning.PROVISIONING_SUCCESS_OLD_AP,
                ):
                    return True
                if provisioning_state != network_management_pb2.EnumProvisioning.PROVISIONING_STARTED:
                    return False
                provisioning_waiter = self._register_waiter(
                    lambda r: isinstance(r, ProtobufResponse)
                    and r.uuid == NETWORK_MGMT_RSP_UUID
                    and r.feature_id == 0x02
                    and r.action_id == 0x0C
                )
        return False

    async def release_network(self) -> bool:
        """Disconnect from STA WiFi network so the camera returns to AP mode."""
        if not self.client:
            raise RuntimeError("BLE client not connected")

        message = network_management_pb2.RequestReleaseNetwork()
        response = await self._send_protobuf(
            COMMAND_REQ_UUID,
            feature_id=0xF1,
            action_id=0x78,  # RELEASE_NETWORK
            message=message,
            expected_action_id=0xF8,  # RELEASE_NETWORK_RSP
        )
        if isinstance(response.data, response_generic_pb2.ResponseGeneric):
            return response.data.result == response_generic_pb2.EnumResultGeneric.RESULT_SUCCESS
        return False

    async def read_wifi_ap_credentials(self) -> tuple[str, str]:
        if not self.client:
            raise RuntimeError("BLE client not connected")
        ssid = (await self.client.read_gatt_char(WIFI_AP_SSID_UUID)).decode(errors="ignore")
        password = (await self.client.read_gatt_char(WIFI_AP_PASSWORD_UUID)).decode(errors="ignore")
        return ssid, password

    # =========================================================================
    # COHN (Camera On Home Network) Operations
    # =========================================================================

    async def get_cohn_status(self, register_notifications: bool = False) -> Optional[cohn_pb2.NotifyCOHNStatus]:
        """Get COHN status including credentials (username, password, IP)."""
        if not self.client:
            raise RuntimeError("BLE client not connected")

        message = cohn_pb2.RequestGetCOHNStatus(register_cohn_status=register_notifications)
        response = await self._send_protobuf(
            QUERY_REQ_UUID,
            feature_id=0xF5,
            action_id=0x6F,  # REQUEST_GET_COHN_STATUS
            message=message,
            expected_action_id=0xEF,  # RESPONSE_GET_COHN_STATUS
        )
        if isinstance(response.data, cohn_pb2.NotifyCOHNStatus):
            return response.data
        return None

    async def create_cohn_cert(self, override: bool = True) -> bool:
        """Create COHN certificate on the camera."""
        if not self.client:
            raise RuntimeError("BLE client not connected")

        message = cohn_pb2.RequestCreateCOHNCert(override=override)
        response = await self._send_protobuf(
            COMMAND_REQ_UUID,
            feature_id=0xF1,
            action_id=0x67,  # REQUEST_CREATE_COHN_CERT
            message=message,
            expected_action_id=0xE7,  # RESPONSE_CREATE_COHN_CERT
            timeout=30.0,  # Cert creation can take time
        )
        if isinstance(response.data, response_generic_pb2.ResponseGeneric):
            return response.data.result == response_generic_pb2.EnumResultGeneric.RESULT_SUCCESS
        return False

    async def enable_cohn(self, enabled: bool = True) -> bool:
        """Enable or disable COHN on the camera."""
        if not self.client:
            raise RuntimeError("BLE client not connected")

        message = cohn_pb2.RequestSetCOHNSetting(cohn_active=enabled)
        response = await self._send_protobuf(
            COMMAND_REQ_UUID,
            feature_id=0xF1,
            action_id=0x65,  # REQUEST_COHN_SETTING
            message=message,
            expected_action_id=0xE5,  # RESPONSE_COHN_SETTING
        )
        if isinstance(response.data, response_generic_pb2.ResponseGeneric):
            return response.data.result == response_generic_pb2.EnumResultGeneric.RESULT_SUCCESS
        return False

    async def get_cohn_cert(self) -> Optional[str]:
        """Get the COHN SSL certificate from the camera."""
        if not self.client:
            raise RuntimeError("BLE client not connected")

        message = cohn_pb2.RequestCOHNCert()
        response = await self._send_protobuf(
            QUERY_REQ_UUID,
            feature_id=0xF5,
            action_id=0x6E,  # REQUEST_GET_COHN_CERT
            message=message,
            expected_action_id=0xEE,  # RESPONSE_GET_COHN_CERT
        )
        if isinstance(response.data, cohn_pb2.ResponseCOHNCert):
            if response.data.result == response_generic_pb2.EnumResultGeneric.RESULT_SUCCESS:
                return response.data.cert
        return None

    async def wait_for_cohn_network_connected(
        self,
        timeout: float = 60.0,
    ) -> Optional[cohn_pb2.NotifyCOHNStatus]:
        """Wait until COHN reports NetworkConnected, returning the latest status."""
        if not self.client:
            raise RuntimeError("BLE client not connected")

        status = await self.get_cohn_status(register_notifications=True)
        if status and status.state == cohn_pb2.EnumCOHNNetworkState.COHN_STATE_NetworkConnected:
            return status

        deadline = asyncio.get_running_loop().time() + timeout
        last_status = status
        while asyncio.get_running_loop().time() < deadline:
            remaining = max(1.0, deadline - asyncio.get_running_loop().time())
            waiter = self._register_waiter(
                lambda r: isinstance(r, ProtobufResponse)
                and r.uuid == QUERY_RSP_UUID
                and r.feature_id == 0xF5
                and r.action_id == 0xEF
            )
            notification = await self._await_waiter(waiter, remaining)
            if isinstance(notification, ProtobufResponse) and isinstance(
                notification.data, cohn_pb2.NotifyCOHNStatus
            ):
                last_status = notification.data
                if last_status.state == cohn_pb2.EnumCOHNNetworkState.COHN_STATE_NetworkConnected:
                    return last_status
        return last_status

    async def _trigger_pairing_via_encrypted_read(self, client: BleakClient) -> None:
        try:
            await asyncio.wait_for(
                client.read_gatt_char(WIFI_AP_PASSWORD_UUID),
                timeout=PAIRING_TRIGGER_TIMEOUT_SECONDS,
            )
            logger.info("Triggered BLE pairing by reading encrypted characteristic")
        except Exception as exc:
            logger.debug("BLE pairing trigger read failed: %s", exc)

    async def _scan_for_device(self, timeout: float = 5.0, retries: int = 6) -> BLEDevice:
        logger.info("Scanning for GoPro BLE advertisements")
        for attempt in range(retries):
            devices: dict[str, BLEDevice] = {}
            advertisements: dict[str, AdvertisementData] = {}

            def _callback(device: BLEDevice, advertisement: AdvertisementData) -> None:
                devices[device.address] = device
                advertisements[device.address] = advertisement

            try:
                discovered = await BleakScanner.discover(timeout=timeout, detection_callback=_callback)
            except TypeError:
                discovered = await BleakScanner.discover(timeout=timeout)
            for device in discovered:
                devices[device.address] = device

            if self.address:
                device = devices.get(self.address)
                if device:
                    return device
                logger.warning("BLE scan attempt %s did not find target address %s", attempt + 1, self.address)

            for address, device in devices.items():
                advertisement = advertisements.get(address)
                if self.identifier or self.camera_name:
                    if _matches_identifier(self.identifier, self.camera_name, device, advertisement):
                        return device
                elif not self.address and _has_gopro_service(advertisement):
                    return device

            logger.warning("BLE scan attempt %s did not find target device", attempt + 1)
        raise RuntimeError("Unable to locate GoPro BLE device")

    async def _ensure_services(self, client: BleakClient) -> None:
        if hasattr(client, "get_services"):
            await client.get_services()
            return
        backend = getattr(client, "_backend", None)
        if backend and hasattr(backend, "_get_services"):
            await backend._get_services()
            return
        _ = client.services

    async def _enable_notifications(self, client: BleakClient) -> None:
        for service in client.services:
            for char in service.characteristics:
                if "notify" in char.properties or "indicate" in char.properties:
                    if _normalize_uuid(char.uuid) in NOTIFY_RESPONSE_UUIDS:
                        await client.start_notify(char, self._notification_handler)

    async def _notification_handler(self, characteristic: BleakGATTCharacteristic, data: bytearray) -> None:
        uuid = _normalize_uuid(characteristic.uuid)
        accumulator = self._accumulators.get(uuid)
        if not accumulator:
            accumulator = _ResponseAccumulator(uuid)
            self._accumulators[uuid] = accumulator
        accumulator.accumulate(data)
        if accumulator.is_received:
            parsed = self._parse_response(uuid, bytes(accumulator.raw_bytes))
            self._dispatch_response(parsed)
            accumulator.reset()

    def _parse_response(self, uuid: str, raw: bytes) -> object:
        if len(raw) >= 2:
            key = (raw[0], raw[1])
            message_cls = PROTOBUF_RESPONSE_MAP.get(key)
            if message_cls:
                return _parse_protobuf(uuid, raw, message_cls)
        if uuid == QUERY_RSP_UUID:
            return _parse_query(uuid, raw)
        return _parse_tlv(uuid, raw)

    def _dispatch_response(self, response: object) -> None:
        for waiter in list(self._waiters):
            if waiter.predicate(response):
                if not waiter.future.done():
                    waiter.future.set_result(response)
                self._waiters.remove(waiter)
                return

        if isinstance(response, QueryResponse) and response.id in {QUERY_GET_STATUS, QUERY_ASYNC_STATUS}:
            self.status_values.update(response.data)

    async def _wait_for_response(self, predicate: Callable[[object], bool], timeout: float = 10.0) -> object:
        waiter = self._register_waiter(predicate)
        return await self._await_waiter(waiter, timeout)

    def _register_waiter(self, predicate: Callable[[object], bool]) -> _ResponseWaiter:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        waiter = _ResponseWaiter(predicate=predicate, future=future)
        self._waiters.append(waiter)
        return waiter

    async def _await_waiter(self, waiter: _ResponseWaiter, timeout: float) -> object:
        try:
            return await asyncio.wait_for(waiter.future, timeout=timeout)
        finally:
            if waiter in self._waiters:
                self._waiters.remove(waiter)

    async def _send_tlv_command(
        self,
        command_id: int,
        params: Optional[bytes | Iterable[bytes]],
        use_settings: bool,
        timeout: float = 10.0,
    ) -> TlvResponse:
        payload = bytes([command_id]) + _encode_tlv_params(params)
        request_uuid = SETTINGS_REQ_UUID if use_settings else COMMAND_REQ_UUID
        response_uuid = SETTINGS_RSP_UUID if use_settings else COMMAND_RSP_UUID
        waiter = self._register_waiter(
            lambda r: isinstance(r, TlvResponse) and r.uuid == response_uuid and r.id == command_id
        )
        await self._write_payload(request_uuid, payload, timeout=timeout)
        response = await self._await_waiter(waiter, timeout)
        return response  # type: ignore[return-value]

    async def _send_protobuf(
        self,
        request_uuid: str,
        feature_id: int,
        action_id: int,
        message: object,
        expected_action_id: int,
        timeout: float = 20.0,
    ) -> ProtobufResponse:
        payload = bytes([feature_id, action_id]) + message.SerializeToString()
        waiter = self._register_waiter(
            lambda r: isinstance(r, ProtobufResponse)
            and r.feature_id == feature_id
            and r.action_id == expected_action_id
        )
        await self._write_payload(request_uuid, payload, timeout=timeout)
        response = await self._await_waiter(waiter, timeout)
        return response  # type: ignore[return-value]

    async def _write_payload(self, uuid: str, payload: bytes, *, timeout: float | None = None) -> None:
        if not self.client:
            raise RuntimeError("BLE client not connected")
        async with self._send_lock:
            for packet in _payload_packets(payload):
                if timeout is None:
                    await self.client.write_gatt_char(uuid, packet, response=True)
                else:
                    await asyncio.wait_for(
                        self.client.write_gatt_char(uuid, packet, response=True),
                        timeout=timeout,
                    )

    async def _keep_alive_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._send_tlv_command(
                    KEEP_ALIVE_COMMAND_ID, bytes([KEEP_ALIVE_VALUE]), use_settings=True, timeout=5.0
                )
            except Exception as exc:
                logger.debug("Keep-alive failed: %s", exc)
            await asyncio.sleep(KEEP_ALIVE_INTERVAL_SECONDS)

    async def _scan_networks(self) -> int:
        message = network_management_pb2.RequestStartScan()
        notification_waiter = self._register_waiter(
            lambda r: isinstance(r, ProtobufResponse)
            and r.uuid == NETWORK_MGMT_RSP_UUID
            and r.feature_id == 0x02
            and r.action_id == 0x0B
        )
        response = await self._send_protobuf(
            NETWORK_MGMT_REQ_UUID,
            feature_id=0x02,
            action_id=0x02,
            message=message,
            expected_action_id=0x82,
            timeout=10.0,
        )
        if int(response.data.result) != int(response_generic_pb2.EnumResultGeneric.RESULT_SUCCESS):
            if notification_waiter in self._waiters:
                self._waiters.remove(notification_waiter)
            raise RuntimeError("Wi-Fi scan request rejected")

        while True:
            notification = await self._await_waiter(notification_waiter, 20.0)
            if isinstance(notification, ProtobufResponse):
                if notification.data.scanning_state == network_management_pb2.EnumScanning.SCANNING_SUCCESS:
                    return notification.data.scan_id
                notification_waiter = self._register_waiter(
                    lambda r: isinstance(r, ProtobufResponse)
                    and r.uuid == NETWORK_MGMT_RSP_UUID
                    and r.feature_id == 0x02
                    and r.action_id == 0x0B
                )

    async def _find_scan_entry(
        self, scan_id: int, ssid: str
    ) -> Optional[network_management_pb2.ResponseGetApEntries.ScanEntry]:
        message = network_management_pb2.RequestGetApEntries(start_index=0, max_entries=100, scan_id=scan_id)
        response = await self._send_protobuf(
            NETWORK_MGMT_REQ_UUID,
            feature_id=0x02,
            action_id=0x03,
            message=message,
            expected_action_id=0x83,
            timeout=10.0,
        )
        if int(response.data.result) != int(response_generic_pb2.EnumResultGeneric.RESULT_SUCCESS):
            return None
        for entry in response.data.entries:
            if entry.ssid == ssid:
                return entry
        return None


class GoProBleManager:
    def __init__(self) -> None:
        self._connections: dict[str, GoProBleConnection] = {}

    async def ensure_connection(
        self, serial: str, identifier: Optional[str] = None, camera_name: Optional[str] = None
    ) -> GoProBleConnection:
        existing = self._connections.get(serial)
        if existing and existing.is_connected:
            return existing
        connection = GoProBleConnection(serial=serial, identifier=identifier, camera_name=camera_name)
        await connection.connect()
        self._connections[serial] = connection
        return connection

    async def disconnect(self, serial: str) -> None:
        connection = self._connections.pop(serial, None)
        if connection:
            await connection.disconnect()

    async def shutdown(self) -> None:
        for serial in list(self._connections.keys()):
            await self.disconnect(serial)
