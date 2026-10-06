import sys
import types
import unittest


def _install_bleak_stub() -> None:
    try:
        import bleak  # noqa: F401
        return
    except Exception:
        pass

    if "bleak" in sys.modules:
        return

    bleak_module = types.ModuleType("bleak")

    class DummyBleakClient:
        def __init__(self, *args, **kwargs):
            pass

    class DummyBleakScanner:
        @staticmethod
        async def discover(*args, **kwargs):
            return []

    bleak_module.BleakClient = DummyBleakClient
    bleak_module.BleakScanner = DummyBleakScanner
    sys.modules["bleak"] = bleak_module

    backends = types.ModuleType("bleak.backends")
    characteristic_mod = types.ModuleType("bleak.backends.characteristic")
    device_mod = types.ModuleType("bleak.backends.device")
    scanner_mod = types.ModuleType("bleak.backends.scanner")

    class DummyBleakGATTCharacteristic:
        def __init__(self, *args, **kwargs):
            self.uuid = ""

    class DummyBLEDevice:
        def __init__(self, *args, **kwargs):
            self.name = ""
            self.address = ""

    class DummyAdvertisementData:
        def __init__(self, *args, **kwargs):
            self.service_data = {}
            self.manufacturer_data = {}
            self.local_name = ""

    characteristic_mod.BleakGATTCharacteristic = DummyBleakGATTCharacteristic
    device_mod.BLEDevice = DummyBLEDevice
    scanner_mod.AdvertisementData = DummyAdvertisementData

    sys.modules["bleak.backends"] = backends
    sys.modules["bleak.backends.characteristic"] = characteristic_mod
    sys.modules["bleak.backends.device"] = device_mod
    sys.modules["bleak.backends.scanner"] = scanner_mod


def _install_ble_proto_stub() -> None:
    try:
        import google.protobuf  # noqa: F401
        return
    except Exception:
        pass

    if "gopro.ble_proto" in sys.modules:
        return

    proto_pkg = types.ModuleType("gopro.ble_proto")
    sys.modules["gopro.ble_proto"] = proto_pkg

    def _make_module(name):
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        return mod

    class DummyProto:
        def __init__(self, *args, **kwargs):
            pass

        def SerializeToString(self):
            return b""

        @classmethod
        def FromString(cls, _data):
            return cls()

    response_generic = _make_module("gopro.ble_proto.response_generic_pb2")
    response_generic.ResponseGeneric = DummyProto

    camera_control = _make_module("gopro.ble_proto.camera_control_pb2")
    camera_control.RequestSetCameraName = DummyProto

    set_control = _make_module("gopro.ble_proto.set_camera_control_status_pb2")

    class EnumCameraControlStatus:
        CAMERA_EXTERNAL_CONTROL = 1

    set_control.EnumCameraControlStatus = EnumCameraControlStatus
    set_control.RequestSetCameraControlStatus = DummyProto

    network_mgmt = _make_module("gopro.ble_proto.network_management_pb2")

    class EnumPairingFinishState:
        SUCCESS = 1

    class EnumResultGeneric:
        RESULT_SUCCESS = 1

    class EnumScanEntryFlags:
        SCAN_FLAG_CONFIGURED = 1

    class EnumProvisioning:
        PROVISIONING_SUCCESS_NEW_AP = 1
        PROVISIONING_SUCCESS_OLD_AP = 2
        PROVISIONING_STARTED = 3

    class EnumScanning:
        SCANNING_SUCCESS = 1

    network_mgmt.EnumPairingFinishState = EnumPairingFinishState
    network_mgmt.EnumResultGeneric = EnumResultGeneric
    network_mgmt.EnumScanEntryFlags = EnumScanEntryFlags
    network_mgmt.EnumProvisioning = EnumProvisioning
    network_mgmt.EnumScanning = EnumScanning
    network_mgmt.RequestPairingFinish = DummyProto
    network_mgmt.RequestStartScan = DummyProto
    network_mgmt.RequestGetApEntries = DummyProto
    network_mgmt.RequestConnect = DummyProto
    network_mgmt.RequestConnectNew = DummyProto
    network_mgmt.ResponseStartScanning = DummyProto
    network_mgmt.NotifStartScanning = DummyProto
    network_mgmt.ResponseGetApEntries = DummyProto
    network_mgmt.ResponseConnect = DummyProto
    network_mgmt.ResponseConnectNew = DummyProto
    network_mgmt.NotifProvisioningState = DummyProto

    cohn_mod = _make_module("gopro.ble_proto.cohn_pb2")

    class EnumCOHNNetworkState:
        COHN_STATE_NetworkConnected = 27

    cohn_mod.EnumCOHNNetworkState = EnumCOHNNetworkState
    cohn_mod.ResponseCOHNCert = DummyProto
    cohn_mod.NotifyCOHNStatus = DummyProto


_install_bleak_stub()
_install_ble_proto_stub()

from gopro import ble


class DummyAdvertisement:
    def __init__(self, service_data, local_name=None, manufacturer_data=None):
        self.service_data = service_data
        self.local_name = local_name
        self.manufacturer_data = manufacturer_data or {}


class DummyDevice:
    def __init__(self, name=None):
        self.name = name


class BleProtocolTests(unittest.TestCase):
    def test_payload_packets_continuation_headers(self):
        payload = bytes(range(50))
        packets = list(ble._payload_packets(payload))

        self.assertEqual(len(packets), 3)
        header0 = packets[0][0]
        header1 = packets[0][1]
        self.assertEqual(header0 & 0xE0, 0x20)
        length = ((header0 & 0x1F) << 8) | header1
        self.assertEqual(length, len(payload))

        self.assertEqual(packets[1][0], 0x80)
        self.assertEqual(packets[2][0], 0x80)

    def test_response_accumulator_reassembles(self):
        payload = bytes(range(50))
        packets = list(ble._payload_packets(payload))
        acc = ble._ResponseAccumulator("test")
        for packet in packets:
            acc.accumulate(packet)
        self.assertTrue(acc.is_received)
        self.assertEqual(bytes(acc.raw_bytes), payload)

    def test_response_accumulator_ignores_continuation_low_bits(self):
        payload = bytes(range(50))
        packets = list(ble._payload_packets(payload))
        packets[1] = bytes([0x8F]) + packets[1][1:]
        packets[2] = bytes([0x81]) + packets[2][1:]
        acc = ble._ResponseAccumulator("test")
        for packet in packets:
            acc.accumulate(packet)
        self.assertTrue(acc.is_received)
        self.assertEqual(bytes(acc.raw_bytes), payload)

    def test_serial_suffix_parsing_v2(self):
        uuid = "0000FEA6-0000-1000-8000-00805F9B34FB"
        data = b"\x01\x02\x03\x04" + b"1234"
        adv = DummyAdvertisement({uuid: data})
        self.assertEqual(ble._serial_suffix_from_advertisement(adv), "1234")
        self.assertEqual(ble._ap_mac_suffix_from_advertisement(adv), "01020304")

    def test_serial_suffix_parsing_v3(self):
        uuid = "0000FEA6-0000-1000-8000-00805F9B34FB"
        data = b"\xaa\xbb\xcc\xdd" + b"ABCDEFGH"
        adv = DummyAdvertisement({uuid: data})
        self.assertEqual(ble._serial_suffix_from_advertisement(adv), "ABCDEFGH")

        device = DummyDevice()
        self.assertTrue(ble._matches_identifier("EFGH", None, device, adv))

    def test_encode_tlv_params(self):
        self.assertEqual(ble._encode_tlv_params(None), b"")
        self.assertEqual(ble._encode_tlv_params(b"\x42"), b"\x01\x42")
        self.assertEqual(ble._encode_tlv_params([b"\x01", b"\x02\x03"]), b"\x01\x01\x02\x02\x03")

    def test_has_gopro_service(self):
        uuid = "0000FEA6-0000-1000-8000-00805F9B34FB"
        adv = DummyAdvertisement({uuid: b"\x01\x02\x03\x04\x35\x36\x37\x38"})
        self.assertTrue(ble._has_gopro_service(adv))

        adv_none = DummyAdvertisement({})
        self.assertFalse(ble._has_gopro_service(adv_none))

    def test_gopro_advertisement_status_parses_bits(self):
        manufacturer_data = {0x02F2: bytes([2, 0x1F])}
        adv = DummyAdvertisement({}, manufacturer_data=manufacturer_data)
        status = ble._gopro_advertisement_status(adv)
        self.assertIsNotNone(status)
        self.assertEqual(status.schema_version, 2)
        self.assertTrue(status.processor_awake)
        self.assertTrue(status.wifi_ap_on)
        self.assertTrue(status.pairing)
        self.assertTrue(status.central_role)
        self.assertTrue(status.new_media)

    def test_gopro_advertisement_status_rejects_unknown_schema(self):
        manufacturer_data = {0x02F2: bytes([1, 0x00])}
        adv = DummyAdvertisement({}, manufacturer_data=manufacturer_data)
        self.assertIsNone(ble._gopro_advertisement_status(adv))


if __name__ == "__main__":
    unittest.main()
