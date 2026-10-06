import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from gopro import ble_daemon
from gopro.state import CameraInventoryRecord, Inventory


class DummyBleConnection:
    def __init__(self, serial: str, identifier: str, camera_name=None, address=None, **_kwargs):
        self.serial = serial
        self.identifier = identifier
        self.camera_name = camera_name
        self.address = address
        self._connected = False
        self.last_ap_mode = None
        self.last_camera_name_set = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self, *args, **kwargs) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def finish_pairing(self, _phone_name: str) -> bool:
        return True

    async def claim_control(self) -> bool:
        return True

    async def get_status_values(self, status_ids: list[int]):
        return {int(s): b"\x01" for s in status_ids}

    async def connect_wifi(self, _ssid: str, _password: str) -> bool:
        return True

    async def ensure_open_gopro_api_version(self, *args, **kwargs) -> str:
        return "2.0"

    async def sleep_camera(self) -> bool:
        return True

    async def set_ap_control(self, mode: int) -> bool:
        self.last_ap_mode = mode
        return True

    async def release_network(self) -> bool:
        return True

    async def set_camera_name(self, name: str) -> bool:
        self.camera_name = name
        self.last_camera_name_set = name
        return True


class DummyScanResult:
    def __init__(
        self,
        serial_suffix: str,
        *,
        name: str | None = None,
        address: str | None = None,
        pairing: bool = True,
        awake: bool = True,
        wifi_ap_on: bool | None = None,
    ) -> None:
        self.serial_suffix = serial_suffix
        self.name = name
        self.address = address
        self.pairing = pairing
        self.awake = awake
        self.wifi_ap_on = wifi_ap_on


class BleDaemonTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "ble.sock"
            pid_path = Path(tmp) / "ble.pid"
            daemon = ble_daemon.BleDaemon(socket_path=socket_path, pid_path=pid_path, maintenance_interval=0.01)

            with patch.object(ble_daemon, "GoProBleConnection", DummyBleConnection), patch.object(
                ble_daemon,
                "scan_ble_devices",
                new=AsyncMock(return_value=[DummyScanResult("1234", address="AA:BB:CC")]),
            ):
                result = await daemon._cmd_connect(
                    {
                        "camera": "TEST1234",
                        "finish_pairing": True,
                        "claim_control": True,
                        "phone_name": "PhoneName",
                    }
                )

            self.assertEqual(result["serial"], "TEST1234")
            self.assertEqual(result["identifier"], "1234")
            self.assertTrue(result["connected"])

            status = daemon._status()
            self.assertIn("cameras", status)
            self.assertEqual(status["cameras"][0]["serial"], "TEST1234")

    async def test_state_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "ble.sock"
            pid_path = Path(tmp) / "ble.pid"
            state_path = Path(tmp) / "ble_state.json"
            daemon = ble_daemon.BleDaemon(
                socket_path=socket_path,
                pid_path=pid_path,
                state_path=state_path,
                maintenance_interval=0.01,
            )

            with patch.object(ble_daemon, "GoProBleConnection", DummyBleConnection), patch.object(
                ble_daemon,
                "scan_ble_devices",
                new=AsyncMock(return_value=[DummyScanResult("1234", address="AA:BB:CC")]),
            ):
                await daemon._cmd_connect({"camera": "TEST1234"})

            self.assertTrue(state_path.exists())
            new_daemon = ble_daemon.BleDaemon(
                socket_path=socket_path,
                pid_path=pid_path,
                state_path=state_path,
            )
            new_daemon._load_state()
            self.assertIn("TEST1234", new_daemon._cameras)

    async def test_get_status_returns_hex(self):
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "ble.sock"
            pid_path = Path(tmp) / "ble.pid"
            daemon = ble_daemon.BleDaemon(socket_path=socket_path, pid_path=pid_path, maintenance_interval=0.01)

            with patch.object(ble_daemon, "GoProBleConnection", DummyBleConnection), patch.object(
                ble_daemon, "scan_ble_devices", new=AsyncMock(return_value=[])
            ):
                result = await daemon._cmd_get_status({"camera": "TEST1234", "status_ids": [8, 10]})

            self.assertEqual(result["8"], "01")
            self.assertEqual(result["10"], "01")

    async def test_disconnect_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "ble.sock"
            pid_path = Path(tmp) / "ble.pid"
            daemon = ble_daemon.BleDaemon(socket_path=socket_path, pid_path=pid_path, maintenance_interval=0.01)

            with patch.object(ble_daemon, "GoProBleConnection", DummyBleConnection), patch.object(
                ble_daemon,
                "scan_ble_devices",
                new=AsyncMock(
                    return_value=[
                        DummyScanResult("1234", address="AA:BB:CC"),
                        DummyScanResult("5678", address="DD:EE:FF"),
                    ]
                ),
            ):
                await daemon._cmd_connect({"camera": "TEST1234"})
                await daemon._cmd_connect({"camera": "TEST5678"})
                result = await daemon._cmd_disconnect({"camera": "all"})

            self.assertEqual(result["disconnected"], "all")
            snapshots = daemon._status()["cameras"]
            self.assertEqual({cam["serial"] for cam in snapshots}, {"TEST1234", "TEST5678"})
            self.assertFalse(any(cam["desired"] for cam in snapshots))

    async def test_sleep_disconnects_and_clears_desired(self):
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "ble.sock"
            pid_path = Path(tmp) / "ble.pid"
            daemon = ble_daemon.BleDaemon(socket_path=socket_path, pid_path=pid_path, maintenance_interval=0.01)

            with patch.object(ble_daemon, "GoProBleConnection", DummyBleConnection), patch.object(
                ble_daemon,
                "scan_ble_devices",
                new=AsyncMock(return_value=[DummyScanResult("1234", address="AA:BB:CC")]),
            ):
                await daemon._cmd_connect({"camera": "TEST1234"})
                result = await daemon._cmd_sleep({"camera": "TEST1234"})

            self.assertTrue(result["slept"])
            self.assertFalse(result["desired"])
            self.assertFalse(result["connected"])

    async def test_wifi_ap_calls_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "ble.sock"
            pid_path = Path(tmp) / "ble.pid"
            daemon = ble_daemon.BleDaemon(socket_path=socket_path, pid_path=pid_path, maintenance_interval=0.01)

            with patch.object(ble_daemon, "GoProBleConnection", DummyBleConnection), patch.object(
                ble_daemon,
                "scan_ble_devices",
                new=AsyncMock(return_value=[DummyScanResult("1234", address="AA:BB:CC")]),
            ):
                await daemon._cmd_connect({"camera": "TEST1234"})
                result = await daemon._cmd_wifi_ap({"camera": "TEST1234", "mode": "bounce"})

            self.assertTrue(result["ok"])
            cam = daemon._cameras["TEST1234"]
            self.assertIsNotNone(cam.connection)
            self.assertEqual(cam.connection.last_ap_mode, 2)

    async def test_wifi_release_marks_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "ble.sock"
            pid_path = Path(tmp) / "ble.pid"
            inventory_path = Path(tmp) / "inventory.yaml"

            inventory = Inventory(path=inventory_path)
            record = CameraInventoryRecord(serial_number="TEST1234")
            record.wifi.available = True
            record.wifi.ip = "192.168.1.10"
            inventory.upsert(record)
            inventory.save()

            daemon = ble_daemon.BleDaemon(socket_path=socket_path, pid_path=pid_path, maintenance_interval=0.01)

            def _temp_inventory():
                return Inventory(path=inventory_path)

            with patch.object(ble_daemon, "get_inventory", new=_temp_inventory), patch.object(
                ble_daemon, "GoProBleConnection", DummyBleConnection
            ), patch.object(
                ble_daemon,
                "scan_ble_devices",
                new=AsyncMock(return_value=[DummyScanResult("1234", address="AA:BB:CC")]),
            ):
                await daemon._cmd_connect({"camera": "TEST1234"})
                result = await daemon._cmd_wifi_release({"camera": "TEST1234"})

            self.assertTrue(result["ok"])
            updated = Inventory(path=inventory_path).get("TEST1234")
            self.assertIsNotNone(updated)
            self.assertFalse(updated.wifi.available)

    async def test_set_camera_name_updates_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "ble.sock"
            pid_path = Path(tmp) / "ble.pid"
            inventory_path = Path(tmp) / "inventory.yaml"

            inventory = Inventory(path=inventory_path)
            inventory.upsert(CameraInventoryRecord(serial_number="TEST1234", camera_name="OldName"))
            inventory.save()

            daemon = ble_daemon.BleDaemon(socket_path=socket_path, pid_path=pid_path, maintenance_interval=0.01)

            def _temp_inventory():
                return Inventory(path=inventory_path)

            with patch.object(ble_daemon, "get_inventory", new=_temp_inventory), patch.object(
                ble_daemon, "GoProBleConnection", DummyBleConnection
            ), patch.object(
                ble_daemon,
                "scan_ble_devices",
                new=AsyncMock(return_value=[DummyScanResult("1234", address="AA:BB:CC")]),
            ):
                await daemon._cmd_connect({"camera": "TEST1234"})
                result = await daemon._cmd_set_camera_name({"camera": "TEST1234", "name": "NewName"})

            self.assertTrue(result["ok"])
            self.assertTrue(result["inventory_updated"])
            self.assertEqual(result["camera_name"], "NewName")

            managed = daemon._cameras["TEST1234"]
            self.assertEqual(managed.camera_name, "NewName")
            self.assertIsNotNone(managed.connection)
            self.assertEqual(managed.connection.last_camera_name_set, "NewName")

            updated = Inventory(path=inventory_path).get("TEST1234")
            self.assertIsNotNone(updated)
            self.assertEqual(updated.camera_name, "NewName")

    async def test_dispatch_unknown_method(self):
        daemon = ble_daemon.BleDaemon(socket_path=Path("/tmp/ble.sock"), pid_path=Path("/tmp/ble.pid"))
        with self.assertRaises(ValueError):
            await daemon._dispatch("nope", {})


if __name__ == "__main__":
    unittest.main()
