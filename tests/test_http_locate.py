import unittest
from unittest.mock import patch

from gopro import http as gopro_http
from gopro.state import CameraInventoryRecord


class DummyResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class DummyClient:
    def __init__(self, behavior):
        self.calls = []
        self.behavior = behavior

    def get(self, url: str, params=None):
        self.calls.append((url, params or {}))
        return self.behavior(url, params or {})

    def close(self) -> None:
        return None


class LocateTests(unittest.TestCase):
    def test_locate_tries_base_url_then_fallback(self):
        camera = CameraInventoryRecord(serial_number="C1234567890123")
        camera.provisioning.cohn_enabled = True
        camera.wifi.ip = "192.168.1.10"

        def behavior(url, _params):
            if url.endswith("/gopro/camera/setting"):
                return DummyResponse(200)
            if url.startswith("https://"):
                return DummyResponse(404)
            if url.startswith("http://192.168.1.10/"):
                return DummyResponse(200)
            return DummyResponse(500)

        dummy_client = DummyClient(behavior)

        with patch.object(gopro_http, "create_http_client_for_control", return_value=(dummy_client, "https://192.168.1.10:443", "cohn")):
            ok, msg, transport = gopro_http.locate_camera(camera, True, beep_volume_option=100)

        self.assertTrue(ok)
        self.assertEqual(msg, "OK")
        self.assertEqual(transport, "cohn")

        urls = [u for u, _ in dummy_client.calls]
        self.assertIn("https://192.168.1.10:443/gopro/camera/setting", urls[0])
        self.assertIn("https://192.168.1.10:443/gp/gpControl/command/system/locate", urls)
        self.assertIn("http://192.168.1.10/gp/gpControl/command/system/locate", urls)


if __name__ == "__main__":
    unittest.main()

