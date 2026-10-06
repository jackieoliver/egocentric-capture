import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from gopro.config import (
    COHN_DEFAULT_USERNAME,
    CohnCredentials,
    get_cohn_credentials,
    load_cohn_config,
    save_cohn_config,
    upsert_cohn_credentials,
)
from gopro.http import create_http_client, close_http_client
from gopro.state import CameraInventoryRecord


class CohnConfigTests(unittest.TestCase):
    def test_load_and_save_cohn_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cohn.yaml"
            creds = {
                "CAM123": CohnCredentials(
                    username="gopro",
                    password="secret",
                    certificate="CERT",
                    updated_at="2026-01-01T00:00:00",
                )
            }
            save_cohn_config(creds, path)

            loaded = load_cohn_config(path)
            self.assertIn("CAM123", loaded)
            self.assertEqual(loaded["CAM123"].username, "gopro")
            self.assertEqual(loaded["CAM123"].password, "secret")
            self.assertEqual(loaded["CAM123"].certificate, "CERT")

    def test_upsert_defaults_username(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cohn.yaml"
            upsert_cohn_credentials(
                "CAM456",
                CohnCredentials(username="", password="pw", certificate="CERT"),
                path,
            )
            loaded = load_cohn_config(path)
            self.assertEqual(loaded["CAM456"].username, COHN_DEFAULT_USERNAME)

    def test_get_cohn_credentials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cohn.yaml"
            upsert_cohn_credentials(
                "CAM789",
                CohnCredentials(username="gopro", password="pw", certificate="CERT"),
                path,
            )
            creds = get_cohn_credentials("CAM789", path)
            self.assertIsNotNone(creds)
            self.assertEqual(creds.username, "gopro")


class HttpCohnAuthTests(unittest.TestCase):
    def test_create_http_client_sets_basic_auth(self):
        camera = CameraInventoryRecord(serial_number="CAM123")
        camera.provisioning.cohn_enabled = True
        camera.provisioning.cohn_credential_id = "CAM123"
        camera.wifi.available = True
        camera.wifi.ip = "192.168.0.2"

        creds = CohnCredentials(username="gopro", password="pw", certificate="")
        with patch("gopro.http.load_cohn_config", return_value={"CAM123": creds}):
            client, base_url, transport = create_http_client(camera)
            try:
                self.assertEqual(transport, "cohn")
                self.assertEqual(base_url, "https://192.168.0.2:443")
                self.assertIsInstance(client.auth, httpx.BasicAuth)
            finally:
                if client:
                    close_http_client(client)


if __name__ == "__main__":
    unittest.main()
