import unittest
from unittest.mock import AsyncMock

from gopro.ble import GoProBleConnection, SUPPORTED_OPEN_GOPRO_API_VERSION


class BleApiVersionTests(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_open_gopro_api_version_ok(self):
        conn = GoProBleConnection(serial="SERIAL", identifier="1234", enforce_api_version=False)
        conn.get_open_gopro_api_version = AsyncMock(return_value=SUPPORTED_OPEN_GOPRO_API_VERSION)
        version = await conn.ensure_open_gopro_api_version()
        self.assertEqual(version, SUPPORTED_OPEN_GOPRO_API_VERSION)

    async def test_ensure_open_gopro_api_version_mismatch(self):
        conn = GoProBleConnection(serial="SERIAL", identifier="1234", enforce_api_version=False)
        conn.get_open_gopro_api_version = AsyncMock(return_value="1.0")
        with self.assertRaises(RuntimeError):
            await conn.ensure_open_gopro_api_version()

    async def test_ensure_open_gopro_api_version_unavailable(self):
        conn = GoProBleConnection(serial="SERIAL", identifier="1234", enforce_api_version=False)
        conn.get_open_gopro_api_version = AsyncMock(return_value=None)
        with self.assertRaises(RuntimeError):
            await conn.ensure_open_gopro_api_version(retries=1)


if __name__ == "__main__":
    unittest.main()

