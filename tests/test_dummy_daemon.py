import os
import tempfile
import time
import unittest
from pathlib import Path

from dummy_camera.dummy_daemon import (
    request_dummy_daemon,
    start_dummy_daemon,
    stop_dummy_daemon,
)


def _wait_for_socket(path: Path, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


class DummyDaemonTests(unittest.TestCase):
    def test_dummy_daemon_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            socket_path = tmp_path / "dummy.sock"
            pid_path = tmp_path / "dummy.pid"
            log_path = tmp_path / "dummy.log"

            old_socket = os.environ.get("DUMMY_CAMERA_SOCKET_PATH")
            old_pid = os.environ.get("DUMMY_CAMERA_PID_PATH")
            old_log = os.environ.get("DUMMY_CAMERA_LOG_PATH")
            os.environ["DUMMY_CAMERA_SOCKET_PATH"] = str(socket_path)
            os.environ["DUMMY_CAMERA_PID_PATH"] = str(pid_path)
            os.environ["DUMMY_CAMERA_LOG_PATH"] = str(log_path)

            start_dummy_daemon(
                socket_path=socket_path,
                pid_path=pid_path,
                log_path=log_path,
                camera_count=2,
            )
            self.assertTrue(_wait_for_socket(socket_path))
            try:
                status = request_dummy_daemon(method="status", socket_path=socket_path)
                cameras = status.get("cameras") or []
                self.assertTrue(cameras)
                camera = cameras[0]
                serial = camera.get("serial")

                resp = request_dummy_daemon(
                    method="shutter",
                    params={"camera": serial, "action": "start"},
                    socket_path=socket_path,
                )
                self.assertTrue(resp.get("ok"))

                resp = request_dummy_daemon(
                    method="shutter",
                    params={"camera": serial, "action": "stop"},
                    socket_path=socket_path,
                )
                self.assertTrue(resp.get("ok"))

                last = request_dummy_daemon(
                    method="last_captured_resolved",
                    params={"camera": serial, "wait_seconds": 0.0},
                    socket_path=socket_path,
                )
                self.assertIsNotNone(last.get("file"))
                dest = tmp_path / "out.mp4"
                download = request_dummy_daemon(
                    method="download_media",
                    params={
                        "camera": serial,
                        "folder": last.get("folder"),
                        "file": last.get("file"),
                        "dest_path": str(dest),
                    },
                    socket_path=socket_path,
                )
                self.assertTrue(download.get("ok"))
                self.assertTrue(dest.exists())
                self.assertGreater(dest.stat().st_size, 0)
            finally:
                stop_dummy_daemon(socket_path=socket_path, pid_path=pid_path)
                if old_socket is None:
                    os.environ.pop("DUMMY_CAMERA_SOCKET_PATH", None)
                else:
                    os.environ["DUMMY_CAMERA_SOCKET_PATH"] = old_socket
                if old_pid is None:
                    os.environ.pop("DUMMY_CAMERA_PID_PATH", None)
                else:
                    os.environ["DUMMY_CAMERA_PID_PATH"] = old_pid
                if old_log is None:
                    os.environ.pop("DUMMY_CAMERA_LOG_PATH", None)
                else:
                    os.environ["DUMMY_CAMERA_LOG_PATH"] = old_log


if __name__ == "__main__":
    unittest.main()
