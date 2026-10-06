import argparse
import os
import tempfile
import unittest
from pathlib import Path

from dummy_camera.dummy_daemon import start_dummy_daemon, stop_dummy_daemon
from recording.store import get_session_store


class RecordingCliDummyTests(unittest.TestCase):
    def test_session_take_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sessions_dir = tmp_path / "sessions"
            active_path = tmp_path / "active_session"
            socket_path = tmp_path / "dummy.sock"
            pid_path = tmp_path / "dummy.pid"
            log_path = tmp_path / "dummy.log"

            old_sessions = os.environ.get("HAPTICA_SESSIONS_DIR")
            old_active = os.environ.get("HAPTICA_ACTIVE_SESSION_PATH")
            old_socket = os.environ.get("DUMMY_CAMERA_SOCKET_PATH")
            old_pid = os.environ.get("DUMMY_CAMERA_PID_PATH")
            old_log = os.environ.get("DUMMY_CAMERA_LOG_PATH")
            os.environ["HAPTICA_SESSIONS_DIR"] = str(sessions_dir)
            os.environ["HAPTICA_ACTIVE_SESSION_PATH"] = str(active_path)
            os.environ["DUMMY_CAMERA_SOCKET_PATH"] = str(socket_path)
            os.environ["DUMMY_CAMERA_PID_PATH"] = str(pid_path)
            os.environ["DUMMY_CAMERA_LOG_PATH"] = str(log_path)

            start_dummy_daemon(
                socket_path=socket_path,
                pid_path=pid_path,
                log_path=log_path,
                camera_count=2,
            )
            try:
                from recording import cli as rec_cli

                rec_cli.cmd_session_init(
                    argparse.Namespace(
                        name="test",
                        participant="P001",
                        cameras=None,
                        device="dummy",
                    )
                )
                store = get_session_store()
                session = store.get_active()
                self.assertIsNotNone(session)
                self.assertEqual(len(session.devices), 2)

                rec_cli.cmd_take_start(argparse.Namespace(label="demo", device="dummy"))
                rec_cli.cmd_take_stop(argparse.Namespace(device="dummy"))

                session = store.get_active()
                self.assertIsNotNone(session)
                self.assertEqual(len(session.takes), 1)
                take = session.takes[0]
                self.assertFalse(take.is_recording)
                self.assertTrue(take.recordings)
                self.assertEqual(take.recordings[0].status.value, "linked")
                events_path = store.events_path(session.session_id)
                self.assertTrue(events_path.exists())
                lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertTrue(any('"session_created"' in line for line in lines))
                self.assertTrue(any('"take_started"' in line for line in lines))
                self.assertTrue(any('"recording_started"' in line for line in lines))
                self.assertTrue(any('"recording_linked"' in line for line in lines))
                self.assertTrue(any('"take_stopped"' in line for line in lines))
            finally:
                stop_dummy_daemon(socket_path=socket_path, pid_path=pid_path)
                if old_sessions is None:
                    os.environ.pop("HAPTICA_SESSIONS_DIR", None)
                else:
                    os.environ["HAPTICA_SESSIONS_DIR"] = old_sessions
                if old_active is None:
                    os.environ.pop("HAPTICA_ACTIVE_SESSION_PATH", None)
                else:
                    os.environ["HAPTICA_ACTIVE_SESSION_PATH"] = old_active
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
