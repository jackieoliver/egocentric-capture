import json
import tempfile
import unittest
from pathlib import Path

import yaml

from recording.models import DeviceSnapshot, DeviceState, DeviceStateEntry, RecordingEntry, RecordingStatus
from recording.store import SessionStore


class RecordingStoreTests(unittest.TestCase):
    def test_session_store_roundtrip_json_and_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            active = Path(tmp) / "active_session"
            store = SessionStore(root_dir=root, active_path=active, index_db_path=Path(tmp) / "index.sqlite3")

            session = store.create(name="kitchen", participant="P001")
            self.assertEqual(store.get_active_id(), session.session_id)

            store.add_device(
                session,
                snapshot=DeviceSnapshot(device_id="cam1", kind="gopro", role="HEAD"),
                state=DeviceStateEntry(device_id="cam1", kind="gopro", role="HEAD", state=DeviceState.IDLE),
            )
            take = store.start_take(session, label="wash_tomato")
            store.add_recording(
                session,
                take,
                RecordingEntry(
                    device_id="cam1",
                    kind="gopro",
                    role="HEAD",
                    modality="video",
                    status=RecordingStatus.PENDING,
                    source="cohn",
                    started_at="2026-01-01T00:00:00.000000",
                ),
            )
            store.complete_recording_linked(
                session,
                take,
                device_id="cam1",
                folder="100GOPRO",
                file="GX010001.MP4",
                duration_sec=3,
                size_bytes=123,
            )
            store.stop_take(session)
            store.close(session)

            loaded = store.load(session.session_id)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.session_id, session.session_id)
            self.assertEqual(loaded.participant, "P001")
            self.assertEqual(loaded.state.value, "closed")
            self.assertEqual(len(loaded.takes), 1)
            self.assertEqual(loaded.takes[0].label, "wash_tomato")
            self.assertEqual(loaded.takes[0].recordings[0].status.value, "linked")

            events_path = store.events_path(session.session_id)
            self.assertTrue(events_path.exists())
            lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(any(item.get("type") == "session_created" for item in lines))
            self.assertTrue(any(item.get("type") == "take_started" for item in lines))
            self.assertTrue(any(item.get("type") == "recording_linked" for item in lines))

    def test_list_sessions_includes_legacy_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            root.mkdir(parents=True, exist_ok=True)
            legacy_id = "2026-01-01_legacy"
            legacy_dir = root / legacy_id
            legacy_dir.mkdir(parents=True, exist_ok=True)
            (legacy_dir / "session.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 2,
                        "id": legacy_id,
                        "created_at": "2026-01-01T00:00:00.000000",
                        "state": "open",
                        "camera_snapshots": [],
                        "cameras": [],
                        "takes": [],
                        "alerts": [],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            store = SessionStore(
                root_dir=root,
                active_path=Path(tmp) / "active_session",
                index_db_path=Path(tmp) / "index.sqlite3",
            )
            sessions = store.list_sessions(limit=10)
            self.assertIn(legacy_id, sessions)
            loaded = store.load(legacy_id)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.session_id, legacy_id)


if __name__ == "__main__":
    unittest.main()
