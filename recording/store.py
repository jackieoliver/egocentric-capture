from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import (
    Alert,
    AlertSeverity,
    DeviceSnapshot,
    DeviceState,
    DeviceStateEntry,
    RecordingEntry,
    RecordingStatus,
    EVENT_SCHEMA_NAME,
    EVENT_SCHEMA_VERSION,
    SessionManifest,
    SessionState,
    TakeEntry,
    now_iso,
)
from .identity import load_identity
from .names import generate_codename
from .paths import active_session_path, sessions_root


def _safe_session_slug(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in value.strip())
    safe = safe.strip("_-")
    return safe or None


def _session_base_slug(name: Optional[str]) -> tuple[str, Optional[str]]:
    if not name:
        name = generate_codename()
    return name, _safe_session_slug(name)


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, sort_keys=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class SessionStore:
    """
    File-first “database” for capture sessions.

    - One directory per session
    - `session.json` is a derived snapshot (rebuildable)
    - `events.jsonl` is the append-only audit log (source of truth)
    - Optional `index.sqlite3` for fast listing/query
    """

    def __init__(
        self,
        *,
        root_dir: Optional[Path] = None,
        active_path: Optional[Path] = None,
        index_db_path: Optional[Path] = None,
    ) -> None:
        self.root_dir = root_dir or sessions_root()
        self.active_path = active_path or active_session_path()
        self.index_db_path = index_db_path or (self.root_dir / "index.sqlite3")
        self._event_seq_cache: dict[str, int] = {}
        self._last_event_meta: dict[str, dict[str, Any]] = {}

    def session_dir(self, session_id: str) -> Path:
        return self.root_dir / session_id

    def manifest_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session.json"

    def legacy_manifest_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session.yaml"

    def events_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "events.jsonl"

    def _next_event_seq(self, session_id: str) -> int:
        seq = self._event_seq_cache.get(session_id)
        if seq is None:
            seq = 0
            path = self.events_path(session_id)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    for seq, _ in enumerate(f, start=1):
                        pass
            self._event_seq_cache[session_id] = seq
        self._event_seq_cache[session_id] += 1
        return self._event_seq_cache[session_id]

    def _apply_snapshot_marker(self, session: SessionManifest, event: Optional[Dict[str, Any]] = None) -> None:
        meta = event or self._last_event_meta.get(session.session_id)
        if not meta:
            return
        session.snapshot_from_event_id = str(meta.get("event_id") or "") or None
        session.snapshot_from_seq = meta.get("seq")
        session.snapshot_from_time = meta.get("time")
        session.snapshot_from_time_ns = meta.get("time_unix_ns")

    # ---------------------------------------------------------------------
    # Active session pointer
    # ---------------------------------------------------------------------

    def set_active(self, session_id: str) -> None:
        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_path.write_text(session_id, encoding="utf-8")

    def get_active_id(self) -> Optional[str]:
        if not self.active_path.exists():
            return None
        value = self.active_path.read_text(encoding="utf-8").strip()
        return value or None

    def clear_active(self) -> None:
        self.active_path.unlink(missing_ok=True)

    def get_active(self) -> Optional[SessionManifest]:
        session_id = self.get_active_id()
        if not session_id:
            return None
        return self.load(session_id)

    # ---------------------------------------------------------------------
    # Index DB (optional)
    # ---------------------------------------------------------------------

    def _ensure_index_db(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.index_db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                  session_id TEXT PRIMARY KEY,
                  created_at TEXT,
                  name TEXT,
                  participant TEXT,
                  state TEXT,
                  closed_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS takes (
                  session_id TEXT,
                  take_number INTEGER,
                  started_at TEXT,
                  stopped_at TEXT,
                  label TEXT,
                  PRIMARY KEY (session_id, take_number)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _index_upsert_session(self, session: SessionManifest) -> None:
        self._ensure_index_db()
        conn = sqlite3.connect(self.index_db_path)
        try:
            conn.execute(
                """
                INSERT INTO sessions(session_id, created_at, name, participant, state, closed_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  created_at=excluded.created_at,
                  name=excluded.name,
                  participant=excluded.participant,
                  state=excluded.state,
                  closed_at=excluded.closed_at
                """,
                (
                    session.session_id,
                    session.created_at,
                    session.name,
                    session.participant,
                    session.state.value,
                    session.closed_at,
                ),
            )
            for take in session.takes:
                conn.execute(
                    """
                    INSERT INTO takes(session_id, take_number, started_at, stopped_at, label)
                    VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, take_number) DO UPDATE SET
                      started_at=excluded.started_at,
                      stopped_at=excluded.stopped_at,
                      label=excluded.label
                    """,
                    (
                        session.session_id,
                        take.take_number,
                        take.started_at,
                        take.stopped_at,
                        take.label,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    # ---------------------------------------------------------------------
    # CRUD
    # ---------------------------------------------------------------------

    def create(self, *, name: Optional[str] = None, participant: Optional[str] = None) -> SessionManifest:
        date_str = datetime.now().strftime("%Y-%m-%d")
        name, slug = _session_base_slug(name)
        timestamp = int(time.time())
        base = f"{date_str}_{timestamp}_{slug}" if slug else f"{date_str}_{timestamp}"

        session_id = base
        counter = 1
        while self.session_dir(session_id).exists():
            counter += 1
            session_id = f"{base}_{counter:03d}"

        session = SessionManifest(
            session_id=session_id,
            session_uid=uuid.uuid4().hex,
            created_at=now_iso(),
            state=SessionState.OPEN,
            name=name,
            participant=participant,
        )
        identity = load_identity()
        session.recorder_id = identity.recorder_id
        session.operator_id = identity.operator_id
        session.operator_name = identity.operator_name
        self.session_dir(session_id).mkdir(parents=True, exist_ok=True)
        event = self.append_event(session_id, "session_created", {"name": name, "participant": participant})
        self._apply_snapshot_marker(session, event)
        self.save(session)
        self.set_active(session_id)
        return session

    def load(self, session_id: str) -> Optional[SessionManifest]:
        path = self.manifest_path(session_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8") or "{}")
                return SessionManifest.from_dict(data)
            except json.JSONDecodeError:
                backup_path = path.with_suffix(path.suffix + ".bak")
                if backup_path.exists():
                    try:
                        data = json.loads(backup_path.read_text(encoding="utf-8") or "{}")
                        return SessionManifest.from_dict(data)
                    except json.JSONDecodeError:
                        pass

        legacy = self.legacy_manifest_path(session_id)
        if legacy.exists():
            import yaml

            data = yaml.safe_load(legacy.read_text(encoding="utf-8")) or {}
            return SessionManifest.from_dict(data)

        return None

    def save(self, session: SessionManifest) -> None:
        self._apply_snapshot_marker(session)
        path = self.manifest_path(session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(session.to_dict(), indent=2, sort_keys=False) + "\n"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        if path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            try:
                path.replace(backup_path)
            except Exception:
                pass
        tmp_path.replace(path)
        try:
            self._index_upsert_session(session)
        except Exception:
            pass

    def list_sessions(self, limit: int = 50) -> List[str]:
        if not self.root_dir.exists():
            return []

        # File-first: include legacy sessions even if the index DB is missing/out-of-date.
        file_sessions = {
            d.name
            for d in self.root_dir.iterdir()
            if d.is_dir() and ((d / "session.json").exists() or (d / "session.yaml").exists())
        }

        ordered: List[str] = []
        if self.index_db_path.exists():
            conn = sqlite3.connect(self.index_db_path)
            try:
                rows = conn.execute(
                    "SELECT session_id FROM sessions ORDER BY created_at DESC",
                ).fetchall()
                for (sid,) in rows:
                    if sid in file_sessions:
                        ordered.append(str(sid))
            finally:
                conn.close()

        # Add any sessions not present in the index.
        for sid in sorted(file_sessions, reverse=True):
            if sid not in ordered:
                ordered.append(sid)

        return ordered[: int(limit)]

    def close(self, session: SessionManifest) -> None:
        session.state = SessionState.CLOSED
        session.closed_at = now_iso()
        self.append_event(session.session_id, "session_closed", {"total_takes": len(session.takes)})
        self.save(session)
        if self.get_active_id() == session.session_id:
            self.clear_active()

    # ---------------------------------------------------------------------
    # Events
    # ---------------------------------------------------------------------

    def append_event(
        self,
        session_id: str,
        event_type: str,
        data: Dict[str, Any],
        *,
        source: str = "recording.store",
    ) -> Dict[str, Any]:
        writer = JsonlWriter(self.events_path(session_id))
        seq = self._next_event_seq(session_id)
        identity = load_identity()
        payload = {
            "schema": EVENT_SCHEMA_NAME,
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": uuid.uuid4().hex,
            "seq": seq,
            "time": now_iso(),
            "time_unix_ns": time.time_ns(),
            "clock": "host_wall_utc",
            "source": source,
            "recorder_id": identity.recorder_id,
            "operator_id": identity.operator_id,
            "operator_name": identity.operator_name,
            "type": event_type,
        }
        for key, value in (data or {}).items():
            if key in payload:
                continue
            payload[key] = value
        writer.append(payload)
        self._last_event_meta[session_id] = {
            "event_id": payload["event_id"],
            "seq": seq,
            "time": payload["time"],
            "time_unix_ns": payload["time_unix_ns"],
        }
        return payload

    # ---------------------------------------------------------------------
    # Mutations (session/take/recording)
    # ---------------------------------------------------------------------

    def add_device(
        self,
        session: SessionManifest,
        *,
        snapshot: DeviceSnapshot,
        state: DeviceStateEntry,
    ) -> None:
        session.device_snapshots.append(snapshot)
        session.devices.append(state)
        self.append_event(
            session.session_id,
            "device_added",
            {"device_id": snapshot.device_id, "kind": snapshot.kind, "role": snapshot.role, "name": snapshot.name},
        )
        self.save(session)

    def update_device_state(
        self,
        session: SessionManifest,
        *,
        device_id: str,
        state: Optional[DeviceState] = None,
        connection: Optional[str] = None,
        battery_pct: Optional[int] = None,
        storage_gb: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> None:
        target = None
        for dev in session.devices:
            if dev.device_id == device_id:
                target = dev
                break
        if target is None:
            suffix_matches = [dev for dev in session.devices if dev.device_id.endswith(device_id)]
            if len(suffix_matches) == 1:
                target = suffix_matches[0]
        if not target:
            return
        if state is not None:
            target.state = state
        target.last_seen = now_iso()
        if connection is not None:
            target.connection = connection
        if battery_pct is not None:
            target.battery_pct = battery_pct
        if storage_gb is not None:
            target.storage_gb = storage_gb
        if extra:
            target.extra.update(extra)
        if save:
            self.save(session)

    def add_alert(
        self,
        session: SessionManifest,
        *,
        code: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        device_id: Optional[str] = None,
    ) -> None:
        alert = Alert(time=now_iso(), severity=severity, code=code, message=message, device_id=device_id)
        session.alerts.append(alert)
        self.append_event(session.session_id, "alert_added", alert.to_dict())
        self.save(session)

    def start_take(self, session: SessionManifest, *, label: Optional[str] = None) -> TakeEntry:
        if session.current_take:
            raise ValueError("take_already_recording")
        if session.state != SessionState.OPEN or session.closed_at:
            raise ValueError("session_closed")
        take = TakeEntry(
            take_number=session.next_take_number,
            started_at=now_iso(),
            take_uid=uuid.uuid4().hex,
            label=label,
        )
        session.takes.append(take)
        self.append_event(session.session_id, "take_started", {"take_number": take.take_number, "label": label})
        self.save(session)
        return take

    def stop_take(self, session: SessionManifest) -> Optional[TakeEntry]:
        take = session.current_take
        if not take:
            return None
        take.stopped_at = now_iso()
        self.append_event(
            session.session_id,
            "take_stopped",
            {"take_number": take.take_number, "duration_sec": take.duration_sec, "recordings_count": len(take.recordings)},
        )
        self.save(session)
        return take

    def add_recording(self, session: SessionManifest, take: TakeEntry, recording: RecordingEntry) -> None:
        take.recordings.append(recording)
        self.append_event(
            session.session_id,
            "recording_started",
            {
                "take_number": take.take_number,
                "device_id": recording.device_id,
                "kind": recording.kind,
                "modality": recording.modality,
                "source": recording.source,
            },
        )
        self.save(session)

    def complete_recording_linked(
        self,
        session: SessionManifest,
        take: TakeEntry,
        *,
        device_id: str,
        folder: Optional[str],
        file: Optional[str],
        duration_sec: Optional[int],
        size_bytes: Optional[int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        for rec in take.recordings:
            if rec.device_id == device_id:
                rec.stopped_at = now_iso()
                rec.folder = folder
                rec.file = file
                rec.duration_sec = duration_sec
                rec.size_bytes = size_bytes
                if extra:
                    rec.extra.update(extra)
                if folder and file:
                    rec.status = RecordingStatus.LINKED
                break
        self.append_event(
            session.session_id,
            "recording_linked",
            {
                "take_number": take.take_number,
                "device_id": device_id,
                "folder": folder,
                "file": file,
                "size_bytes": size_bytes,
                "extra": extra or {},
            },
        )
        self.save(session)


def get_session_store() -> SessionStore:
    return SessionStore()
