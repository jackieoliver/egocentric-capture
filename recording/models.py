from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class SessionState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class DeviceState(str, Enum):
    UNKNOWN = "unknown"
    SLEEPING = "sleeping"
    WAKING = "waking"
    CONNECTING = "connecting"
    IDLE = "idle"
    RECORDING = "recording"
    UNREACHABLE = "unreachable"
    ERROR = "error"


class RecordingStatus(str, Enum):
    PENDING = "pending"
    LINKED = "linked"
    INGESTED = "ingested"
    VERIFIED = "verified"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


SCHEMA_NAME = "haptica.egorec.session"
SCHEMA_VERSION = 1
EVENT_SCHEMA_NAME = "haptica.egorec.event"
EVENT_SCHEMA_VERSION = 1


@dataclass
class Alert:
    time: str
    severity: AlertSeverity
    code: str
    message: str
    device_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "device_id": self.device_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        sev = data.get("severity", "info")
        try:
            severity = AlertSeverity(sev)
        except ValueError:
            severity = AlertSeverity.INFO
        return cls(
            time=str(data.get("time", "")),
            severity=severity,
            code=str(data.get("code", "")),
            message=str(data.get("message", "")),
            device_id=data.get("device_id") or data.get("camera_serial") or data.get("serial"),
        )


@dataclass
class DeviceSnapshot:
    """
    Immutable identity snapshot captured when the session is created.
    """

    device_id: str
    kind: str
    role: Optional[str] = None
    name: Optional[str] = None
    model: Optional[str] = None
    firmware: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "kind": self.kind,
            "role": self.role,
            "name": self.name,
            "model": self.model,
            "firmware": self.firmware,
            "extra": self.extra or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceSnapshot":
        return cls(
            device_id=str(data.get("device_id") or data.get("serial") or ""),
            kind=str(data.get("kind") or "unknown"),
            role=data.get("role"),
            name=data.get("name") or data.get("camera_name"),
            model=data.get("model") or data.get("model_name"),
            firmware=data.get("firmware") or data.get("firmware_version"),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class DeviceStateEntry:
    """
    Mutable per-session state for a device.
    """

    device_id: str
    kind: str
    role: Optional[str] = None
    state: DeviceState = DeviceState.UNKNOWN
    connection: Optional[str] = None
    last_seen: Optional[str] = None
    battery_pct: Optional[int] = None
    storage_gb: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "kind": self.kind,
            "role": self.role,
            "state": self.state.value,
            "connection": self.connection,
            "last_seen": self.last_seen,
            "battery_pct": self.battery_pct,
            "storage_gb": self.storage_gb,
            "extra": self.extra or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceStateEntry":
        state_str = data.get("state", "unknown")
        try:
            state = DeviceState(state_str)
        except ValueError:
            state = DeviceState.UNKNOWN
        return cls(
            device_id=str(data.get("device_id") or data.get("serial") or ""),
            kind=str(data.get("kind") or "unknown"),
            role=data.get("role"),
            state=state,
            connection=data.get("connection"),
            last_seen=data.get("last_seen"),
            battery_pct=data.get("battery_pct"),
            storage_gb=data.get("storage_gb"),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class RecordingEntry:
    device_id: str
    kind: str
    role: Optional[str] = None
    stream_id: Optional[str] = None
    modality: str = "video"
    status: RecordingStatus = RecordingStatus.PENDING
    source: Optional[str] = None
    folder: Optional[str] = None
    file: Optional[str] = None
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    duration_sec: Optional[int] = None
    size_bytes: Optional[int] = None
    local_path: Optional[str] = None
    sha256: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "kind": self.kind,
            "role": self.role,
            "stream_id": self.stream_id,
            "modality": self.modality,
            "status": self.status.value,
            "source": self.source,
            "folder": self.folder,
            "file": self.file,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "duration_sec": self.duration_sec,
            "size_bytes": self.size_bytes,
            "local_path": self.local_path,
            "sha256": self.sha256,
            "extra": self.extra or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordingEntry":
        status_str = data.get("status", "pending")
        try:
            status = RecordingStatus(status_str)
        except ValueError:
            status = RecordingStatus.PENDING
        return cls(
            device_id=str(data.get("device_id") or data.get("serial") or ""),
            kind=str(data.get("kind") or data.get("source") or "unknown"),
            role=data.get("role"),
            stream_id=data.get("stream_id") or data.get("stream"),
            modality=str(data.get("modality") or "video"),
            status=status,
            source=data.get("source"),
            folder=data.get("folder"),
            file=data.get("file"),
            started_at=data.get("started_at"),
            stopped_at=data.get("stopped_at"),
            duration_sec=data.get("duration_sec"),
            size_bytes=data.get("size_bytes"),
            local_path=data.get("local_path"),
            sha256=data.get("sha256"),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class TakeEntry:
    take_number: int
    started_at: str
    take_uid: Optional[str] = None
    stopped_at: Optional[str] = None
    label: Optional[str] = None
    notes: Optional[str] = None
    recordings: List[RecordingEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_recording(self) -> bool:
        return self.stopped_at is None

    @property
    def duration_sec(self) -> Optional[int]:
        if not self.started_at or not self.stopped_at:
            return None
        start = datetime.fromisoformat(self.started_at)
        stop = datetime.fromisoformat(self.stopped_at)
        return int((stop - start).total_seconds())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "take_number": self.take_number,
            "started_at": self.started_at,
            "take_uid": self.take_uid,
            "stopped_at": self.stopped_at,
            "label": self.label,
            "notes": self.notes,
            "recordings": [r.to_dict() for r in self.recordings],
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TakeEntry":
        return cls(
            take_number=int(data.get("take_number") or data.get("take_id") or 0),
            started_at=str(data.get("started_at", "")),
            take_uid=data.get("take_uid") or data.get("take_id") or data.get("uid"),
            stopped_at=data.get("stopped_at"),
            label=data.get("label"),
            notes=data.get("notes"),
            recordings=[RecordingEntry.from_dict(r) for r in (data.get("recordings") or [])],
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SessionManifest:
    schema: str = SCHEMA_NAME
    schema_version: int = SCHEMA_VERSION
    session_id: str = ""
    session_uid: Optional[str] = None
    created_at: str = ""
    state: SessionState = SessionState.OPEN
    closed_at: Optional[str] = None
    name: Optional[str] = None
    participant: Optional[str] = None
    recorder_id: Optional[str] = None
    operator_id: Optional[str] = None
    operator_name: Optional[str] = None
    scene_id: Optional[str] = None
    tactition_id: Optional[str] = None
    rig: Optional[str] = None
    rig_roles: List[str] = field(default_factory=list)
    registry_revision: Optional[str] = None
    snapshot_from_event_id: Optional[str] = None
    snapshot_from_seq: Optional[int] = None
    snapshot_from_time: Optional[str] = None
    snapshot_from_time_ns: Optional[int] = None
    device_snapshots: List[DeviceSnapshot] = field(default_factory=list)
    devices: List[DeviceStateEntry] = field(default_factory=list)
    takes: List[TakeEntry] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)

    @property
    def current_take(self) -> Optional[TakeEntry]:
        for take in self.takes:
            if take.is_recording:
                return take
        return None

    @property
    def next_take_number(self) -> int:
        if not self.takes:
            return 1
        return max(t.take_number for t in self.takes) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "session_uid": self.session_uid,
            "created_at": self.created_at,
            "state": self.state.value,
            "closed_at": self.closed_at,
            "name": self.name,
            "participant": self.participant,
            "recorder_id": self.recorder_id,
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "scene_id": self.scene_id,
            "tactition_id": self.tactition_id,
            "rig": self.rig,
            "rig_roles": self.rig_roles or [],
            "registry_revision": self.registry_revision,
            "snapshot_from_event_id": self.snapshot_from_event_id,
            "snapshot_from_seq": self.snapshot_from_seq,
            "snapshot_from_time": self.snapshot_from_time,
            "snapshot_from_time_ns": self.snapshot_from_time_ns,
            "device_snapshots": [s.to_dict() for s in self.device_snapshots],
            "devices": [d.to_dict() for d in self.devices],
            "takes": [t.to_dict() for t in self.takes],
            "alerts": [a.to_dict() for a in self.alerts],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionManifest":
        # Back-compat for older on-disk formats.
        session_id = data.get("session_id") or data.get("id") or ""
        created_at = data.get("created_at") or data.get("created") or ""
        session_uid = data.get("session_uid") or data.get("uid")
        state_str = data.get("state", "open")
        try:
            state = SessionState(state_str)
        except ValueError:
            state = SessionState.OPEN

        snapshots = data.get("device_snapshots")
        if snapshots is None:
            snapshots = data.get("camera_snapshots") or []

        devices = data.get("devices")
        if devices is None:
            devices = data.get("cameras") or []

        return cls(
            schema=str(data.get("schema") or SCHEMA_NAME),
            schema_version=int(data.get("schema_version") or data.get("schemaVersion") or SCHEMA_VERSION),
            session_id=str(session_id),
            session_uid=str(session_uid) if session_uid else None,
            created_at=str(created_at),
            state=state,
            closed_at=data.get("closed_at"),
            name=data.get("name"),
            participant=data.get("participant"),
            recorder_id=data.get("recorder_id"),
            operator_id=data.get("operator_id"),
            operator_name=data.get("operator_name"),
            scene_id=data.get("scene_id"),
            tactition_id=data.get("tactition_id"),
            rig=data.get("rig"),
            rig_roles=list(data.get("rig_roles") or []),
            registry_revision=data.get("registry_revision"),
            snapshot_from_event_id=data.get("snapshot_from_event_id"),
            snapshot_from_seq=data.get("snapshot_from_seq"),
            snapshot_from_time=data.get("snapshot_from_time"),
            snapshot_from_time_ns=data.get("snapshot_from_time_ns"),
            device_snapshots=[DeviceSnapshot.from_dict(s) for s in (snapshots or [])],
            devices=[DeviceStateEntry.from_dict(d) for d in (devices or [])],
            takes=[TakeEntry.from_dict(t) for t in (data.get("takes") or [])],
            alerts=[Alert.from_dict(a) for a in (data.get("alerts") or [])],
        )
