from __future__ import annotations

import time
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .models import DeviceSnapshot, DeviceState, DeviceStateEntry, RecordingEntry, RecordingStatus, SessionManifest, TakeEntry
from .rigs import RigSpec, resolve_rig
from .store import SessionStore, get_session_store
from .paths import repo_root
from gopro.sd import sd_extra_from_health


def _read_timeout(env_key: str, default: float) -> float:
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


RECORDING_LINK_TIMEOUT_SECONDS = _read_timeout("HAPTICA_RECORDING_LINK_TIMEOUT", 4.0)


def _prefer_cli_state() -> bool:
    return os.environ.get("HAPTICA_PREFER_CLI_STATE", "").strip().lower() in {"1", "true", "yes", "on"}


def _device_kind(device: str) -> str:
    if device == "gopro":
        return "gopro"
    if device == "dummy":
        return "dummy_camera"
    raise ValueError(f"unknown_device:{device}")


def _device_backend(device: str) -> tuple[Path, Callable[..., Any], Callable[..., Any]]:
    if device == "gopro":
        from gopro.gopro_daemon import (
            DEFAULT_SOCKET_PATH as SOCKET,
            request_gopro_daemon,
            start_gopro_daemon,
        )
        return SOCKET, request_gopro_daemon, start_gopro_daemon
    if device == "dummy":
        from dummy_camera.dummy_daemon import (
            default_socket_path,
            request_dummy_daemon,
            start_dummy_daemon,
        )
        return default_socket_path(), request_dummy_daemon, start_dummy_daemon
    raise ValueError(f"unknown_device:{device}")


def _candidate_gopro_daemon_sockets(preferred: Path) -> list[Path]:
    from gopro.state import state_dir as gopro_state_dir

    candidates: list[Path] = []

    def _add(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    repo_socket = repo_root() / "state" / "gopro_daemon.sock"
    env_socket = gopro_state_dir() / "gopro_daemon.sock"
    if _prefer_cli_state():
        _add(repo_socket)
        _add(env_socket)
    else:
        _add(env_socket)
        _add(repo_socket)
    _add(preferred)
    return candidates


def _resolve_gopro_daemon_socket(preferred: Path) -> tuple[Path, bool]:
    from gopro.gopro_daemon import request_gopro_daemon

    for candidate in _candidate_gopro_daemon_sockets(preferred):
        if not candidate.exists():
            continue
        try:
            request_gopro_daemon(method="status", socket_path=candidate)
            return candidate, True
        except Exception:
            continue
    return preferred, False


def _match_camera(candidate: dict[str, Any], requested: Iterable[str]) -> bool:
    serial = str(candidate.get("serial") or "")
    short_id = str(candidate.get("short_id") or "")
    role = str(candidate.get("role") or "").upper()
    for req in requested:
        if not req:
            continue
        req_up = req.upper()
        if req_up == role:
            return True
        if req in serial or serial.endswith(req):
            return True
        if req == short_id:
            return True
    return False


def _select_cameras(
    cameras: list[dict[str, Any]],
    requested: Optional[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not requested:
        return list(cameras), []
    requested_ids = [r.strip() for r in requested.split(",") if r.strip()]
    matched = [cam for cam in cameras if _match_camera(cam, requested_ids)]
    missing = [req for req in requested_ids if not any(_match_camera(cam, [req]) for cam in cameras)]
    return matched, missing


def _device_state_from_health(health: dict[str, Any]) -> DeviceState:
    ok = bool(health.get("ok"))
    is_recording = health.get("is_recording")
    if not ok:
        return DeviceState.UNREACHABLE
    if is_recording:
        return DeviceState.RECORDING
    return DeviceState.IDLE


@dataclass
class Recorder:
    device: str = "gopro"
    store: SessionStore = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = get_session_store()
        self.device_kind = _device_kind(self.device)
        self.socket_path, self.request_fn, self.start_fn = _device_backend(self.device)

    def _request(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        socket_path = self.socket_path
        start_kwargs: dict[str, Any] = {"socket_path": socket_path}
        existing_daemon = False
        if self.device_kind == "gopro":
            socket_path, existing_daemon = _resolve_gopro_daemon_socket(self.socket_path)
            start_kwargs = {
                "socket_path": socket_path,
                "pid_path": socket_path.with_name("gopro_daemon.pid"),
                "log_path": socket_path.with_name("gopro_daemon.log"),
            }
        if not existing_daemon:
            self.start_fn(**start_kwargs)
        last_error: Optional[Exception] = None
        for _ in range(10):
            try:
                return self.request_fn(method=method, params=params or {}, socket_path=socket_path) or {}
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                last_error = exc
                time.sleep(0.1)
        if last_error:
            raise last_error
        return self.request_fn(method=method, params=params or {}, socket_path=socket_path) or {}

    def status(self) -> dict[str, Any]:
        return self._request("status")

    def request(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return self._request(method, params)

    def create_session(
        self,
        *,
        name: Optional[str] = None,
        participant: Optional[str] = None,
        cameras: Optional[str] = None,
        rig: Optional[str] = None,
        rig_spec: Optional[RigSpec] = None,
    ) -> SessionManifest:
        status = self.status()
        available = status.get("cameras") or []
        if rig_spec:
            if rig_spec.device_kind != self.device_kind:
                raise ValueError("rig_device_mismatch")
            resolved_rig = rig_spec
        else:
            resolved_rig = resolve_rig(self.device_kind, rig, available)
        selected, missing_requested = _select_cameras(available, cameras)
        if cameras:
            if missing_requested:
                raise ValueError(f"cameras_not_found:{','.join(missing_requested)}")
        elif resolved_rig.roles:
            selected = [cam for cam in available if cam.get("role") and str(cam.get("role")).upper() in resolved_rig.roles]

        if resolved_rig.roles:
            present_roles = {str(cam.get("role")).upper() for cam in selected if cam.get("role")}
            missing_roles = [role for role in resolved_rig.roles if role not in present_roles]
            if missing_roles:
                raise ValueError(f"rig_missing:{','.join(missing_roles)}")

        if not selected:
            raise ValueError("no_cameras")
        session = self.store.create(name=name, participant=participant)
        session.rig = resolved_rig.name
        if resolved_rig.roles:
            session.rig_roles = list(resolved_rig.roles)
        else:
            session.rig_roles = [str(cam.get("role")).upper() for cam in selected if cam.get("role")]
        for cam in selected:
            health = cam.get("health") or {}
            state = DeviceStateEntry(
                device_id=str(cam.get("serial") or ""),
                kind=self.device_kind,
                role=cam.get("role"),
                state=_device_state_from_health(health),
                connection=health.get("transport"),
                last_seen=session.created_at,
                battery_pct=health.get("battery_percent"),
                storage_gb=health.get("sd_space_remaining_gb"),
                extra=sd_extra_from_health(health),
            )
            snapshot = DeviceSnapshot(
                device_id=str(cam.get("serial") or ""),
                kind=self.device_kind,
                role=cam.get("role"),
                name=cam.get("name"),
                extra={"short_id": cam.get("short_id")},
            )
            self.store.add_device(session, snapshot=snapshot, state=state)
        self.store.save(session)
        return session

    def refresh_session(self, session: SessionManifest) -> SessionManifest:
        status = self.status()
        daemon_by_serial = {c.get("serial"): c for c in (status.get("cameras") or []) if c.get("serial")}
        for dev in session.devices:
            if dev.kind != self.device_kind:
                continue
            cam = daemon_by_serial.get(dev.device_id) or {}
            health = cam.get("health") or {}
            self.store.update_device_state(
                session,
                device_id=dev.device_id,
                state=_device_state_from_health(health),
                connection=health.get("transport"),
                battery_pct=health.get("battery_percent"),
                storage_gb=health.get("sd_space_remaining_gb"),
                extra=sd_extra_from_health(health),
                save=False,
            )
        self.store.save(session)
        return session

    def start_take(self, session: SessionManifest, *, label: Optional[str] = None, force: bool = False) -> Optional[TakeEntry]:
        if session.current_take:
            raise ValueError("take_already_recording")
        self.refresh_session(session)
        devices = [dev for dev in session.devices if dev.kind == self.device_kind]
        not_ready = [dev for dev in devices if dev.state not in (DeviceState.IDLE, DeviceState.UNKNOWN)]
        if not_ready and not force:
            roles = ", ".join([dev.role or dev.device_id for dev in not_ready])
            raise ValueError(f"devices_not_ready:{roles}")
        take = self.store.start_take(session, label=label)
        started = 0
        started_devices: list[DeviceStateEntry] = []
        for dev in session.devices:
            if dev.kind != self.device_kind:
                continue
            if dev.state not in (DeviceState.IDLE, DeviceState.UNKNOWN):
                continue
            resp = self._request("shutter", {"camera": dev.device_id, "action": "start"})
            if not resp.get("ok"):
                continue
            transport = resp.get("transport")
            self.store.update_device_state(session, device_id=dev.device_id, state=DeviceState.RECORDING, connection=transport)
            self.store.add_recording(
                session,
                take,
                RecordingEntry(
                    device_id=dev.device_id,
                    kind=self.device_kind,
                    role=dev.role,
                    stream_id="video_main",
                    modality="video",
                    status=RecordingStatus.PENDING,
                    source=transport,
                    started_at=session.current_take.started_at if session.current_take else None,
                ),
            )
            started += 1
            started_devices.append(dev)

        if started == 0:
            session.takes.remove(take)
            self.store.save(session)
            return None
        if started != len(devices):
            started_ids = {dev.device_id for dev in started_devices}
            failed_devices = [dev for dev in devices if dev.device_id not in started_ids]
            roles = ", ".join([dev.role or dev.device_id for dev in failed_devices])
            self.store.add_alert(
                session,
                code="start_partial",
                message=f"Partial start; missing devices: {roles}",
            )
            for dev in failed_devices:
                self.store.update_device_state(session, device_id=dev.device_id, state=DeviceState.UNKNOWN)
            self.store.save(session)
            return take

        self.store.save(session)
        return take

    def stop_take(self, session: SessionManifest) -> Optional[TakeEntry]:
        take = session.current_take
        if not take:
            return None
        failed_devices: list[DeviceStateEntry] = []
        for dev in session.devices:
            if dev.kind != self.device_kind:
                continue
            if dev.state != DeviceState.RECORDING:
                continue
            resp: Optional[dict[str, Any]] = None
            stop_ok = False
            try:
                resp = self._request("shutter", {"camera": dev.device_id, "action": "stop"})
                stop_ok = bool(resp.get("ok")) if isinstance(resp, dict) else False
            except Exception:
                stop_ok = False
            if stop_ok:
                transport = resp.get("transport") if isinstance(resp, dict) else None
                self.store.update_device_state(
                    session,
                    device_id=dev.device_id,
                    state=DeviceState.IDLE,
                    connection=transport,
                )
            else:
                failed_devices.append(dev)
                self.store.update_device_state(session, device_id=dev.device_id, state=DeviceState.UNKNOWN)

            last = None
            try:
                last = self._request(
                    "last_captured_resolved",
                    {
                        "camera": dev.device_id,
                        "wait_seconds": 1.0,
                        "timeout_seconds": RECORDING_LINK_TIMEOUT_SECONDS,
                    },
                )
            except Exception:
                last = None
            folder = last.get("folder") if isinstance(last, dict) else None
            file = last.get("file") if isinstance(last, dict) else None
            gumi = last.get("gumi") if isinstance(last, dict) else None
            duration_sec = None
            size_bytes = None
            if folder and file:
                media_info = last.get("info") if isinstance(last, dict) else None
                if media_info and media_info.get("s"):
                    duration_sec = media_info.get("dur")
                    size_bytes = media_info.get("s")
            self.store.complete_recording_linked(
                session,
                take,
                device_id=dev.device_id,
                folder=folder,
                file=file,
                duration_sec=duration_sec,
                size_bytes=size_bytes,
                extra={"gumi": gumi} if gumi else None,
            )

        if failed_devices:
            roles = ", ".join([dev.role or dev.device_id for dev in failed_devices])
            self.store.add_alert(
                session,
                code="stop_failed",
                message=f"Failed to stop devices: {roles}",
            )

        self.store.stop_take(session)
        self.store.save(session)
        return take
