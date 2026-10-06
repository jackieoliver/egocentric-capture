from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from recording.paths import haptica_home


STATE_DIR = haptica_home() / "state" / "dummy_camera"

def default_socket_path() -> Path:
    override = os.environ.get("DUMMY_CAMERA_SOCKET_PATH")
    return Path(override).expanduser() if override else (STATE_DIR / "dummy_daemon.sock")


def default_pid_path() -> Path:
    override = os.environ.get("DUMMY_CAMERA_PID_PATH")
    return Path(override).expanduser() if override else (STATE_DIR / "dummy_daemon.pid")


def default_log_path() -> Path:
    override = os.environ.get("DUMMY_CAMERA_LOG_PATH")
    return Path(override).expanduser() if override else (STATE_DIR / "dummy_daemon.log")


DEFAULT_SOCKET_PATH = default_socket_path()
DEFAULT_PID_PATH = default_pid_path()
DEFAULT_LOG_PATH = default_log_path()

DEFAULT_ROLES = ("HEAD", "LWRIST", "RWRIST")
_PROCESSES: dict[int, subprocess.Popen] = {}


def _now() -> str:
    return datetime.now().isoformat(timespec="microseconds")


def _is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid(pid_path: Path) -> Optional[int]:
    try:
        return int(pid_path.read_text().strip())
    except Exception:
        return None


def _write_pid(pid_path: Path, pid: int) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(pid))


@dataclass
class DummyMedia:
    folder: str
    file: str
    gumi: str
    size_bytes: int
    duration_sec: float


@dataclass
class DummyCamera:
    serial: str
    short_id: str
    role: str
    name: str
    is_recording: bool = False
    recording_started_at: Optional[float] = None
    media_index: int = 0
    last_media: Optional[DummyMedia] = None

    def start_recording(self) -> None:
        if self.is_recording:
            return
        self.is_recording = True
        self.recording_started_at = time.time()

    def stop_recording(self) -> DummyMedia:
        if not self.is_recording:
            raise RuntimeError("not_recording")
        self.is_recording = False
        started_at = self.recording_started_at or time.time()
        duration = max(0.1, time.time() - started_at)
        self.recording_started_at = None
        self.media_index += 1
        media = DummyMedia(
            folder="100DUMMY",
            file=f"DM{self.media_index:06d}.MP4",
            gumi=uuid.uuid4().hex,
            size_bytes=int(1024 * 1024 + duration * 1024 * 256),
            duration_sec=duration,
        )
        self.last_media = media
        return media


class DummyDaemon:
    def __init__(
        self,
        *,
        socket_path: Optional[Path] = None,
        pid_path: Optional[Path] = None,
        camera_count: int = 3,
    ) -> None:
        self.socket_path = socket_path or default_socket_path()
        self.pid_path = pid_path or default_pid_path()
        self.camera_count = camera_count
        self.started_at = _now()
        self._server: Optional[asyncio.AbstractServer] = None
        self._stop_event = asyncio.Event()
        self._cameras = self._init_cameras(camera_count)

    def _init_cameras(self, camera_count: int) -> list[DummyCamera]:
        cameras: list[DummyCamera] = []
        for idx in range(camera_count):
            role = DEFAULT_ROLES[idx] if idx < len(DEFAULT_ROLES) else f"CAM{idx+1}"
            serial = f"DUMMY{idx+1:04d}"
            cameras.append(
                DummyCamera(
                    serial=serial,
                    short_id=serial[-4:],
                    role=role,
                    name=f"Dummy {role}",
                )
            )
        return cameras

    async def run(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except Exception:
                pass

        _write_pid(self.pid_path, os.getpid())
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop_event.set)
            except NotImplementedError:
                pass

        self._server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        await self._stop_event.wait()
        await self.shutdown()

    async def shutdown(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except Exception:
                pass
        if self.pid_path.exists():
            try:
                self.pid_path.unlink()
            except Exception:
                pass

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                request = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                writer.write(json.dumps({"ok": False, "error": "invalid_json", "id": None}).encode("utf-8") + b"\n")
                await writer.drain()
                continue

            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params") or {}

            try:
                result = await self._dispatch(str(method), dict(params))
                response = {"ok": True, "id": request_id, "result": result}
            except Exception as exc:
                response = {"ok": False, "id": request_id, "error": str(exc)}

            writer.write(json.dumps(response).encode("utf-8") + b"\n")
            await writer.drain()

        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    def _camera_by_id(self, camera_id: str) -> DummyCamera:
        for cam in self._cameras:
            if cam.serial == camera_id or cam.short_id == camera_id or cam.role == camera_id:
                return cam
        raise ValueError("camera_not_found")

    async def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        match method:
            case "status":
                return self._status()
            case "shutdown":
                self._stop_event.set()
                return {"stopping": True}
            case "shutter":
                camera_id = str(params.get("camera") or "")
                action = str(params.get("action") or "")
                if not camera_id:
                    raise ValueError("camera_required")
                if action not in {"start", "stop"}:
                    raise ValueError("action_invalid")
                cam = self._camera_by_id(camera_id)
                if action == "start":
                    cam.start_recording()
                    return {"camera": cam.serial, "role": cam.role, "transport": "dummy", "ok": True, "msg": "started"}
                try:
                    cam.stop_recording()
                    return {"camera": cam.serial, "role": cam.role, "transport": "dummy", "ok": True, "msg": "stopped"}
                except Exception as exc:
                    return {"camera": cam.serial, "role": cam.role, "transport": "dummy", "ok": False, "msg": str(exc)}
            case "last_captured":
                camera_id = str(params.get("camera") or "")
                if not camera_id:
                    raise ValueError("camera_required")
                cam = self._camera_by_id(camera_id)
                media = cam.last_media
                return {
                    "camera": cam.serial,
                    "role": cam.role,
                    "folder": media.folder if media else None,
                    "file": media.file if media else None,
                    "gumi": media.gumi if media else None,
                    "type": "video",
                }
            case "last_captured_resolved":
                camera_id = str(params.get("camera") or "")
                wait_seconds = float(params.get("wait_seconds") or 0.0)
                if not camera_id:
                    raise ValueError("camera_required")
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
                cam = self._camera_by_id(camera_id)
                media = cam.last_media
                info = None
                if media:
                    info = {"s": media.size_bytes, "dur": media.duration_sec}
                return {
                    "camera": cam.serial,
                    "role": cam.role,
                    "folder": media.folder if media else None,
                    "file": media.file if media else None,
                    "gumi": media.gumi if media else None,
                    "type": "video",
                    "info": info,
                }
            case "download_media":
                camera_id = str(params.get("camera") or "")
                folder = str(params.get("folder") or "")
                file = str(params.get("file") or "")
                dest_path = str(params.get("dest_path") or "")
                if not camera_id:
                    raise ValueError("camera_required")
                if not folder or not file:
                    raise ValueError("folder_file_required")
                if not dest_path:
                    raise ValueError("dest_path_required")
                cam = self._camera_by_id(camera_id)
                media = cam.last_media
                size_bytes = media.size_bytes if media else 1024
                dest = Path(dest_path).expanduser()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(b"\0" * size_bytes)
                return {
                    "camera": cam.serial,
                    "role": cam.role,
                    "folder": folder,
                    "file": file,
                    "dest_path": dest_path,
                    "transport": "dummy",
                    "ok": True,
                    "size_bytes": size_bytes,
                    "error": None,
                }
            case _:
                raise ValueError(f"unknown_method:{method}")

    def _status(self) -> dict[str, Any]:
        cameras = []
        for cam in self._cameras:
            cameras.append(
                {
                    "serial": cam.serial,
                    "short_id": cam.short_id,
                    "role": cam.role,
                    "name": cam.name,
                    "health": {
                        "ok": True,
                        "transport": "dummy",
                        "battery_percent": 100,
                        "sd_present": True,
                        "sd_full": False,
                        "sd_status": "ok",
                        "sd_space_remaining_kib": 1024 * 1024,
                        "sd_space_remaining_gib": 1.0,
                        "sd_space_remaining_gb": 1.0,
                        "sd_space_remaining_gb_decimal": 1.0,
                        "sd": {
                            "present": True,
                            "full": False,
                            "status": "ok",
                            "remaining_kib": 1024 * 1024,
                            "remaining_gib": 1.0,
                            "remaining_gb": 1.0,
                        },
                        "is_recording": cam.is_recording,
                        "last_error": None,
                        "last_updated_at": _now(),
                    },
                }
            )
        return {"started_at": self.started_at, "socket": str(self.socket_path), "cameras": cameras}


def request_dummy_daemon(
    *,
    method: str,
    params: Optional[dict[str, Any]] = None,
    socket_path: Optional[Path] = None,
) -> Any:
    socket_path = socket_path or default_socket_path()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(socket_path))
        request_id = uuid.uuid4().hex
        payload = {"id": request_id, "method": method, "params": params or {}}
        sock.sendall(json.dumps(payload).encode("utf-8") + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        line = buf.split(b"\n", 1)[0].decode("utf-8")
        response = json.loads(line) if line else {}
    finally:
        sock.close()

    if not response.get("ok"):
        raise RuntimeError(response.get("error") or "daemon_error")
    return response.get("result")


def start_dummy_daemon(
    *,
    socket_path: Optional[Path] = None,
    pid_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
    camera_count: Optional[int] = None,
) -> int:
    socket_path = socket_path or default_socket_path()
    pid_path = pid_path or default_pid_path()
    log_path = log_path or default_log_path()
    pid = _read_pid(pid_path)
    if pid and _is_process_running(pid):
        return pid

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a")
    try:
        cmd = [
            sys.executable,
            "-m",
            "dummy_camera.dummy_daemon",
            "run",
            "--socket",
            str(socket_path),
            "--pid",
            str(pid_path),
        ]
        if camera_count:
            cmd += ["--count", str(camera_count)]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    finally:
        log_file.close()

    _PROCESSES[proc.pid] = proc
    _write_pid(pid_path, proc.pid)
    return proc.pid


def stop_dummy_daemon(*, socket_path: Optional[Path] = None, pid_path: Optional[Path] = None) -> bool:
    socket_path = socket_path or default_socket_path()
    pid_path = pid_path or default_pid_path()
    try:
        request_dummy_daemon(method="shutdown", socket_path=socket_path)
    except Exception:
        pid = _read_pid(pid_path)
        if pid and _is_process_running(pid):
            os.kill(pid, signal.SIGTERM)
        return False

    deadline = time.time() + 5.0
    while time.time() < deadline:
        pid = _read_pid(pid_path)
        if not pid or not _is_process_running(pid):
            _wait_known_processes()
            return True
        time.sleep(0.1)
    pid = _read_pid(pid_path)
    if pid and _is_process_running(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    _wait_known_processes()
    return False


def _wait_known_processes() -> None:
    for pid, proc in list(_PROCESSES.items()):
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=2.0)
        except Exception:
            pass
        _PROCESSES.pop(pid, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy camera daemon")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the dummy daemon")
    run.add_argument("--socket", type=Path, default=default_socket_path())
    run.add_argument("--pid", type=Path, default=default_pid_path())
    run.add_argument("--count", type=int, default=int(os.environ.get("DUMMY_CAMERA_COUNT", "3")))
    args = parser.parse_args()

    if args.command == "run":
        daemon = DummyDaemon(socket_path=args.socket, pid_path=args.pid, camera_count=args.count)
        asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
