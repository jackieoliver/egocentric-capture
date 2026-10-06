from __future__ import annotations

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
from typing import Any, Optional, TYPE_CHECKING

from . import http as gopro_http
from .ble import scan_ble_devices
from .ble_daemon import DEFAULT_SOCKET_PATH as BLE_DAEMON_SOCKET_PATH
from .ble_daemon import request_daemon as request_ble_daemon
from .ble_daemon import start_daemon as start_ble_daemon
from .sd import parse_sd_card_status
from .state import STATE_DIR, get_inventory

if TYPE_CHECKING:
    from .state import CameraInventoryRecord

DEFAULT_SOCKET_PATH = STATE_DIR / "gopro_daemon.sock"
DEFAULT_PID_PATH = STATE_DIR / "gopro_daemon.pid"
DEFAULT_LOG_PATH = STATE_DIR / "gopro_daemon.log"

HEALTH_POLL_SECONDS = 3.0
WIFI_STALE_SECONDS = 10.0
BLE_SCAN_SECONDS = 5.0
CONNECT_RETRY_SECONDS = 5.0


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


def _ensure_ble_daemon() -> bool:
    try:
        status = request_ble_daemon(method="status", socket_path=BLE_DAEMON_SOCKET_PATH)
        if status.get("running"):
            return True
    except Exception:
        pass
    start_ble_daemon()
    time.sleep(1.0)
    try:
        status = request_ble_daemon(method="status", socket_path=BLE_DAEMON_SOCKET_PATH)
        return bool(status.get("running"))
    except Exception:
        return False


@dataclass
class CameraHealth:
    ok: bool = False
    transport: Optional[str] = None
    battery_percent: Optional[int] = None
    sd_present: Optional[bool] = None
    sd_full: Optional[bool] = None
    sd_status: Optional[str] = None
    sd_space_remaining_kib: Optional[int] = None
    sd_space_remaining_gib: Optional[float] = None
    sd_space_remaining_gb: Optional[float] = None
    sd_space_remaining_gb_decimal: Optional[float] = None
    is_recording: Optional[bool] = None
    wifi_connected: Optional[bool] = None
    wifi_last_ok_at: Optional[str] = None
    wifi_last_error: Optional[str] = None
    wifi_last_check_at: Optional[str] = None
    last_error: Optional[str] = None
    last_updated_at: Optional[str] = None


@dataclass
class CameraStream:
    active: bool = False
    port: Optional[int] = None
    transport: Optional[str] = None
    watchdog_enabled: bool = False
    watchdog_active: bool = False
    stall_timeout_seconds: float = 3.0
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    first_packet_at: Optional[str] = None
    first_packet_at_monotonic: Optional[float] = None
    last_packet_at: Optional[str] = None
    last_packet_at_monotonic: Optional[float] = None
    last_restart_at: Optional[str] = None
    last_error: Optional[str] = None


@dataclass
class RecordingContext:
    recording_id: str
    camera_serial: str
    role: Optional[str] = None
    transport: Optional[str] = None
    session_id: Optional[str] = None
    take_id: Optional[str] = None
    output_root: Optional[str] = None
    started_at: Optional[str] = None
    started_at_epoch: Optional[float] = None
    stopped_at: Optional[str] = None
    baseline: Optional[gopro_http.LastCaptured] = None
    baseline_info: Optional[dict] = None
    baseline_timestamp: Optional[float] = None
    status: str = "recording"
    media: Optional[dict] = None
    link_method: Optional[str] = None
    link_attempts: int = 0
    last_error: Optional[str] = None
    last_check_at: Optional[str] = None
    last_size: Optional[int] = None
    last_timestamp: Optional[float] = None
    stable_polls: int = 0
    link_not_before_monotonic: Optional[float] = None
    fetch_job: Optional[dict] = None


class GoProDaemon:
    def __init__(
        self,
        *,
        socket_path: Path = DEFAULT_SOCKET_PATH,
        pid_path: Path = DEFAULT_PID_PATH,
        health_poll_seconds: float = HEALTH_POLL_SECONDS,
        ble_scan_seconds: float = BLE_SCAN_SECONDS,
    ) -> None:
        self.socket_path = socket_path
        self.pid_path = pid_path
        self.health_poll_seconds = health_poll_seconds
        self.ble_scan_seconds = ble_scan_seconds

        self.started_at = _now()
        self._server: Optional[asyncio.AbstractServer] = None
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

        self._health: dict[str, CameraHealth] = {}
        self._ble_scan_cache: dict[str, dict[str, Any]] = {}
        self._last_connect_attempt = 0.0
        self._streams: dict[str, CameraStream] = {}
        self._stream_sockets: dict[str, socket.socket] = {}
        self._recordings: dict[str, RecordingContext] = {}
        self._active_recording_by_serial: dict[str, str] = {}
        self._sleeping_serials: set[str] = set()

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

        self._tasks = [
            asyncio.create_task(self._connect_loop(), name="gopro-daemon-connect"),
            asyncio.create_task(self._health_loop(), name="gopro-daemon-health"),
            asyncio.create_task(self._ble_scan_loop(), name="gopro-daemon-ble-scan"),
            asyncio.create_task(self._stream_watchdog_loop(), name="gopro-daemon-stream-watchdog"),
            asyncio.create_task(self._recording_link_loop(), name="gopro-daemon-recording-link"),
        ]

        await self._stop_event.wait()
        await self.shutdown()

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks = []

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
        for sock in list(self._stream_sockets.values()):
            try:
                sock.close()
            except Exception:
                pass
        self._stream_sockets = {}

    async def _connect_loop(self) -> None:
        while not self._stop_event.is_set():
            now = asyncio.get_running_loop().time()
            if now - self._last_connect_attempt < CONNECT_RETRY_SECONDS:
                await asyncio.sleep(0.5)
                continue
            self._last_connect_attempt = now
            await asyncio.to_thread(self._connect_all_best_effort)
            await asyncio.sleep(0.5)

    def _connect_all_best_effort(self) -> None:
        inventory = get_inventory()
        if not _ensure_ble_daemon():
            return
        try:
            request_ble_daemon(method="sync_inventory", socket_path=BLE_DAEMON_SOCKET_PATH)
        except Exception:
            pass
        for cam in inventory.cameras:
            if cam.serial_number in self._sleeping_serials:
                continue
            try:
                request_ble_daemon(
                    method="connect",
                    params={
                        "camera": cam.serial_number,
                        "finish_pairing": not cam.provisioning.ble_paired,
                        "claim_control": True,
                    },
                    socket_path=BLE_DAEMON_SOCKET_PATH,
                )
            except Exception:
                continue

    async def _health_loop(self) -> None:
        while not self._stop_event.is_set():
            await self._refresh_health()
            await asyncio.sleep(self.health_poll_seconds)

    async def _refresh_health(self) -> None:
        inventory = get_inventory()
        for cam in inventory.cameras:
            await asyncio.to_thread(self._refresh_one_camera_health, cam)

    def _refresh_one_camera_health(self, camera: "CameraInventoryRecord | str") -> None:
        if isinstance(camera, str):
            inventory = get_inventory()
            cam = inventory.get(camera) or inventory.get_by_role(camera)
        else:
            cam = camera
        if not cam:
            return
        health = self._health.get(cam.serial_number) or CameraHealth()
        now = _now()
        wifi_should_probe = bool(cam.wifi.ip) and bool(cam.provisioning.cohn_enabled)
        health.wifi_last_check_at = now
        try:
            client, base_url, transport = gopro_http.create_http_client(
                cam,
                timeout=2.0,
                prefer_usb=False,
                allow_usb=False,
            )
            if not client or not base_url or not transport:
                health.ok = False
                health.transport = None
                health.last_error = "no_http_path"
                health.last_updated_at = now
                if wifi_should_probe:
                    health.wifi_connected = False
                    health.wifi_last_error = "no_http_path"
                else:
                    health.wifi_connected = None
                    health.wifi_last_error = None
                self._health[cam.serial_number] = health
                return
            try:
                resp = client.get(f"{base_url}/gopro/camera/state")
                resp.raise_for_status()
                status = resp.json().get("status", {})
                health.wifi_connected = True
                health.wifi_last_ok_at = now
                health.wifi_last_error = None
            except Exception as exc:
                health.ok = False
                health.transport = None
                health.last_error = str(exc)
                health.last_updated_at = now
                if wifi_should_probe:
                    health.wifi_connected = False
                    health.wifi_last_error = str(exc)
                else:
                    health.wifi_connected = None
                    health.wifi_last_error = None
                self._health[cam.serial_number] = health
                return
            finally:
                gopro_http.close_http_client(client)

            battery = _safe_int(status.get("70"))
            recording = _safe_int(status.get("10"))
            sd = parse_sd_card_status(status)

            health.ok = True
            health.transport = transport
            health.battery_percent = battery
            health.sd_present = sd.present
            health.sd_full = sd.full
            health.sd_status = sd.status
            health.sd_space_remaining_kib = sd.remaining_kib
            health.sd_space_remaining_gib = sd.remaining_gib
            health.sd_space_remaining_gb = sd.remaining_gb
            health.sd_space_remaining_gb_decimal = sd.remaining_gb
            health.is_recording = bool(recording) if recording is not None else None
            health.last_error = None
            health.last_updated_at = now
        except Exception as exc:
            health.ok = False
            health.transport = None
            health.last_error = str(exc)
            health.last_updated_at = now
            if wifi_should_probe:
                health.wifi_connected = False
                health.wifi_last_error = str(exc)
            else:
                health.wifi_connected = None
                health.wifi_last_error = None
        self._health[cam.serial_number] = health

    async def _ble_scan_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                results = await scan_ble_devices(timeout=2.0)
                cache: dict[str, dict[str, Any]] = {}
                for item in results:
                    if item.serial_suffix:
                        cache[item.serial_suffix] = {
                            "name": item.name,
                            "address": item.address,
                            "pairing": item.pairing,
                            "awake": item.awake,
                            "wifi_ap_on": item.wifi_ap_on,
                            "schema_version": item.schema_version,
                        }
                self._ble_scan_cache = cache
            except Exception:
                pass
            await asyncio.sleep(self.ble_scan_seconds)

    async def _stream_watchdog_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            now_monotonic = loop.time()
            for serial, stream in list(self._streams.items()):
                if not stream.active or not stream.watchdog_enabled or not stream.port:
                    continue

                sock = self._stream_sockets.get(serial)
                if not sock:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        if hasattr(socket, "SO_REUSEPORT"):
                            try:
                                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                            except OSError:
                                pass
                        sock.bind(("", int(stream.port)))
                        sock.setblocking(False)
                        self._stream_sockets[serial] = sock
                        stream.watchdog_active = True
                    except Exception as exc:
                        stream.watchdog_active = False
                        stream.last_error = f"watchdog_bind_failed:{exc}"
                        try:
                            if sock:
                                sock.close()
                        except Exception:
                            pass
                        self._stream_sockets.pop(serial, None)
                        continue

                if sock:
                    if not stream.watchdog_active:
                        stream.watchdog_active = True
                    try:
                        while True:
                            sock.recvfrom(8192)
                            stream.last_packet_at_monotonic = now_monotonic
                            stream.last_packet_at = _now()
                            if stream.first_packet_at is None:
                                stream.first_packet_at = stream.last_packet_at
                                stream.first_packet_at_monotonic = now_monotonic
                    except BlockingIOError:
                        pass
                    except Exception as exc:
                        stream.last_error = f"watchdog_recv_failed:{exc}"

                last_packet = stream.last_packet_at_monotonic
                if last_packet is None:
                    last_packet = now_monotonic
                    stream.last_packet_at_monotonic = last_packet
                if now_monotonic - last_packet > stream.stall_timeout_seconds:
                    await asyncio.to_thread(self._restart_stream, serial)
            await asyncio.sleep(0.5)

    async def _recording_link_loop(self) -> None:
        while not self._stop_event.is_set():
            pending = [rec for rec in self._recordings.values() if rec.status == "pending"]
            if not pending:
                await asyncio.sleep(0.5)
                continue

            inventory = get_inventory()
            now_monotonic = time.monotonic()
            for rec in pending:
                if rec.link_not_before_monotonic and now_monotonic < rec.link_not_before_monotonic:
                    continue
                cam = inventory.get(rec.camera_serial) or inventory.get_by_role(rec.camera_serial)
                if not cam:
                    rec.last_error = "camera_not_found"
                    rec.last_check_at = _now()
                    continue
                await self._try_link_recording(cam, rec)
            await asyncio.sleep(0.5)

    async def _try_link_recording(self, cam, rec: RecordingContext) -> None:
        rec.link_attempts += 1
        rec.last_check_at = _now()
        rec.last_error = None
        allow_usb = bool(cam.usb.available and cam.usb.ip)

        if rec.baseline and rec.baseline_timestamp is None and rec.baseline.folder and rec.baseline.file:
            try:
                baseline_info = await asyncio.to_thread(
                    gopro_http.get_media_info,
                    cam,
                    rec.baseline.folder,
                    rec.baseline.file,
                    prefer_usb=False,
                    allow_usb=allow_usb,
                )
            except Exception as exc:
                rec.last_error = f"baseline_info_failed:{exc}"
                baseline_info = None
            if baseline_info:
                rec.baseline_info = baseline_info
                rec.baseline_timestamp = _extract_media_timestamp(baseline_info)

        try:
            last = await asyncio.to_thread(
                gopro_http.get_last_captured,
                cam,
                prefer_usb=False,
                allow_usb=allow_usb,
            )
        except Exception as exc:
            rec.last_error = f"last_captured_failed:{exc}"
            return

        if not last.folder or not last.file:
            return
        if rec.baseline and _same_capture(last, rec.baseline):
            return

        try:
            info = await asyncio.to_thread(
                gopro_http.get_media_info,
                cam,
                last.folder,
                last.file,
                prefer_usb=False,
                allow_usb=allow_usb,
            )
        except Exception as exc:
            rec.last_error = f"media_info_failed:{exc}"
            return

        if not info:
            return
        size_bytes = info.get("s")
        if size_bytes is None or int(size_bytes) <= 0:
            return
        candidate_timestamp = _extract_media_timestamp(info)
        if candidate_timestamp is None:
            return

        min_timestamp = rec.baseline_timestamp
        if min_timestamp is not None and candidate_timestamp <= min_timestamp:
            return

        if rec.last_size == size_bytes and rec.last_timestamp == candidate_timestamp:
            rec.stable_polls += 1
        else:
            rec.stable_polls = 1
            rec.last_size = int(size_bytes)
            rec.last_timestamp = candidate_timestamp
        if rec.stable_polls < 2:
            return

        rec.status = "linked"
        rec.link_method = "timestamp"
        rec.media = {
            "folder": last.folder,
            "file": last.file,
            "gumi": last.gumi,
            "type": last.media_type,
            "info": info,
        }
        rec.fetch_job = _build_fetch_job(rec, cam)
        rec.last_error = None

    def _restart_stream(self, serial: str) -> None:
        stream = self._streams.get(serial)
        if not stream or not stream.port:
            return
        inventory = get_inventory()
        cam = inventory.get(serial) or inventory.get_by_role(serial)
        if not cam:
            stream.last_error = "camera_not_found"
            return

        gopro_http.stop_stream(cam, prefer_usb=False, allow_usb=False)
        ok, msg, transport = gopro_http.start_stream(
            cam,
            stream.port,
            prefer_usb=False,
            allow_usb=False,
        )
        stream.active = bool(ok)
        stream.transport = transport
        if ok:
            stream.started_at = _now()
            stream.last_restart_at = _now()
            stream.last_error = None
            stream.first_packet_at = None
            stream.first_packet_at_monotonic = None
            stream.last_packet_at_monotonic = None
        else:
            stream.last_error = f"restart_failed:{msg}"

    def _status(self) -> dict[str, Any]:
        inventory = get_inventory()
        ble_status: dict[str, Any] = {}
        try:
            ble_status = request_ble_daemon(method="status", socket_path=BLE_DAEMON_SOCKET_PATH) or {}
        except Exception as exc:
            ble_status = {"running": False, "error": str(exc)}

        ble_by_serial: dict[str, dict[str, Any]] = {}
        for cam in ble_status.get("cameras", []) or []:
            serial = cam.get("serial")
            if serial:
                ble_by_serial[str(serial)] = cam

        cameras: list[dict[str, Any]] = []
        for cam in inventory.cameras:
            ble = ble_by_serial.get(cam.serial_number) or {}
            scan = self._ble_scan_cache.get(cam.short_id) or {}
            health = self._health.get(cam.serial_number) or CameraHealth(ok=False)
            stream = self._streams.get(cam.serial_number)
            age_ms = None
            first_packet_ms = None
            if stream and stream.last_packet_at:
                epoch = _iso_to_epoch(stream.last_packet_at)
                if epoch is not None:
                    age_ms = max(0.0, (time.time() - epoch) * 1000.0)
            if stream and stream.started_at and stream.first_packet_at:
                started_epoch = _iso_to_epoch(stream.started_at)
                first_epoch = _iso_to_epoch(stream.first_packet_at)
                if started_epoch is not None and first_epoch is not None:
                    first_packet_ms = max(0.0, (first_epoch - started_epoch) * 1000.0)
            wifi_connected = health.wifi_connected
            wifi_stale = False
            stream_stalled = False
            if wifi_connected is True and _wifi_is_stale(health.wifi_last_ok_at):
                wifi_connected = False
                wifi_stale = True
            if stream and stream.active and stream.last_packet_at and stream.stall_timeout_seconds:
                last_epoch = _iso_to_epoch(stream.last_packet_at)
                if last_epoch is not None:
                    if (time.time() - last_epoch) > float(stream.stall_timeout_seconds):
                        wifi_connected = False
                        wifi_stale = True
                        stream_stalled = True
                    elif wifi_connected is None:
                        wifi_connected = True

            health_ok = bool(health.ok)
            health_error = health.last_error
            if wifi_connected is False:
                health_ok = False
                if not health_error:
                    health_error = "wifi_disconnected"
            if wifi_stale:
                health_ok = False
                if not health_error:
                    health_error = "wifi_stale"
            if stream_stalled and not health_error:
                health_error = "stream_stalled"
            wifi_ip = cam.wifi.ip if wifi_connected is not False else None
            cameras.append(
                {
                    "serial": cam.serial_number,
                    "short_id": cam.short_id,
                    "role": cam.role,
                    "name": cam.camera_name,
                    "sleeping": cam.serial_number in self._sleeping_serials,
                    "ble": {
                        "paired": bool(cam.provisioning.ble_paired),
                        "connected": bool(ble.get("connected")),
                        "address": cam.ble.address,
                        "last_error": ble.get("last_error"),
                        "pairing_mode": scan.get("pairing"),
                        "awake": scan.get("awake"),
                    },
                    "wifi": {
                        "available": bool(cam.wifi.available),
                        "ip": wifi_ip,
                        "ip_cached": cam.wifi.ip,
                        "cohn_enabled": bool(cam.provisioning.cohn_enabled),
                        "connected": wifi_connected,
                        "last_ok_at": health.wifi_last_ok_at,
                        "last_error": health.wifi_last_error,
                        "last_check_at": health.wifi_last_check_at,
                        "stale": wifi_stale,
                    },
                    "health": {
                        "ok": health_ok,
                        "transport": health.transport,
                        "battery_percent": health.battery_percent,
                        "sd_present": health.sd_present,
                        "sd_full": health.sd_full,
                        "sd_status": health.sd_status,
                        "sd_space_remaining_kib": health.sd_space_remaining_kib,
                        "sd_space_remaining_gib": health.sd_space_remaining_gib,
                        "sd_space_remaining_gb": health.sd_space_remaining_gb,
                        "sd_space_remaining_gb_decimal": health.sd_space_remaining_gb_decimal,
                        "sd": {
                            "present": health.sd_present,
                            "full": health.sd_full,
                            "status": health.sd_status,
                            "remaining_kib": health.sd_space_remaining_kib,
                            "remaining_gib": health.sd_space_remaining_gib,
                            "remaining_gb": health.sd_space_remaining_gb,
                        },
                        "is_recording": health.is_recording,
                        "last_error": health_error,
                        "last_updated_at": health.last_updated_at,
                    },
                    "stream": {
                        "active": bool(stream.active) if stream else False,
                        "port": stream.port if stream else None,
                        "transport": stream.transport if stream else None,
                        "watchdog_enabled": stream.watchdog_enabled if stream else False,
                        "watchdog_active": stream.watchdog_active if stream else False,
                        "stall_timeout_seconds": stream.stall_timeout_seconds if stream else None,
                        "started_at": stream.started_at if stream else None,
                        "stopped_at": stream.stopped_at if stream else None,
                        "first_packet_at": stream.first_packet_at if stream else None,
                        "last_packet_at": stream.last_packet_at if stream else None,
                        "last_restart_at": stream.last_restart_at if stream else None,
                        "last_error": stream.last_error if stream else None,
                        "age_ms": age_ms,
                        "first_packet_ms": first_packet_ms,
                    },
                }
            )

        return {
            "started_at": self.started_at,
            "socket": str(self.socket_path),
            "ble_daemon": {
                "running": bool(ble_status.get("running")),
                "pid": ble_status.get("pid"),
                "socket": ble_status.get("socket"),
                "error": ble_status.get("error"),
            },
            "cameras": cameras,
        }

    def _stream_health(self) -> dict[str, Any]:
        inventory = get_inventory()
        now = time.time()
        streams: list[dict[str, Any]] = []
        summary = {
            "active": 0,
            "waiting": 0,
            "stalled": 0,
            "inactive": 0,
            "watchdog_off": 0,
        }

        for cam in inventory.cameras:
            stream = self._streams.get(cam.serial_number) or CameraStream()
            port = stream.port or cam.udp_preview_port
            age_ms = None
            if stream.last_packet_at:
                epoch = _iso_to_epoch(stream.last_packet_at)
                if epoch is not None:
                    age_ms = max(0.0, (now - epoch) * 1000.0)
            first_packet_ms = None
            if stream.started_at and stream.first_packet_at:
                started_epoch = _iso_to_epoch(stream.started_at)
                first_epoch = _iso_to_epoch(stream.first_packet_at)
                if started_epoch is not None and first_epoch is not None:
                    first_packet_ms = max(0.0, (first_epoch - started_epoch) * 1000.0)

            status = "inactive"
            if stream.active:
                status = "active"
                if stream.watchdog_enabled:
                    if not stream.watchdog_active:
                        status = "watchdog_off"
                    elif age_ms is None:
                        status = "waiting"
                    elif age_ms / 1000.0 > float(stream.stall_timeout_seconds):
                        status = "stalled"

            if status in summary:
                summary[status] += 1

            streams.append(
                {
                    "serial": cam.serial_number,
                    "short_id": cam.short_id,
                    "role": cam.role,
                    "name": cam.camera_name,
                    "port": port,
                    "active": bool(stream.active),
                    "transport": stream.transport,
                    "watchdog_enabled": bool(stream.watchdog_enabled),
                    "watchdog_active": bool(stream.watchdog_active),
                    "stall_timeout_seconds": stream.stall_timeout_seconds,
                    "started_at": stream.started_at,
                    "first_packet_at": stream.first_packet_at,
                    "last_packet_at": stream.last_packet_at,
                    "last_restart_at": stream.last_restart_at,
                    "last_error": stream.last_error,
                    "age_ms": age_ms,
                    "first_packet_ms": first_packet_ms,
                    "status": status,
                }
            )

        return {"updated_at": _now(), "summary": summary, "streams": streams}

    async def _call_ble_daemon(self, method: str, params: dict[str, Any]) -> Any:
        return await asyncio.to_thread(
            request_ble_daemon,
            method=method,
            params=params,
            socket_path=BLE_DAEMON_SOCKET_PATH,
        )

    def _resolve_inventory_camera(self, camera_id: str) -> Optional[object]:
        inventory = get_inventory()
        return inventory.get(camera_id) or inventory.get_by_role(camera_id)

    async def _wake_camera(self, camera_id: str) -> dict[str, Any]:
        cam = self._resolve_inventory_camera(camera_id)
        if not cam:
            raise ValueError("camera_not_found")
        self._sleeping_serials.discard(cam.serial_number)
        if not _ensure_ble_daemon():
            raise RuntimeError("ble_daemon_unavailable")
        result = await self._call_ble_daemon("wake", {"camera": cam.serial_number})
        return {"camera": cam.serial_number, "role": cam.role, "result": result}

    async def _sleep_camera(self, camera_id: str) -> dict[str, Any]:
        cam = self._resolve_inventory_camera(camera_id)
        if not cam:
            raise ValueError("camera_not_found")
        if not _ensure_ble_daemon():
            raise RuntimeError("ble_daemon_unavailable")
        result = await self._call_ble_daemon("sleep", {"camera": cam.serial_number})
        self._sleeping_serials.add(cam.serial_number)
        return {"camera": cam.serial_number, "role": cam.role, "result": result}

    async def _arm_camera(self, camera_id: str, params: dict[str, Any]) -> dict[str, Any]:
        cam = self._resolve_inventory_camera(camera_id)
        if not cam:
            raise ValueError("camera_not_found")
        self._sleeping_serials.discard(cam.serial_number)
        if not _ensure_ble_daemon():
            raise RuntimeError("ble_daemon_unavailable")

        bounce_ap = bool(params.get("bounce_ap", False))
        skip_wifi = bool(params.get("skip_wifi", False))
        skip_cohn = bool(params.get("skip_cohn", False))
        allow_cohn_enable = not bool(params.get("no_cohn_enable", False))

        await self._call_ble_daemon(
            "connect",
            {
                "camera": cam.serial_number,
                "finish_pairing": not cam.provisioning.ble_paired,
                "claim_control": True,
            },
        )
        if bounce_ap:
            await self._call_ble_daemon("wifi_ap", {"camera": cam.serial_number, "mode": "bounce"})
        wifi_result = None
        if not skip_wifi:
            wifi_result = await self._call_ble_daemon("wifi_join", {"camera": cam.serial_number})
        cohn_result = None
        if not skip_cohn:
            cohn_result = await self._call_ble_daemon("cohn_status", {"camera": cam.serial_number})
            if allow_cohn_enable and not cohn_result.get("ready"):
                cohn_result = await self._call_ble_daemon("cohn_enable", {"camera": cam.serial_number})

        await asyncio.to_thread(self._refresh_one_camera_health, cam)
        return {
            "camera": cam.serial_number,
            "role": cam.role,
            "wifi_join": wifi_result,
            "cohn": cohn_result,
        }

    async def _arm_all(self, params: dict[str, Any]) -> dict[str, Any]:
        inventory = get_inventory()
        results: list[dict[str, Any]] = []
        for cam in inventory.cameras:
            result = await self._arm_camera(cam.serial_number, params)
            results.append(result)
        return {"cameras": results}

    async def _disarm_camera(self, camera_id: str, params: dict[str, Any]) -> dict[str, Any]:
        cam = self._resolve_inventory_camera(camera_id)
        if not cam:
            raise ValueError("camera_not_found")
        if cam.serial_number in self._sleeping_serials:
            return {"camera": cam.serial_number, "role": cam.role, "skipped": "sleeping"}
        if not _ensure_ble_daemon():
            raise RuntimeError("ble_daemon_unavailable")

        keep_ap = bool(params.get("keep_ap", False))
        release_result = await self._call_ble_daemon("wifi_release", {"camera": cam.serial_number})
        ap_result = None
        if not keep_ap:
            ap_result = await self._call_ble_daemon("wifi_ap", {"camera": cam.serial_number, "mode": "disable"})

        await asyncio.to_thread(self._refresh_one_camera_health, cam)
        return {
            "camera": cam.serial_number,
            "role": cam.role,
            "wifi_release": release_result,
            "wifi_ap": ap_result,
        }

    async def _disarm_all(self, params: dict[str, Any]) -> dict[str, Any]:
        inventory = get_inventory()
        results: list[dict[str, Any]] = []
        for cam in inventory.cameras:
            result = await self._disarm_camera(cam.serial_number, params)
            results.append(result)
        return {"cameras": results}

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while not reader.at_eof():
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

    async def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        match method:
            case "status":
                return self._status()
            case "stream_health":
                return self._stream_health()
            case "shutdown":
                self._stop_event.set()
                return {"stopping": True}
            case "refresh":
                await self._refresh_health()
                return {"refreshed_at": _now()}
            case "connect_all":
                await asyncio.to_thread(self._connect_all_best_effort)
                return {"requested_at": _now()}
            case "wake":
                camera_id = str(params.get("camera") or "")
                if not camera_id:
                    raise ValueError("camera_required")
                return await self._wake_camera(camera_id)
            case "sleep":
                camera_id = str(params.get("camera") or "")
                if not camera_id:
                    raise ValueError("camera_required")
                return await self._sleep_camera(camera_id)
            case "arm":
                camera_id = str(params.get("camera") or "")
                use_all = bool(params.get("all", False))
                if not camera_id and not use_all:
                    raise ValueError("camera_required")
                if use_all:
                    return await self._arm_all(params)
                return await self._arm_camera(camera_id, params)
            case "disarm":
                camera_id = str(params.get("camera") or "")
                use_all = bool(params.get("all", False))
                if not camera_id and not use_all:
                    raise ValueError("camera_required")
                if use_all:
                    return await self._disarm_all(params)
                return await self._disarm_camera(camera_id, params)
            case "locate":
                camera_id = str(params.get("camera") or "")
                if not camera_id:
                    raise ValueError("camera_required")
                duration = float(params.get("duration", 8.0))
                on = params.get("on")
                if on is None:
                    on = True
                on = bool(on)
                beep_volume = params.get("beep_volume")
                if beep_volume is None or beep_volume == "":
                    beep_volume_option = 100
                else:
                    beep_volume_option = int(beep_volume)

                inventory = get_inventory()
                cam = inventory.get(camera_id) or inventory.get_by_role(camera_id)
                if not cam:
                    raise ValueError("camera_not_found")

                ok_on, msg_on, transport = await asyncio.to_thread(
                    gopro_http.locate_camera,
                    cam,
                    on,
                    beep_volume_option=beep_volume_option if on else None,
                    timeout=2.0,
                    prefer_usb=False,
                    allow_usb=False,
                )

                ok_off: Optional[bool] = None
                msg_off: Optional[str] = None
                if on and duration and duration > 0:
                    await asyncio.sleep(duration)
                    ok_off, msg_off, _ = await asyncio.to_thread(
                        gopro_http.locate_camera,
                        cam,
                        False,
                        beep_volume_option=None,
                        timeout=2.0,
                        prefer_usb=False,
                        allow_usb=False,
                    )

                return {
                    "camera": cam.serial_number,
                    "role": cam.role,
                    "transport": transport,
                    "on": {"ok": ok_on, "msg": msg_on},
                    "off": {"ok": ok_off, "msg": msg_off},
                }
            case "shutter":
                camera_id = str(params.get("camera") or "")
                action = str(params.get("action") or "")
                if not camera_id:
                    raise ValueError("camera_required")
                if action not in {"start", "stop"}:
                    raise ValueError("action_invalid")

                inventory = get_inventory()
                cam = inventory.get(camera_id) or inventory.get_by_role(camera_id)
                if not cam:
                    raise ValueError("camera_not_found")

                ok, msg, transport = await asyncio.to_thread(
                    gopro_http.shutter_control,
                    cam,
                    action,
                    prefer_usb=False,
                    allow_usb=False,
                )
                return {
                    "camera": cam.serial_number,
                    "role": cam.role,
                    "transport": transport,
                    "ok": bool(ok),
                    "msg": msg,
                }
            case "record_start":
                camera_id = str(params.get("camera") or "")
                if not camera_id:
                    raise ValueError("camera_required")

                inventory = get_inventory()
                cam = inventory.get(camera_id) or inventory.get_by_role(camera_id)
                if not cam:
                    raise ValueError("camera_not_found")

                baseline = await asyncio.to_thread(
                    gopro_http.get_last_captured,
                    cam,
                    prefer_usb=False,
                    allow_usb=False,
                )
                session_id = params.get("session_id")
                take_id = params.get("take_id")
                output_root = params.get("output_root")
                baseline_info = None
                if baseline.folder and baseline.file:
                    baseline_info = await asyncio.to_thread(
                        gopro_http.get_media_info,
                        cam,
                        baseline.folder,
                        baseline.file,
                        prefer_usb=False,
                        allow_usb=False,
                    )

                ok, msg, transport = await asyncio.to_thread(
                    gopro_http.shutter_control,
                    cam,
                    "start",
                    prefer_usb=False,
                    allow_usb=False,
                )
                started_at = _now()
                if ok:
                    recording_id = uuid.uuid4().hex
                    baseline_timestamp = _extract_media_timestamp(baseline_info)
                    started_at_epoch = _iso_to_epoch(started_at)
                    self._recordings[recording_id] = RecordingContext(
                        recording_id=recording_id,
                        camera_serial=cam.serial_number,
                        role=cam.role,
                        transport=transport,
                        session_id=str(session_id) if session_id else None,
                        take_id=str(take_id) if take_id else None,
                        output_root=str(output_root) if output_root else None,
                        started_at=started_at,
                        started_at_epoch=started_at_epoch,
                        baseline=baseline,
                        baseline_info=baseline_info,
                        baseline_timestamp=baseline_timestamp,
                        status="recording",
                    )
                    self._active_recording_by_serial[cam.serial_number] = recording_id
                else:
                    recording_id = None
                return {
                    "camera": cam.serial_number,
                    "role": cam.role,
                    "transport": transport,
                    "ok": bool(ok),
                    "msg": msg,
                    "started_at": started_at,
                    "recording_id": recording_id,
                    "baseline": baseline.to_dict() if baseline else None,
                    "baseline_info": baseline_info,
                }
            case "record_stop":
                camera_id = str(params.get("camera") or "")
                if not camera_id:
                    raise ValueError("camera_required")
                wait_seconds = params.get("wait_seconds")
                if wait_seconds is None or wait_seconds == "":
                    wait_seconds_f = 1.0
                else:
                    wait_seconds_f = float(wait_seconds)
                session_id = params.get("session_id")
                take_id = params.get("take_id")
                output_root = params.get("output_root")

                inventory = get_inventory()
                cam = inventory.get(camera_id) or inventory.get_by_role(camera_id)
                if not cam:
                    raise ValueError("camera_not_found")

                ok, msg, transport = await asyncio.to_thread(
                    gopro_http.shutter_control,
                    cam,
                    "stop",
                    prefer_usb=False,
                    allow_usb=False,
                )
                stopped_at = _now()
                recording_id = params.get("recording_id") or params.get("id")
                recording_id = str(recording_id) if recording_id else None

                recording = None
                if recording_id:
                    recording = self._recordings.get(recording_id)
                    if recording and recording.camera_serial != cam.serial_number:
                        recording = None
                else:
                    active_id = self._active_recording_by_serial.get(cam.serial_number)
                    if active_id:
                        recording_id = active_id
                        recording = self._recordings.get(active_id)
                        if not recording:
                            self._active_recording_by_serial.pop(cam.serial_number, None)

                if not ok:
                    if recording:
                        recording.last_error = msg
                    return {
                        "camera": cam.serial_number,
                        "role": cam.role,
                        "transport": transport,
                        "ok": bool(ok),
                        "msg": msg,
                        "stopped_at": stopped_at,
                        "recording_id": recording_id,
                        "link_status": "failed",
                    }

                if recording:
                    recording.status = "pending"
                    recording.stopped_at = stopped_at
                    recording.transport = transport or recording.transport
                    if session_id:
                        recording.session_id = str(session_id)
                    if take_id:
                        recording.take_id = str(take_id)
                    if output_root:
                        recording.output_root = str(output_root)
                    recording.link_not_before_monotonic = time.monotonic() + max(0.0, wait_seconds_f)
                    if self._active_recording_by_serial.get(cam.serial_number) == recording.recording_id:
                        self._active_recording_by_serial.pop(cam.serial_number, None)
                    return {
                        "camera": cam.serial_number,
                        "role": cam.role,
                        "transport": transport,
                        "ok": True,
                        "msg": msg,
                        "stopped_at": stopped_at,
                        "recording_id": recording.recording_id,
                        "link_status": recording.status,
                    }

                return {
                    "camera": cam.serial_number,
                    "role": cam.role,
                    "transport": transport,
                    "ok": True,
                    "msg": "recording_not_tracked",
                    "stopped_at": stopped_at,
                    "recording_id": None,
                    "link_status": "untracked",
                }
            case "record_link_status":
                recording_id = str(params.get("recording_id") or params.get("id") or "")
                if not recording_id:
                    raise ValueError("recording_id_required")
                session_id = params.get("session_id")
                take_id = params.get("take_id")
                output_root = params.get("output_root")
                recording = self._recordings.get(recording_id)
                if not recording:
                    return {"ok": False, "msg": "recording_not_found", "recording_id": recording_id}

                inventory = get_inventory()
                cam = inventory.get(recording.camera_serial) or inventory.get_by_role(recording.camera_serial)
                fetch_job = None
                if cam and recording.status == "linked":
                    fetch_job = _build_fetch_job(
                        recording,
                        cam,
                        session_id=str(session_id) if session_id else None,
                        take_id=str(take_id) if take_id else None,
                        output_root=str(output_root) if output_root else None,
                    )

                return {
                    "ok": True,
                    "recording_id": recording.recording_id,
                    "camera": recording.camera_serial,
                    "role": recording.role,
                    "status": recording.status,
                    "ready": recording.status == "linked",
                    "started_at": recording.started_at,
                    "stopped_at": recording.stopped_at,
                    "media": recording.media,
                    "fetch_job": fetch_job or recording.fetch_job,
                    "link": {
                        "method": recording.link_method,
                        "attempts": recording.link_attempts,
                        "stable_polls": recording.stable_polls,
                        "baseline": recording.baseline.to_dict() if recording.baseline else None,
                        "baseline_timestamp": recording.baseline_timestamp,
                        "last_error": recording.last_error,
                        "last_check_at": recording.last_check_at,
                    },
                }
            case "stream_start":
                camera_id = str(params.get("camera") or "")
                if not camera_id:
                    raise ValueError("camera_required")
                port = params.get("port")
                if port is None or port == "":
                    port_i = 8554
                else:
                    port_i = int(port)
                watchdog_raw = params.get("watchdog")
                watchdog = True if watchdog_raw is None else bool(watchdog_raw)
                stall_timeout = params.get("stall_timeout")
                if stall_timeout is None or stall_timeout == "":
                    stall_timeout_f = 3.0
                else:
                    stall_timeout_f = float(stall_timeout)

                inventory = get_inventory()
                cam = inventory.get(camera_id) or inventory.get_by_role(camera_id)
                if not cam:
                    raise ValueError("camera_not_found")

                status, _transport = gopro_http.get_camera_state(
                    cam,
                    prefer_usb=False,
                    allow_usb=False,
                )
                preview_available = None
                if status is not None:
                    preview_raw = status.get("55")
                    try:
                        preview_val = int(preview_raw) if preview_raw is not None else None
                    except Exception:
                        preview_val = None
                    preview_available = (preview_val == 1) if preview_val is not None else None

                if preview_available is False:
                    return {
                        "camera": cam.serial_number,
                        "role": cam.role,
                        "port": port_i,
                        "ok": False,
                        "msg": "preview_unavailable",
                        "preview_available": preview_available,
                    }

                sock = self._stream_sockets.pop(cam.serial_number, None)
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass

                ok, msg, transport = await asyncio.to_thread(
                    gopro_http.start_stream,
                    cam,
                    port_i,
                    prefer_usb=False,
                    allow_usb=False,
                )

                stream = self._streams.get(cam.serial_number) or CameraStream()
                stream.active = bool(ok)
                stream.port = port_i
                stream.transport = transport
                stream.watchdog_enabled = bool(watchdog)
                stream.watchdog_active = False
                stream.stall_timeout_seconds = float(stall_timeout_f)
                stream.started_at = _now() if ok else None
                stream.stopped_at = None
                stream.last_error = None if ok else msg
                stream.first_packet_at = None
                stream.first_packet_at_monotonic = None
                stream.last_packet_at = None
                stream.last_packet_at_monotonic = None
                stream.last_restart_at = None
                self._streams[cam.serial_number] = stream

                return {
                    "camera": cam.serial_number,
                    "role": cam.role,
                    "transport": transport,
                    "port": port_i,
                    "ok": bool(ok),
                    "msg": msg,
                    "preview_available": preview_available,
                    "watchdog_enabled": bool(watchdog),
                    "stall_timeout_seconds": float(stall_timeout_f),
                    "started_at": stream.started_at,
                }
            case "stream_stop":
                camera_id = str(params.get("camera") or "")
                if not camera_id:
                    raise ValueError("camera_required")

                inventory = get_inventory()
                cam = inventory.get(camera_id) or inventory.get_by_role(camera_id)
                if not cam:
                    raise ValueError("camera_not_found")

                ok, msg, transport = await asyncio.to_thread(
                    gopro_http.stop_stream,
                    cam,
                    prefer_usb=False,
                    allow_usb=False,
                )

                stream = self._streams.get(cam.serial_number) or CameraStream()
                stream.active = False
                stream.transport = transport
                stream.watchdog_active = False
                stream.stopped_at = _now()
                stream.last_error = None if ok else msg
                stream.first_packet_at = None
                stream.first_packet_at_monotonic = None
                stream.last_packet_at = None
                stream.last_packet_at_monotonic = None
                self._streams[cam.serial_number] = stream
                sock = self._stream_sockets.pop(cam.serial_number, None)
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass

                return {
                    "camera": cam.serial_number,
                    "role": cam.role,
                    "transport": transport,
                    "ok": bool(ok),
                    "msg": msg,
                    "stopped_at": stream.stopped_at,
                }
            case "last_captured":
                camera_id = str(params.get("camera") or "")
                if not camera_id:
                    raise ValueError("camera_required")

                inventory = get_inventory()
                cam = inventory.get(camera_id) or inventory.get_by_role(camera_id)
                if not cam:
                    raise ValueError("camera_not_found")

                last = await asyncio.to_thread(
                    gopro_http.get_last_captured,
                    cam,
                    prefer_usb=False,
                    allow_usb=False,
                )
                return {
                    "camera": cam.serial_number,
                    "role": cam.role,
                    "folder": last.folder,
                    "file": last.file,
                    "gumi": last.gumi,
                    "type": last.media_type,
                }
            case "last_captured_resolved":
                camera_id = str(params.get("camera") or "")
                if not camera_id:
                    raise ValueError("camera_required")
                wait_seconds = params.get("wait_seconds")
                if wait_seconds is None or wait_seconds == "":
                    wait_seconds_f = 1.0
                else:
                    wait_seconds_f = float(wait_seconds)
                timeout_seconds = params.get("timeout_seconds") or params.get("timeout")
                if timeout_seconds is None or timeout_seconds == "":
                    timeout_seconds_f = gopro_http.DEFAULT_MEDIA_TIMEOUT
                else:
                    timeout_seconds_f = float(timeout_seconds)

                inventory = get_inventory()
                cam = inventory.get(camera_id) or inventory.get_by_role(camera_id)
                if not cam:
                    raise ValueError("camera_not_found")

                last = await asyncio.to_thread(
                    gopro_http.get_last_captured,
                    cam,
                    timeout=timeout_seconds_f,
                    prefer_usb=False,
                    allow_usb=False,
                )

                info = None
                if last.folder and last.file:
                    await asyncio.sleep(max(0.0, wait_seconds_f))
                    info = await asyncio.to_thread(
                        gopro_http.get_media_info,
                        cam,
                        last.folder,
                        last.file,
                        timeout=timeout_seconds_f,
                        prefer_usb=False,
                        allow_usb=False,
                    )

                return {
                    "camera": cam.serial_number,
                    "role": cam.role,
                    "folder": last.folder,
                    "file": last.file,
                    "gumi": last.gumi,
                    "type": last.media_type,
                    "info": info,
                }
            case "media_info":
                camera_id = str(params.get("camera") or "")
                folder = str(params.get("folder") or "")
                file = str(params.get("file") or "")
                if not camera_id:
                    raise ValueError("camera_required")
                if not folder or not file:
                    raise ValueError("folder_file_required")

                inventory = get_inventory()
                cam = inventory.get(camera_id) or inventory.get_by_role(camera_id)
                if not cam:
                    raise ValueError("camera_not_found")

                info = await asyncio.to_thread(
                    gopro_http.get_media_info,
                    cam,
                    folder,
                    file,
                    prefer_usb=False,
                    allow_usb=False,
                )
                return {
                    "camera": cam.serial_number,
                    "role": cam.role,
                    "folder": folder,
                    "file": file,
                    "info": info,
                }
            case "download_media":
                camera_id = str(params.get("camera") or "")
                folder = str(params.get("folder") or "")
                file = str(params.get("file") or "")
                dest_path = str(params.get("dest_path") or "")
                prefer_usb = bool(params.get("prefer_usb", False))
                allow_usb = bool(params.get("allow_usb", False))
                if not camera_id:
                    raise ValueError("camera_required")
                if not folder or not file:
                    raise ValueError("folder_file_required")
                if not dest_path:
                    raise ValueError("dest_path_required")

                inventory = get_inventory()
                cam = inventory.get(camera_id) or inventory.get_by_role(camera_id)
                if not cam:
                    raise ValueError("camera_not_found")

                ok, size_bytes, transport, err = await asyncio.to_thread(
                    gopro_http.download_media,
                    cam,
                    folder,
                    file,
                    Path(dest_path).expanduser(),
                    prefer_usb=prefer_usb,
                    allow_usb=allow_usb,
                )
                return {
                    "camera": cam.serial_number,
                    "role": cam.role,
                    "folder": folder,
                    "file": file,
                    "dest_path": dest_path,
                    "transport": transport,
                    "ok": bool(ok),
                    "size_bytes": size_bytes,
                    "error": err,
                }
            case "download_telemetry":
                camera_id = str(params.get("camera") or "")
                folder = str(params.get("folder") or "")
                file = str(params.get("file") or "")
                dest_path = str(params.get("dest_path") or "")
                prefer_usb = bool(params.get("prefer_usb", False))
                allow_usb = bool(params.get("allow_usb", False))
                if not camera_id:
                    raise ValueError("camera_required")
                if not folder or not file:
                    raise ValueError("folder_file_required")
                if not dest_path:
                    raise ValueError("dest_path_required")

                inventory = get_inventory()
                cam = inventory.get(camera_id) or inventory.get_by_role(camera_id)
                if not cam:
                    raise ValueError("camera_not_found")

                ok, size_bytes, transport, err = await asyncio.to_thread(
                    gopro_http.download_telemetry,
                    cam,
                    folder,
                    file,
                    Path(dest_path).expanduser(),
                    prefer_usb=prefer_usb,
                    allow_usb=allow_usb,
                )
                return {
                    "camera": cam.serial_number,
                    "role": cam.role,
                    "folder": folder,
                    "file": file,
                    "dest_path": dest_path,
                    "transport": transport,
                    "ok": bool(ok),
                    "size_bytes": size_bytes,
                    "error": err,
                }
            case _:
                raise ValueError(f"unknown_method:{method}")


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _iso_to_epoch(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return None


def _wifi_is_stale(last_ok_at: Optional[str], *, stale_seconds: float = WIFI_STALE_SECONDS) -> bool:
    if not last_ok_at:
        return False
    epoch = _iso_to_epoch(last_ok_at)
    if epoch is None:
        return False
    return (time.time() - epoch) > stale_seconds


def _build_fetch_job(
    rec: RecordingContext,
    cam: "gopro_http.CameraInventoryRecord",
    *,
    session_id: Optional[str] = None,
    take_id: Optional[str] = None,
    output_root: Optional[str] = None,
) -> Optional[dict]:
    media = rec.media or {}
    folder = media.get("folder")
    file = media.get("file")
    if not folder or not file:
        return None

    base_url, transport = gopro_http.get_http_base_url(
        cam,
        prefer_usb=False,
        allow_usb=False,
    )
    transport_value = rec.transport or transport
    media_path = f"{folder}/{file}"
    urls = None
    if base_url:
        urls = {
            "mp4": f"{base_url}/videos/DCIM/{folder}/{file}",
            "media_info": f"{base_url}/gopro/media/info?path={media_path}",
            "telemetry": f"{base_url}/gopro/media/telemetry?path={media_path}",
            "gpmf": f"{base_url}/gopro/media/gpmf?path={media_path}",
        }

    output_paths = None
    output_root_value = output_root or rec.output_root
    if output_root_value:
        root = Path(output_root_value)
        camera_id = cam.serial_number
        base_name = Path(file).stem
        output_paths = {
            "mp4": str(root / "recordings" / "gopro" / camera_id / file),
            "media_info": str(
                root
                / "metadata"
                / "vision"
                / "gopro"
                / camera_id
                / f"{base_name}.media_info.json"
            ),
            "telemetry": str(
                root
                / "metadata"
                / "telemetry"
                / "gopro"
                / camera_id
                / f"{base_name}.telemetry.gpmf.bin"
            ),
            "fetch_manifest": str(
                root
                / "metadata"
                / "telemetry"
                / "gopro"
                / camera_id
                / f"{base_name}.fetch_manifest.json"
            ),
        }

    info = media.get("info") or {}
    size_bytes = info.get("s") if isinstance(info, dict) else None
    return {
        "recording_id": rec.recording_id,
        "session_id": session_id or rec.session_id,
        "take_id": take_id or rec.take_id,
        "camera_id": cam.serial_number,
        "role": cam.role,
        "transport": transport_value,
        "camera_ip": cam.wifi.ip if cam.wifi and cam.wifi.ip else None,
        "base_url": base_url,
        "media_path": media_path,
        "folder": folder,
        "file": file,
        "size_bytes": int(size_bytes) if size_bytes is not None else None,
        "urls": urls,
        "output": output_paths,
        "created_at": _now(),
    }


def _same_capture(a: gopro_http.LastCaptured, b: gopro_http.LastCaptured) -> bool:
    if a.gumi and b.gumi and a.gumi == b.gumi:
        return True
    if a.folder and b.folder and a.file and b.file:
        return a.folder == b.folder and a.file == b.file
    return False


def _extract_media_timestamp(info: Optional[dict]) -> Optional[float]:
    if not info:
        return None
    for key in ("mod", "cre", "cts", "ts"):
        if key not in info:
            continue
        try:
            value = float(info[key])
        except (TypeError, ValueError):
            continue
        if value > 1_000_000_000_000:
            value = value / 1000.0
        return value
    return None


def request_gopro_daemon(
    *,
    method: str,
    params: Optional[dict[str, Any]] = None,
    socket_path: Path = DEFAULT_SOCKET_PATH,
    timeout: Optional[float] = 10.0,
) -> Any:
    """Request the GoPro daemon via Unix socket.

    Args:
        method: RPC method name
        params: Optional method parameters
        socket_path: Path to daemon socket
        timeout: Socket timeout in seconds (None = no timeout, default 10s)
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        if timeout is not None:
            sock.settimeout(timeout)
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


def start_gopro_daemon(
    *,
    socket_path: Path = DEFAULT_SOCKET_PATH,
    pid_path: Path = DEFAULT_PID_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
) -> int:
    pid = _read_pid(pid_path)
    if pid and _is_process_running(pid):
        return pid

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a")
    try:
        cmd = [
            sys.executable,
            "-m",
            "gopro.gopro_daemon",
            "run",
            "--socket",
            str(socket_path),
            "--pid",
            str(pid_path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    finally:
        log_file.close()

    _write_pid(pid_path, proc.pid)
    return proc.pid


def stop_gopro_daemon(*, socket_path: Path = DEFAULT_SOCKET_PATH, pid_path: Path = DEFAULT_PID_PATH) -> bool:
    try:
        request_gopro_daemon(method="shutdown", socket_path=socket_path)
    except Exception:
        pid = _read_pid(pid_path)
        if pid and _is_process_running(pid):
            os.kill(pid, signal.SIGTERM)
        return False

    deadline = time.time() + 5.0
    while time.time() < deadline:
        pid = _read_pid(pid_path)
        if not pid or not _is_process_running(pid):
            return True
        time.sleep(0.1)
    return False


def _parse_args(argv: list[str]) -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="GoPro manager daemon")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run daemon in foreground")
    run.add_argument("--socket", type=Path, default=DEFAULT_SOCKET_PATH)
    run.add_argument("--pid", type=Path, default=DEFAULT_PID_PATH)
    run.add_argument("--health-interval", type=float, default=HEALTH_POLL_SECONDS)
    run.add_argument("--ble-scan-interval", type=float, default=BLE_SCAN_SECONDS)

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    if args.command == "run":
        daemon = GoProDaemon(
            socket_path=args.socket,
            pid_path=args.pid,
            health_poll_seconds=args.health_interval,
            ble_scan_seconds=args.ble_scan_interval,
        )
        asyncio.run(daemon.run())
        return
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
