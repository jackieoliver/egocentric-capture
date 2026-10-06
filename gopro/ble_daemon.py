from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .ble import (
    DEFAULT_STATUS_IDS,
    GoProBleConnection,
    LAST_PAIRING_SUCCESS_STATUS_ID,
    LAST_PAIRING_TYPE_STATUS_ID,
    PAIRING_STATE_STATUS_ID,
    WIFI_STATUS_IDS,
    interpret_wifi_status,
    scan_ble_devices,
)
from .cohn import is_cohn_ready, store_cohn_credentials
from .config import load_network_config
from .state import BLE_STATE_DIR, CameraInventoryRecord, ProvisioningState, STATE_DIR, get_inventory

logger = logging.getLogger(__name__)

DEFAULT_SOCKET_PATH = BLE_STATE_DIR / "ble_daemon.sock"
DEFAULT_PID_PATH = BLE_STATE_DIR / "ble_daemon.pid"
DEFAULT_LOG_PATH = BLE_STATE_DIR / "ble_daemon.log"
DEFAULT_STATE_PATH = BLE_STATE_DIR / "ble_daemon_state.json"
LEGACY_SOCKET_PATH = STATE_DIR / "ble_daemon.sock"
LEGACY_PID_PATH = STATE_DIR / "ble_daemon.pid"
LEGACY_LOG_PATH = STATE_DIR / "ble_daemon.log"
LEGACY_STATE_PATH = STATE_DIR / "ble_daemon_state.json"
STATE_SCHEMA_VERSION = 1
INVENTORY_SYNC_INTERVAL_SECONDS = 10.0
DEFAULT_SMOKE_STATUS_IDS = (
    8,
    10,
    17,
    PAIRING_STATE_STATUS_ID,
    LAST_PAIRING_TYPE_STATUS_ID,
    LAST_PAIRING_SUCCESS_STATUS_ID,
    33,
    70,
    82,
)
WIFI_VERIFY_RETRIES = 3
WIFI_VERIFY_DELAY_SECONDS = 1.0
CONNECT_SCAN_TIMEOUT_SECONDS = 6.0
PAIRING_RETRY_DELAY_SECONDS = 5.0


def _serialize_cohn_status(status: object | None) -> dict[str, Any] | None:
    if not status:
        return None
    def _to_int(value: object) -> int | None:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    return {
        "status": _to_int(getattr(status, "status", None)),
        "state": _to_int(getattr(status, "state", None)),
        "enabled": getattr(status, "enabled", None),
        "username": getattr(status, "username", None),
        "password": getattr(status, "password", None),
        "ipaddress": getattr(status, "ipaddress", None),
        "ssid": getattr(status, "ssid", None),
        "macaddress": getattr(status, "macaddress", None),
    }


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


def _migrate_ble_state() -> None:
    legacy_pid = _read_pid(LEGACY_PID_PATH)
    if legacy_pid and _is_process_running(legacy_pid):
        return
    BLE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    migrations = (
        (LEGACY_STATE_PATH, DEFAULT_STATE_PATH),
        (LEGACY_LOG_PATH, DEFAULT_LOG_PATH),
        (LEGACY_PID_PATH, DEFAULT_PID_PATH),
        (LEGACY_SOCKET_PATH, DEFAULT_SOCKET_PATH),
    )
    for old_path, new_path in migrations:
        if old_path.exists() and not new_path.exists():
            try:
                old_path.rename(new_path)
            except OSError:
                pass


def _resolve_socket_path(socket_path: Path) -> Path:
    if socket_path.exists():
        return socket_path
    if socket_path == DEFAULT_SOCKET_PATH and LEGACY_SOCKET_PATH.exists():
        return LEGACY_SOCKET_PATH
    return socket_path


def _resolve_camera(camera: str) -> Optional[CameraInventoryRecord]:
    inventory = get_inventory()
    return inventory.get_by_role(camera) or inventory.get(camera)


@dataclass
class ManagedBleCamera:
    serial: str
    identifier: str
    camera_name: Optional[str]
    address: Optional[str]
    desired: bool = True
    finish_pairing: bool = False
    claim_control: bool = False
    phone_name: str = "HapticaProvisioner"
    last_connected_at: Optional[str] = None
    last_error: Optional[str] = None
    next_connect_time: float = 0.0
    backoff_seconds: float = 1.0
    connection: Optional[GoProBleConnection] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def snapshot(self) -> dict[str, Any]:
        return {
            "serial": self.serial,
            "identifier": self.identifier,
            "camera_name": self.camera_name,
            "address": self.address,
            "desired": self.desired,
            "connected": bool(self.connection and self.connection.is_connected),
            "finish_pairing": self.finish_pairing,
            "claim_control": self.claim_control,
            "last_connected_at": self.last_connected_at,
            "last_error": self.last_error,
        }


class BleDaemon:
    def __init__(
        self,
        *,
        socket_path: Path = DEFAULT_SOCKET_PATH,
        pid_path: Path = DEFAULT_PID_PATH,
        state_path: Path = DEFAULT_STATE_PATH,
        maintenance_interval: float = 2.0,
        default_phone_name: str = "HapticaProvisioner",
    ) -> None:
        self.socket_path = socket_path
        self.pid_path = pid_path
        self.state_path = state_path
        self.maintenance_interval = maintenance_interval
        self.default_phone_name = default_phone_name
        self.started_at = _now()
        self._server: Optional[asyncio.AbstractServer] = None
        self._stop_event = asyncio.Event()
        self._maintenance_task: Optional[asyncio.Task] = None
        self._cameras: dict[str, ManagedBleCamera] = {}
        self._last_inventory_sync = 0.0
        self._last_scan_at = 0.0
        self._scan_results: list[Any] = []

    async def run(self) -> None:
        _migrate_ble_state()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()

        _write_pid(self.pid_path, os.getpid())
        self._load_state()
        self._sync_inventory()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop_event.set)
            except NotImplementedError:
                pass

        self._server = await asyncio.start_unix_server(self._handle_client, path=str(self.socket_path))
        self._maintenance_task = asyncio.create_task(self._maintenance_loop(), name="ble-daemon-maintenance")

        await self._stop_event.wait()
        await self.shutdown()

    async def shutdown(self) -> None:
        self._save_state()
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None

        for camera in list(self._cameras.values()):
            await self._disconnect_camera(camera)

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

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
        except Exception:
            return
        if data.get("schema_version") != STATE_SCHEMA_VERSION:
            return
        for item in data.get("cameras", []):
            serial = item.get("serial")
            if not serial:
                continue
            self._cameras[serial] = ManagedBleCamera(
                serial=serial,
                identifier=item.get("identifier") or str(serial)[-4:],
                camera_name=item.get("camera_name"),
                address=item.get("address"),
                desired=bool(item.get("desired", True)),
                finish_pairing=bool(item.get("finish_pairing", False)),
                claim_control=bool(item.get("claim_control", False)),
                phone_name=item.get("phone_name") or self.default_phone_name,
                last_connected_at=item.get("last_connected_at"),
                last_error=None,
            )

    def _save_state(self) -> None:
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "saved_at": _now(),
            "cameras": [
                {
                    "serial": cam.serial,
                    "identifier": cam.identifier,
                    "camera_name": cam.camera_name,
                    "address": cam.address,
                    "desired": cam.desired,
                    "finish_pairing": cam.finish_pairing,
                    "claim_control": cam.claim_control,
                    "phone_name": cam.phone_name,
                    "last_connected_at": cam.last_connected_at,
                    "last_error": cam.last_error,
                }
                for cam in self._cameras.values()
            ],
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, indent=2))

    def _inventory_record_for(self, camera: ManagedBleCamera) -> Optional[CameraInventoryRecord]:
        inventory = get_inventory()
        return inventory.get(camera.serial)

    async def _scan_for_camera(
        self,
        camera: ManagedBleCamera,
        results: Optional[list[Any]] = None,
    ) -> Optional[dict[str, Any]]:
        if results is None:
            results = await scan_ble_devices(timeout=CONNECT_SCAN_TIMEOUT_SECONDS)
        match = None
        for result in results:
            if camera.identifier and result.serial_suffix:
                if result.serial_suffix == camera.identifier or result.serial_suffix.endswith(camera.identifier):
                    match = result
                    break
            if camera.camera_name and result.name and camera.camera_name == result.name:
                match = result
                break
            if camera.identifier and result.name and camera.identifier in result.name:
                match = result
                break
        if not match:
            return None

        camera.address = match.address
        record = self._inventory_record_for(camera)
        if record and match.address and record.ble.address != match.address:
            record.ble.address = match.address
            inventory = get_inventory()
            inventory.upsert(record)
            inventory.save()
        return {
            "address": match.address,
            "pairing": match.pairing,
            "awake": match.awake,
            "wifi_ap_on": match.wifi_ap_on,
        }

    def _sync_inventory(self) -> dict[str, int]:
        inventory = get_inventory()
        added = 0
        updated = 0

        for record in inventory.cameras:
            if not record.serial_number:
                continue
            managed = self._cameras.get(record.serial_number)
            if not managed:
                self._cameras[record.serial_number] = ManagedBleCamera(
                    serial=record.serial_number,
                    identifier=record.short_id or record.serial_number[-4:],
                    camera_name=record.camera_name,
                    address=record.ble.address,
                    desired=bool(record.provisioning.ble_paired or record.ble.address),
                )
                added += 1
                updated += 1
                continue

            if record.camera_name and record.camera_name != managed.camera_name:
                managed.camera_name = record.camera_name
                updated += 1
            if record.ble.address and record.ble.address != managed.address:
                managed.address = record.ble.address
                updated += 1
            if (record.provisioning.ble_paired or record.ble.address) and not managed.desired:
                managed.desired = True
                updated += 1

        if updated:
            self._save_state()
        return {"added": added, "tracked": len(self._cameras)}

    def _ensure_discovered(self, record: CameraInventoryRecord, via: str) -> None:
        if not record.provisioning.discovered_at:
            record.provisioning.discovered_at = _now()
        if not record.provisioning.discovered_via:
            record.provisioning.discovered_via = via

    def _update_inventory_on_connect(self, camera: ManagedBleCamera, paired: bool) -> None:
        inventory = get_inventory()
        record = inventory.get(camera.serial)
        if not record:
            return
        record.ble.available = True
        if camera.address:
            record.ble.address = camera.address
        self._ensure_discovered(record, "ble")
        if paired:
            record.provisioning.ble_paired = True
            record.provisioning.ble_paired_at = _now()
            if record.provisioning.state in (ProvisioningState.UNKNOWN, ProvisioningState.IDENTIFIED):
                record.provisioning.state = ProvisioningState.BLE_PAIRED
        inventory.upsert(record)
        inventory.save()

    def _update_inventory_on_disconnect(self, camera: ManagedBleCamera) -> None:
        inventory = get_inventory()
        record = inventory.get(camera.serial)
        if not record:
            return
        record.ble.available = False
        inventory.upsert(record)
        inventory.save()
    async def _maintenance_loop(self) -> None:
        while not self._stop_event.is_set():
            now = asyncio.get_running_loop().time()
            if now - self._last_inventory_sync >= INVENTORY_SYNC_INTERVAL_SECONDS:
                self._sync_inventory()
                self._last_inventory_sync = now

            pending: list[ManagedBleCamera] = []
            for camera in list(self._cameras.values()):
                if not camera.desired:
                    continue
                if camera.connection and not camera.connection.is_connected:
                    await self._disconnect_camera(camera)
                if camera.connection and camera.connection.is_connected:
                    continue
                if now < camera.next_connect_time:
                    continue
                pending.append(camera)

            scan_results: Optional[list[Any]] = None
            if pending:
                try:
                    scan_results = await scan_ble_devices(timeout=CONNECT_SCAN_TIMEOUT_SECONDS)
                    self._scan_results = list(scan_results)
                    self._last_scan_at = now
                except Exception:
                    scan_results = None

            for camera in pending:
                await self._connect_camera(camera, scan_results)
            await asyncio.sleep(self.maintenance_interval)

    async def _connect_camera(
        self,
        camera: ManagedBleCamera,
        scan_results: Optional[list[Any]] = None,
    ) -> None:
        async with camera.lock:
            if camera.connection and camera.connection.is_connected:
                return
            record = self._inventory_record_for(camera)
            scan_info = None
            try:
                scan_info = await self._scan_for_camera(camera, scan_results)
            except Exception as exc:
                camera.last_error = f"scan_failed:{exc}"
                now = asyncio.get_running_loop().time()
                camera.next_connect_time = now + camera.backoff_seconds
                camera.backoff_seconds = min(camera.backoff_seconds * 2.0, 60.0)
                self._save_state()
                return

            if scan_info and scan_info.get("pairing") is False:
                camera.last_error = "not in pairing mode"
                now = asyncio.get_running_loop().time()
                camera.next_connect_time = now + PAIRING_RETRY_DELAY_SECONDS
                camera.backoff_seconds = min(camera.backoff_seconds, PAIRING_RETRY_DELAY_SECONDS)
                self._save_state()
                return

            if (
                camera.connection is None
                or camera.connection.identifier != camera.identifier
                or camera.connection.camera_name != camera.camera_name
            ):
                camera.connection = GoProBleConnection(
                    serial=camera.serial,
                    identifier=camera.identifier,
                    camera_name=camera.camera_name,
                    address=None,
                    phone_name=camera.phone_name,
                    auto_finish_pairing=True,
                    use_cached=None,
                )
            paired = False
            try:
                if not scan_info:
                    raise RuntimeError("not_advertising")
                await camera.connection.connect()
                if camera.finish_pairing or (record and not record.provisioning.ble_paired):
                    paired = await camera.connection.finish_pairing(camera.phone_name)
                if camera.claim_control:
                    await camera.connection.claim_control()
                camera.last_connected_at = _now()
                camera.last_error = None
                camera.backoff_seconds = 1.0
                camera.next_connect_time = 0.0
                self._update_inventory_on_connect(camera, paired=paired)
                self._save_state()
            except Exception as exc:
                note = f"{type(exc).__name__}: {exc}"
                if scan_info and scan_info.get("pairing") is False:
                    note = f"{note} (not in pairing mode)"
                if note != camera.last_error:
                    logger.warning("BLE connect failed for %s: %s", camera.serial, note, exc_info=True)
                camera.last_error = note
                camera.connection = None
                camera.address = None
                now = asyncio.get_running_loop().time()
                camera.next_connect_time = now + camera.backoff_seconds
                camera.backoff_seconds = min(camera.backoff_seconds * 2.0, 60.0)
                self._save_state()

    async def _disconnect_camera(self, camera: ManagedBleCamera) -> None:
        async with camera.lock:
            if camera.connection:
                try:
                    await camera.connection.disconnect()
                except Exception:
                    pass

    async def _verify_wifi_connection(
        self,
        connection: GoProBleConnection,
        expected_ssid: Optional[str],
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}
        for attempt in range(WIFI_VERIFY_RETRIES):
            values = await connection.get_status_values(list(WIFI_STATUS_IDS))
            info = interpret_wifi_status(values, expected_ssid)
            info["attempt"] = attempt + 1
            if info.get("verified"):
                return info
            if attempt < WIFI_VERIFY_RETRIES - 1:
                await asyncio.sleep(WIFI_VERIFY_DELAY_SECONDS)
        return info

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while not reader.at_eof():
            line = await reader.readline()
            if not line:
                break
            try:
                request = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                writer.write(
                    json.dumps({"ok": False, "error": "invalid_json", "id": None}).encode("utf-8") + b"\n"
                )
                await writer.drain()
                continue

            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params") or {}

            try:
                result = await self._dispatch(method, params)
                response = {"ok": True, "id": request_id, "result": result}
            except Exception as exc:
                response = {"ok": False, "id": request_id, "error": str(exc)}

            try:
                payload = json.dumps(response)
            except TypeError as exc:
                logger.exception("BLE daemon response encoding failed: %s", exc)
                response = {"ok": False, "id": request_id, "error": f"response_encode_failed:{exc}"}
                payload = json.dumps(response, default=str)

            writer.write(payload.encode("utf-8") + b"\n")
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
            case "shutdown":
                self._stop_event.set()
                return {"stopping": True}
            case "sync_inventory":
                return self._sync_inventory()
            case "connect":
                return await self._cmd_connect(params)
            case "disconnect":
                return await self._cmd_disconnect(params)
            case "wake":
                return await self._cmd_wake(params)
            case "set_camera_name":
                return await self._cmd_set_camera_name(params)
            case "get_status":
                return await self._cmd_get_status(params)
            case "cohn_status":
                return await self._cmd_cohn_status(params)
            case "wifi_join":
                return await self._cmd_wifi_join(params)
            case "wifi_release":
                return await self._cmd_wifi_release(params)
            case "cohn_enable":
                return await self._cmd_cohn_enable(params)
            case "smoke":
                return await self._cmd_smoke(params)
            case "sleep":
                return await self._cmd_sleep(params)
            case "wifi_ap":
                return await self._cmd_wifi_ap(params)
            case _:
                raise ValueError(f"unknown_method:{method}")

    def _status(self) -> dict[str, Any]:
        pid = _read_pid(self.pid_path)
        return {
            "pid": pid,
            "running": bool(pid and _is_process_running(pid)),
            "socket": str(self.socket_path),
            "state_path": str(self.state_path),
            "started_at": self.started_at,
            "cameras": [cam.snapshot() for cam in self._cameras.values()],
        }

    async def _cmd_connect(self, params: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(params.get("camera") or "")
        if not camera_id:
            raise ValueError("camera_required")

        record = _resolve_camera(camera_id)
        serial = record.serial_number if record else camera_id
        identifier = record.short_id if record else camera_id[-4:]
        camera_name = record.camera_name if record else None
        address = params.get("address")

        managed = self._cameras.get(serial)
        if not managed:
            managed = ManagedBleCamera(
                serial=serial,
                identifier=identifier,
                camera_name=camera_name,
                address=address,
            )
            self._cameras[serial] = managed

        managed.desired = True
        managed.finish_pairing = bool(params.get("finish_pairing", False))
        managed.claim_control = bool(params.get("claim_control", False))
        managed.phone_name = str(params.get("phone_name") or self.default_phone_name)
        if address:
            managed.address = str(address)

        self._save_state()
        await self._connect_camera(managed)
        return managed.snapshot()

    async def _cmd_disconnect(self, params: dict[str, Any]) -> dict[str, Any]:
        camera_id = params.get("camera")
        if camera_id in (None, "", "all"):
            for cam in list(self._cameras.values()):
                cam.desired = False
                await self._disconnect_camera(cam)
                self._update_inventory_on_disconnect(cam)
            self._save_state()
            return {"disconnected": "all"}

        record = _resolve_camera(str(camera_id))
        serial = record.serial_number if record else str(camera_id)
        cam = self._cameras.get(serial)
        if not cam:
            return {"disconnected": False, "reason": "not_tracked"}
        cam.desired = False
        await self._disconnect_camera(cam)
        self._update_inventory_on_disconnect(cam)
        self._save_state()
        return {"disconnected": True, "serial": serial}

    async def _cmd_wake(self, params: dict[str, Any]) -> dict[str, Any]:
        params = dict(params)
        params["claim_control"] = True
        return await self._cmd_connect(params)

    async def _cmd_set_camera_name(self, params: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(params.get("camera") or "")
        if not camera_id:
            raise ValueError("camera_required")

        name = str(params.get("name") or "")
        if not name:
            raise ValueError("name_required")

        timeout = float(params.get("timeout") or 20.0)

        record = _resolve_camera(camera_id)
        serial = record.serial_number if record else camera_id
        identifier = record.short_id if record else camera_id[-4:]
        camera_name = record.camera_name if record else None
        address = params.get("address") or (record.ble.address if record else None)
        claim_control = bool(params.get("claim_control", True))

        managed = self._cameras.get(serial)
        if not managed:
            managed = ManagedBleCamera(
                serial=serial,
                identifier=identifier,
                camera_name=camera_name,
                address=address,
                desired=bool(record and (record.provisioning.ble_paired or record.ble.address)),
            )
            self._cameras[serial] = managed

        if params.get("phone_name"):
            managed.phone_name = str(params.get("phone_name") or self.default_phone_name)
        if address:
            managed.address = str(address)

        if managed.connection is None or (address and managed.connection.address != address):
            managed.connection = GoProBleConnection(
                serial=managed.serial,
                identifier=managed.identifier,
                camera_name=managed.camera_name,
                address=managed.address,
                phone_name=managed.phone_name,
            )

        try:
            if not managed.connection.is_connected:
                scan_info = await self._scan_for_camera(managed)
                if not scan_info:
                    raise RuntimeError("not_advertising")
                if managed.connection.address != managed.address:
                    managed.connection.address = managed.address
                await asyncio.wait_for(managed.connection.connect(timeout=timeout, retries=1), timeout=timeout + 5.0)

            claim_ok: Optional[bool] = None
            claim_error: Optional[str] = None
            if claim_control:
                try:
                    claim_ok = await asyncio.wait_for(managed.connection.claim_control(), timeout=timeout)
                except Exception as exc:
                    claim_error = str(exc)

            previous_name = managed.camera_name or camera_name
            ok = await asyncio.wait_for(managed.connection.set_camera_name(name), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("timeout") from exc

        inventory_updated = False
        if ok:
            managed.camera_name = name
            try:
                managed.connection.camera_name = name
            except Exception:
                pass

            if record:
                record.camera_name = name
                self._ensure_discovered(record, "ble")
                inventory = get_inventory()
                inventory.upsert(record)
                inventory.save()
                inventory_updated = True
            self._save_state()

        result: dict[str, Any] = {
            "ok": ok,
            "previous_name": previous_name,
            "requested_name": name,
            "inventory_updated": inventory_updated,
            "claim_control_ok": claim_ok,
        }
        if claim_error:
            result["claim_control_error"] = claim_error
        result.update(managed.snapshot())
        return result

    async def _cmd_get_status(self, params: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(params.get("camera") or "")
        if not camera_id:
            raise ValueError("camera_required")
        record = _resolve_camera(camera_id)
        serial = record.serial_number if record else camera_id
        identifier = record.short_id if record else camera_id[-4:]
        camera_name = record.camera_name if record else None
        address = params.get("address")
        status_ids = params.get("status_ids") or list(DEFAULT_STATUS_IDS)

        managed = self._cameras.get(serial)
        if not managed:
            managed = ManagedBleCamera(
                serial=serial,
                identifier=identifier,
                camera_name=camera_name,
                address=address,
                desired=False,
            )
            self._cameras[serial] = managed
        if managed.connection is None or (address and managed.connection.address != address):
            managed.connection = GoProBleConnection(
                serial=managed.serial,
                identifier=managed.identifier,
                camera_name=managed.camera_name,
                address=address,
            )
        if not managed.connection.is_connected:
            await managed.connection.connect()
        data = await managed.connection.get_status_values([int(x) for x in status_ids])
        return {str(k): v.hex() for k, v in data.items()}

    async def _cmd_sleep(self, params: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(params.get("camera") or "")
        if not camera_id:
            raise ValueError("camera_required")

        record = _resolve_camera(camera_id)
        serial = record.serial_number if record else camera_id
        identifier = record.short_id if record else camera_id[-4:]
        camera_name = record.camera_name if record else None
        address = params.get("address") or (record.ble.address if record else None)

        keep_managed = bool(params.get("keep_managed", False))
        claim_control = bool(params.get("claim_control", True))

        managed = self._cameras.get(serial)
        if not managed:
            managed = ManagedBleCamera(
                serial=serial,
                identifier=identifier,
                camera_name=camera_name,
                address=address,
                desired=True,
            )
            self._cameras[serial] = managed

        if managed.connection is None:
            managed.connection = GoProBleConnection(
                serial=managed.serial,
                identifier=managed.identifier,
                camera_name=managed.camera_name,
                address=managed.address,
            )
        if not managed.connection.is_connected:
            await managed.connection.connect()
        if claim_control:
            try:
                await managed.connection.claim_control()
            except Exception:
                pass

        slept = await managed.connection.sleep_camera()

        await self._disconnect_camera(managed)
        self._update_inventory_on_disconnect(managed)

        managed.desired = keep_managed
        self._save_state()
        return {"slept": slept, "desired": managed.desired, **managed.snapshot()}

    async def _cmd_wifi_ap(self, params: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(params.get("camera") or "")
        if not camera_id:
            raise ValueError("camera_required")

        mode_raw = params.get("mode", 2)
        if isinstance(mode_raw, str):
            mode_text = mode_raw.strip().lower()
            mode_map = {"disable": 0, "off": 0, "enable": 1, "on": 1, "bounce": 2, "reset": 2}
            if mode_text not in mode_map:
                raise ValueError("mode_invalid")
            mode = mode_map[mode_text]
        else:
            mode = int(mode_raw)

        record = _resolve_camera(camera_id)
        serial = record.serial_number if record else camera_id
        identifier = record.short_id if record else camera_id[-4:]
        camera_name = record.camera_name if record else None
        address = params.get("address") or (record.ble.address if record else None)

        managed = self._cameras.get(serial)
        if not managed:
            managed = ManagedBleCamera(
                serial=serial,
                identifier=identifier,
                camera_name=camera_name,
                address=address,
                desired=True,
            )
            self._cameras[serial] = managed

        if managed.connection is None:
            managed.connection = GoProBleConnection(
                serial=managed.serial,
                identifier=managed.identifier,
                camera_name=managed.camera_name,
                address=managed.address,
            )
        if not managed.connection.is_connected:
            await managed.connection.connect()

        ok = await managed.connection.set_ap_control(mode)
        self._save_state()
        return {"mode": mode, "ok": ok, **managed.snapshot()}

    async def _cmd_wifi_release(self, params: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(params.get("camera") or "")
        if not camera_id:
            raise ValueError("camera_required")

        record = _resolve_camera(camera_id)
        serial = record.serial_number if record else camera_id
        identifier = record.short_id if record else camera_id[-4:]
        camera_name = record.camera_name if record else None
        address = params.get("address") or (record.ble.address if record else None)

        managed = self._cameras.get(serial)
        if not managed:
            managed = ManagedBleCamera(
                serial=serial,
                identifier=identifier,
                camera_name=camera_name,
                address=address,
                desired=True,
            )
            self._cameras[serial] = managed

        if managed.connection is None or (address and managed.connection.address != address):
            managed.connection = GoProBleConnection(
                serial=managed.serial,
                identifier=managed.identifier,
                camera_name=managed.camera_name,
                address=address,
            )
        if not managed.connection.is_connected:
            await managed.connection.connect()

        ok = await managed.connection.release_network()

        if record:
            record.wifi.available = False
            inventory = get_inventory()
            inventory.upsert(record)
            inventory.save()

        self._save_state()
        return {"ok": ok, **managed.snapshot()}

    async def _cmd_wifi_join(self, params: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(params.get("camera") or "")
        if not camera_id:
            raise ValueError("camera_required")

        record = _resolve_camera(camera_id)
        serial = record.serial_number if record else camera_id
        identifier = record.short_id if record else camera_id[-4:]
        camera_name = record.camera_name if record else None
        address = params.get("address")

        network = load_network_config()
        ssid = str(params.get("ssid") or network.ssid)
        password = str(params.get("password") or network.password)
        if not ssid or not password:
            raise ValueError("wifi_credentials_required")

        managed = self._cameras.get(serial)
        if not managed:
            managed = ManagedBleCamera(
                serial=serial,
                identifier=identifier,
                camera_name=camera_name,
                address=address,
                desired=True,
            )
            self._cameras[serial] = managed

        if managed.connection is None or (address and managed.connection.address != address):
            managed.connection = GoProBleConnection(
                serial=managed.serial,
                identifier=managed.identifier,
                camera_name=managed.camera_name,
                address=address,
            )
        if not managed.connection.is_connected:
            await managed.connection.connect()
        success = await managed.connection.connect_wifi(ssid, password)
        used_scan = True
        if not success:
            used_scan = False
            success = await managed.connection.connect_wifi(ssid, password, scan=False)

        status_info = await self._verify_wifi_connection(managed.connection, ssid)
        verified = bool(status_info.get("verified"))
        release_attempted = False
        if success and not verified:
            release_attempted = True
            try:
                await managed.connection.release_network()
            except Exception as exc:
                status_info["release_error"] = str(exc)
            success = await managed.connection.connect_wifi(ssid, password, scan=False)
            used_scan = False
            status_info = await self._verify_wifi_connection(managed.connection, ssid)
            verified = bool(status_info.get("verified"))

        if verified and record:
            record.provisioning.wifi_joined = True
            record.provisioning.wifi_ssid = ssid
            if record.provisioning.state == ProvisioningState.BLE_PAIRED:
                record.provisioning.state = ProvisioningState.WIFI_JOINED
            record.wifi.available = True
            self._ensure_discovered(record, "ble")
            inventory = get_inventory()
            inventory.upsert(record)
            inventory.save()
        return {
            "ssid": ssid,
            "joined": success,
            "verified": verified,
            "used_scan": used_scan,
            "release_attempted": release_attempted,
            "status": status_info,
        }

    async def _cmd_cohn_enable(self, params: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(params.get("camera") or "")
        if not camera_id:
            raise ValueError("camera_required")

        record = _resolve_camera(camera_id)
        if not record:
            raise ValueError("camera_not_found")
        serial = record.serial_number
        identifier = record.short_id
        camera_name = record.camera_name
        address = params.get("address")
        force = bool(params.get("force", True))

        managed = self._cameras.get(serial)
        if not managed:
            managed = ManagedBleCamera(
                serial=serial,
                identifier=identifier,
                camera_name=camera_name,
                address=address,
                desired=True,
            )
            self._cameras[serial] = managed

        if managed.connection is None or (address and managed.connection.address != address):
            managed.connection = GoProBleConnection(
                serial=managed.serial,
                identifier=managed.identifier,
                camera_name=managed.camera_name,
                address=address,
            )
        if not managed.connection.is_connected:
            await managed.connection.connect()

        created = await managed.connection.create_cohn_cert(override=force)
        if not created:
            return {"ready": False, "error": "cohn_cert_failed"}
        cert = await managed.connection.get_cohn_cert()
        status = await managed.connection.wait_for_cohn_network_connected(timeout=60.0)

        store_cohn_credentials(record, status, cert)
        ready = is_cohn_ready(status)
        if ready:
            record.provisioning.cohn_enabled = True
            if record.provisioning.state in (
                ProvisioningState.UNKNOWN,
                ProvisioningState.IDENTIFIED,
                ProvisioningState.BLE_PAIRED,
                ProvisioningState.WIFI_JOINED,
            ):
                record.provisioning.state = ProvisioningState.COHN_ENABLED
        self._ensure_discovered(record, "ble")
        inventory = get_inventory()
        inventory.upsert(record)
        inventory.save()

        return {"ready": ready, "ip": record.wifi.ip, "ssid": record.provisioning.wifi_ssid}

    async def _cmd_cohn_status(self, params: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(params.get("camera") or "")
        if not camera_id:
            raise ValueError("camera_required")

        record = _resolve_camera(camera_id)
        if not record:
            raise ValueError("camera_not_found")
        serial = record.serial_number
        identifier = record.short_id
        camera_name = record.camera_name
        address = params.get("address")

        managed = self._cameras.get(serial)
        if not managed:
            managed = ManagedBleCamera(
                serial=serial,
                identifier=identifier,
                camera_name=camera_name,
                address=address,
                desired=True,
            )
            self._cameras[serial] = managed

        if managed.connection is None or (address and managed.connection.address != address):
            managed.connection = GoProBleConnection(
                serial=managed.serial,
                identifier=managed.identifier,
                camera_name=managed.camera_name,
                address=address,
            )
        if not managed.connection.is_connected:
            await managed.connection.connect()

        status = await managed.connection.get_cohn_status(register_notifications=False)
        cert = await managed.connection.get_cohn_cert()
        store_cohn_credentials(record, status, cert)
        ready = is_cohn_ready(status)
        if ready:
            record.provisioning.cohn_enabled = True
            if record.provisioning.state in (
                ProvisioningState.UNKNOWN,
                ProvisioningState.IDENTIFIED,
                ProvisioningState.BLE_PAIRED,
                ProvisioningState.WIFI_JOINED,
            ):
                record.provisioning.state = ProvisioningState.COHN_ENABLED

        self._ensure_discovered(record, "ble")
        inventory = get_inventory()
        inventory.upsert(record)
        inventory.save()

        return {
            "ready": ready,
            "ip": record.wifi.ip,
            "ssid": record.provisioning.wifi_ssid,
            "status": _serialize_cohn_status(status),
        }

    async def _cmd_smoke(self, params: dict[str, Any]) -> dict[str, Any]:
        camera_id = str(params.get("camera") or "")
        if not camera_id:
            raise ValueError("camera_required")

        record = _resolve_camera(camera_id)
        serial = record.serial_number if record else camera_id
        identifier = record.short_id if record else camera_id[-4:]
        camera_name = record.camera_name if record else None
        address = params.get("address")
        status_ids = params.get("status_ids") or list(DEFAULT_SMOKE_STATUS_IDS)

        managed = self._cameras.get(serial)
        if not managed:
            managed = ManagedBleCamera(
                serial=serial,
                identifier=identifier,
                camera_name=camera_name,
                address=address,
                desired=True,
            )
            self._cameras[serial] = managed

        if managed.connection is None or (address and managed.connection.address != address):
            managed.connection = GoProBleConnection(
                serial=managed.serial,
                identifier=managed.identifier,
                camera_name=managed.camera_name,
                address=address,
            )
        if not managed.connection.is_connected:
            await managed.connection.connect()

        api_version = await managed.connection.ensure_open_gopro_api_version()
        data = await managed.connection.get_status_values([int(x) for x in status_ids])
        return {
            "api_version": api_version,
            "statuses": {str(k): v.hex() for k, v in data.items()},
        }


def request_daemon(
    *,
    method: str,
    params: Optional[dict[str, Any]] = None,
    socket_path: Path = DEFAULT_SOCKET_PATH,
    timeout: Optional[float] = None,
) -> Any:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        if timeout is not None:
            sock.settimeout(timeout)
        resolved_path = _resolve_socket_path(socket_path)
        sock.connect(str(resolved_path))
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


def start_daemon(
    *,
    socket_path: Path = DEFAULT_SOCKET_PATH,
    pid_path: Path = DEFAULT_PID_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
    phone_name: str = "HapticaProvisioner",
) -> int:
    _migrate_ble_state()
    pid = _read_pid(pid_path)
    if pid and _is_process_running(pid):
        return pid
    legacy_pid = _read_pid(LEGACY_PID_PATH)
    if legacy_pid and _is_process_running(legacy_pid):
        return legacy_pid

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a")
    try:
        cmd = [
            sys.executable,
            "-m",
            "gopro.ble_daemon",
            "run",
            "--socket",
            str(socket_path),
            "--pid",
            str(pid_path),
            "--phone-name",
            phone_name,
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


def stop_daemon(*, socket_path: Path = DEFAULT_SOCKET_PATH, pid_path: Path = DEFAULT_PID_PATH) -> bool:
    try:
        request_daemon(method="shutdown", socket_path=socket_path, timeout=3.0)
    except Exception:
        for candidate in (pid_path, LEGACY_PID_PATH):
            pid = _read_pid(candidate)
            if pid and _is_process_running(pid):
                os.kill(pid, signal.SIGTERM)
        return False

    deadline = time.time() + 5.0
    while time.time() < deadline:
        pid = _read_pid(pid_path)
        legacy_pid = _read_pid(LEGACY_PID_PATH)
        if (not pid or not _is_process_running(pid)) and (not legacy_pid or not _is_process_running(legacy_pid)):
            return True
        time.sleep(0.1)
    return False


def _parse_args(argv: list[str]) -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="GoPro BLE daemon")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run daemon in foreground")
    run.add_argument("--socket", type=Path, default=DEFAULT_SOCKET_PATH)
    run.add_argument("--pid", type=Path, default=DEFAULT_PID_PATH)
    run.add_argument("--phone-name", type=str, default="HapticaProvisioner")
    run.add_argument("--maintenance-interval", type=float, default=2.0)

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    if args.command == "run":
        daemon = BleDaemon(
            socket_path=args.socket,
            pid_path=args.pid,
            maintenance_interval=args.maintenance_interval,
            default_phone_name=args.phone_name,
        )
        asyncio.run(daemon.run())
        return
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
