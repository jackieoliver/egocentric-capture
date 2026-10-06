#!/usr/bin/env python3
"""
GoPro inventory state.

This module is intentionally GoPro-specific:
- Permanent inventory of known cameras (`state/inventory.yaml`)
- Provisioning progress (BLE/WiFi/COHN/Labs)
- Connection hints (USB/BLE/WiFi addresses)

Session/take recording state is device-agnostic and lives in `recording/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import yaml

from recording.paths import haptica_home


def state_dir() -> Path:
    override = os.environ.get("HAPTICA_STATE_DIR")
    if override:
        return Path(override).expanduser()
    return haptica_home() / "state"


STATE_DIR = state_dir()
INVENTORY_PATH = STATE_DIR / "inventory.yaml"
ACTIVE_SESSION_PATH = STATE_DIR / "active_session"
BLE_STATE_DIR = STATE_DIR / "ble"
QR_STATE_DIR = STATE_DIR / "qr"


class ProvisioningState(str, Enum):
    """Provisioning progress states."""

    UNKNOWN = "unknown"
    IDENTIFIED = "identified"
    BLE_PAIRED = "ble_paired"
    WIFI_JOINED = "wifi_joined"
    COHN_ENABLED = "cohn_enabled"
    SETTINGS_APPLIED = "settings_applied"
    LABS_APPLIED = "labs_applied"
    VERIFIED = "verified"
    PROVISIONED = "provisioned"


class CameraRole(str, Enum):
    """Role assignment for a camera in a rig."""

    HEAD = "HEAD"
    LWRIST = "LWRIST"
    RWRIST = "RWRIST"
    CHEST = "CHEST"
    BACK = "BACK"
    CUSTOM = "CUSTOM"


@dataclass
class ConnectionInfo:
    """Connection path info for a camera."""

    available: bool = False
    ip: Optional[str] = None
    address: Optional[str] = None  # BLE address
    port: Optional[int] = None  # COHN port

    def to_dict(self) -> Dict[str, Any]:
        d = {"available": self.available}
        if self.ip:
            d["ip"] = self.ip
        if self.address:
            d["address"] = self.address
        if self.port:
            d["port"] = self.port
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConnectionInfo":
        return cls(
            available=bool(data.get("available", False)),
            ip=data.get("ip"),
            address=data.get("address"),
            port=data.get("port"),
        )


@dataclass
class ProvisioningInfo:
    """Provisioning progress tracking."""

    state: ProvisioningState = ProvisioningState.UNKNOWN
    discovered_at: Optional[str] = None
    discovered_via: Optional[str] = None  # "usb" or "ble"
    ble_paired: bool = False
    ble_paired_at: Optional[str] = None
    wifi_joined: bool = False
    wifi_ssid: Optional[str] = None
    cohn_enabled: bool = False
    cohn_credential_id: Optional[str] = None
    cohn_cert_expires: Optional[str] = None
    settings_applied: bool = False
    labs_applied: bool = False
    labs_applied_at: Optional[str] = None
    settings_verified: bool = False
    verified_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "discovered_at": self.discovered_at,
            "discovered_via": self.discovered_via,
            "ble_paired": self.ble_paired,
            "ble_paired_at": self.ble_paired_at,
            "wifi_joined": self.wifi_joined,
            "wifi_ssid": self.wifi_ssid,
            "cohn_enabled": self.cohn_enabled,
            "cohn_credential_id": self.cohn_credential_id,
            "cohn_cert_expires": self.cohn_cert_expires,
            "settings_applied": self.settings_applied,
            "labs_applied": self.labs_applied,
            "labs_applied_at": self.labs_applied_at,
            "settings_verified": self.settings_verified,
            "verified_at": self.verified_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvisioningInfo":
        raw_state = data.get("state", "unknown")
        try:
            state = ProvisioningState(raw_state)
        except ValueError:
            state = ProvisioningState.UNKNOWN
        return cls(
            state=state,
            discovered_at=data.get("discovered_at"),
            discovered_via=data.get("discovered_via"),
            ble_paired=bool(data.get("ble_paired", False)),
            ble_paired_at=data.get("ble_paired_at"),
            wifi_joined=bool(data.get("wifi_joined", False)),
            wifi_ssid=data.get("wifi_ssid"),
            cohn_enabled=bool(data.get("cohn_enabled", False)),
            cohn_credential_id=data.get("cohn_credential_id"),
            cohn_cert_expires=data.get("cohn_cert_expires"),
            settings_applied=bool(data.get("settings_applied", False)),
            labs_applied=bool(data.get("labs_applied", False)),
            labs_applied_at=data.get("labs_applied_at"),
            settings_verified=bool(data.get("settings_verified", False)),
            verified_at=data.get("verified_at"),
        )


@dataclass
class CameraInventoryRecord:
    """Permanent GoPro record in inventory."""

    serial_number: str
    camera_name: Optional[str] = None
    model_name: Optional[str] = None
    model_number: Optional[str] = None
    firmware_version: Optional[str] = None
    ap_ssid: Optional[str] = None
    ap_mac_addr: Optional[str] = None

    role: Optional[str] = None
    profile: Optional[str] = None
    handle_id: Optional[int] = None
    udp_preview_port: Optional[int] = None

    provisioning: ProvisioningInfo = field(default_factory=ProvisioningInfo)
    usb: ConnectionInfo = field(default_factory=ConnectionInfo)
    ble: ConnectionInfo = field(default_factory=ConnectionInfo)
    wifi: ConnectionInfo = field(default_factory=ConnectionInfo)

    last_seen: Optional[str] = None

    @property
    def short_id(self) -> str:
        return self.serial_number[-4:] if self.serial_number else ""

    @property
    def is_provisioned(self) -> bool:
        return self.provisioning.state == ProvisioningState.PROVISIONED

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "serial_number": self.serial_number,
            "short_id": self.short_id,
            "provisioning": self.provisioning.to_dict(),
            "connections": {
                "usb": self.usb.to_dict(),
                "ble": self.ble.to_dict(),
                "wifi": self.wifi.to_dict(),
            },
        }
        if self.camera_name:
            d["camera_name"] = self.camera_name
        if self.model_name:
            d["model_name"] = self.model_name
        if self.model_number:
            d["model_number"] = self.model_number
        if self.firmware_version:
            d["firmware_version"] = self.firmware_version
        if self.ap_ssid:
            d["ap_ssid"] = self.ap_ssid
        if self.ap_mac_addr:
            d["ap_mac_addr"] = self.ap_mac_addr
        if self.role:
            d["role"] = self.role
        if self.profile:
            d["profile"] = self.profile
        if self.handle_id:
            d["handle_id"] = self.handle_id
        if self.udp_preview_port:
            d["udp_preview_port"] = self.udp_preview_port
        if self.last_seen:
            d["last_seen"] = self.last_seen
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraInventoryRecord":
        connections = data.get("connections", {}) or {}
        provisioning_data = data.get("provisioning", {}) or {}

        # Legacy format (states field).
        if "states" in data and not provisioning_data:
            states = data["states"] or {}
            provisioning_data = {
                "state": "provisioned" if data.get("provisioned") else "unknown",
                "ble_paired": states.get("ble_paired", False),
                "cohn_enabled": states.get("cohn_enabled", False),
            }

        return cls(
            serial_number=str(data.get("serial_number", "")),
            camera_name=data.get("camera_name"),
            model_name=data.get("model_name"),
            model_number=data.get("model_number"),
            firmware_version=data.get("firmware_version"),
            ap_ssid=data.get("ap_ssid"),
            ap_mac_addr=data.get("ap_mac_addr"),
            role=data.get("role"),
            profile=data.get("profile"),
            handle_id=data.get("handle_id"),
            udp_preview_port=data.get("udp_preview_port"),
            provisioning=ProvisioningInfo.from_dict(provisioning_data),
            usb=ConnectionInfo.from_dict(connections.get("usb", {}) or {}),
            ble=ConnectionInfo.from_dict(connections.get("ble", {}) or {}),
            wifi=ConnectionInfo.from_dict(connections.get("wifi", {}) or {}),
            last_seen=data.get("last_seen"),
        )


class Inventory:
    """Manages the permanent camera inventory."""

    def __init__(self, path: Path = INVENTORY_PATH):
        self.path = path
        self._cameras: Dict[str, CameraInventoryRecord] = {}
        self._generated_at: Optional[str] = None
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._cameras = {}
            self._generated_at = None
            return

        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self._generated_at = data.get("generated_at")

        self._cameras = {}
        for cam_data in data.get("cameras", []) or []:
            record = CameraInventoryRecord.from_dict(cam_data)
            self._cameras[record.serial_number] = record

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "generated_at": _now(),
            "source": "gopro_cli",
            "cameras": [cam.to_dict() for cam in self._cameras.values()],
        }
        self.path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @property
    def cameras(self) -> List[CameraInventoryRecord]:
        return list(self._cameras.values())

    def get(self, serial: str) -> Optional[CameraInventoryRecord]:
        if not serial:
            return None
        if serial in self._cameras:
            return self._cameras[serial]
        matches = [
            cam
            for cam in self._cameras.values()
            if cam.serial_number
            and (cam.serial_number.endswith(serial) or serial.endswith(cam.serial_number))
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def get_by_role(self, role: str) -> Optional[CameraInventoryRecord]:
        for cam in self._cameras.values():
            if cam.role and cam.role.upper() == role.upper():
                return cam
        return None

    def upsert(self, record: CameraInventoryRecord) -> None:
        record.last_seen = _now()
        self._cameras[record.serial_number] = record

    def remove(self, serial: str) -> bool:
        if serial in self._cameras:
            del self._cameras[serial]
            return True
        return False

    def provisioned_cameras(self) -> List[CameraInventoryRecord]:
        return [c for c in self._cameras.values() if c.is_provisioned]


def _now() -> str:
    return datetime.now().isoformat(timespec="microseconds")


def get_inventory() -> Inventory:
    return Inventory()
