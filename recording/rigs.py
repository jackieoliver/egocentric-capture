from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class RigSpec:
    name: str
    device_kind: str
    roles: Optional[tuple[str, ...]] = None


DEFAULT_RIGS: dict[tuple[str, str], RigSpec] = {
    ("gopro", "default"): RigSpec(name="default", device_kind="gopro", roles=("HEAD", "LWRIST", "RWRIST")),
}


def _normalize_role(value: str) -> str:
    return value.strip().upper()


def _roles_from_status(cameras: Iterable[dict[str, object]]) -> tuple[str, ...]:
    roles: list[str] = []
    for cam in cameras:
        role = cam.get("role") if isinstance(cam, dict) else None
        if isinstance(role, str) and role.strip():
            roles.append(_normalize_role(role))
    return tuple(dict.fromkeys(roles))


def resolve_rig(device_kind: str, rig_name: Optional[str], cameras: Iterable[dict[str, object]]) -> RigSpec:
    if rig_name:
        key = (device_kind, rig_name)
        if key in DEFAULT_RIGS:
            return DEFAULT_RIGS[key]
        raise ValueError(f"unknown_rig:{rig_name}")

    if device_kind == "gopro":
        return DEFAULT_RIGS[("gopro", "default")]

    roles = _roles_from_status(cameras)
    return RigSpec(name="auto", device_kind=device_kind, roles=roles or None)
