from __future__ import annotations

import getpass
import json
import os
import platform
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .paths import haptica_home


@dataclass
class RecorderIdentity:
    recorder_id: str
    operator_id: str
    operator_name: str
    device_name: str
    platform: str

    def to_dict(self) -> dict[str, str]:
        return {
            "recorder_id": self.recorder_id,
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "device_name": self.device_name,
            "platform": self.platform,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "RecorderIdentity":
        return cls(
            recorder_id=str(data.get("recorder_id") or uuid.uuid4().hex),
            operator_id=str(data.get("operator_id") or "unknown"),
            operator_name=str(data.get("operator_name") or "unknown"),
            device_name=str(data.get("device_name") or platform.node()),
            platform=str(data.get("platform") or platform.platform()),
        )


_IDENTITY_CACHE: Optional[RecorderIdentity] = None


def identity_path() -> Path:
    override = os.environ.get("TRANSCRIPTIONS_IDENTITY_PATH")
    if override:
        return Path(override).expanduser()
    return haptica_home() / "state" / "recorder.json"


def load_identity(*, create: bool = True) -> RecorderIdentity:
    global _IDENTITY_CACHE
    if _IDENTITY_CACHE is not None:
        return _IDENTITY_CACHE

    path = identity_path()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
        identity = RecorderIdentity.from_dict(data)
        _IDENTITY_CACHE = identity
        return identity

    if not create:
        identity = RecorderIdentity.from_dict({})
        _IDENTITY_CACHE = identity
        return identity

    default_name = os.environ.get("TRANSCRIPTIONS_OPERATOR_NAME") or getpass.getuser()
    default_id = os.environ.get("TRANSCRIPTIONS_OPERATOR_ID") or "unknown"
    identity = RecorderIdentity(
        recorder_id=uuid.uuid4().hex,
        operator_id=default_id,
        operator_name=default_name,
        device_name=platform.node(),
        platform=platform.platform(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    _IDENTITY_CACHE = identity
    return identity


def save_identity(identity: RecorderIdentity) -> None:
    global _IDENTITY_CACHE
    path = identity_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    _IDENTITY_CACHE = identity
