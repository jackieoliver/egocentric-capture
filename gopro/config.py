from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


from recording.paths import repo_root


def config_dir() -> Path:
    override = os.environ.get("HAPTICA_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return repo_root() / "configs"


CONFIG_DIR = config_dir()
COHN_CONFIG_PATH = CONFIG_DIR / "cohn.yaml"
COHN_DEFAULT_USERNAME = "gopro"
COHN_DEFAULT_NOTES = [
    "COHN credentials are per camera and generated during provisioning.",
    "Username is typically 'gopro'; password and certificate are provided by the camera.",
    "Reference the credential id in state/inventory.yaml (provisioning.cohn_credential_id).",
]


@dataclass
class NetworkConfig:
    ssid: str
    password: str
    mode: str = "sta"
    cohn_active: bool = True


@dataclass
class CohnCredentials:
    username: str
    password: str
    certificate: str
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "username": self.username,
            "password": self.password,
            "certificate": self.certificate,
        }
        if self.updated_at:
            data["updated_at"] = self.updated_at
        return data


def load_network_config() -> NetworkConfig:
    path = CONFIG_DIR / "network.yaml"
    if not path.exists():
        return NetworkConfig(ssid="", password="")
    data = yaml.safe_load(path.read_text()) or {}
    return NetworkConfig(
        ssid=str(data.get("ssid", "")),
        password=str(data.get("password", "")),
        mode=str(data.get("mode", "sta")),
        cohn_active=bool(data.get("cohn_active", True)),
    )


def load_profile(profile: str) -> Optional[Dict[str, Any]]:
    """Load a camera profile by name. Returns None if not found."""
    profile_key = profile.strip().lower()

    # Map profile names to config files
    profile_map = {
        "head": "hero13_head.yaml",
        "wrist": "hero13_wrist_left.yaml",
        "wrist_left": "hero13_wrist_left.yaml",
        "wrist_right": "hero13_wrist_right.yaml",
    }

    name = profile_map.get(profile_key)
    if not name:
        # Try direct filename
        name = f"{profile_key}.yaml" if not profile_key.endswith(".yaml") else profile_key

    path = CONFIG_DIR / name
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text()) or {}


def load_cohn_config(path: Optional[Path] = None) -> Dict[str, CohnCredentials]:
    config_path = path or COHN_CONFIG_PATH
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text()) or {}
    raw_credentials = data.get("credentials", {}) or {}
    credentials: Dict[str, CohnCredentials] = {}
    if isinstance(raw_credentials, dict):
        for key, value in raw_credentials.items():
            if not isinstance(value, dict):
                continue
            credentials[str(key)] = CohnCredentials(
                username=str(value.get("username", COHN_DEFAULT_USERNAME) or COHN_DEFAULT_USERNAME),
                password=str(value.get("password", "")),
                certificate=str(value.get("certificate", "")),
                updated_at=value.get("updated_at"),
            )
    return credentials


def save_cohn_config(
    credentials: Dict[str, CohnCredentials],
    path: Optional[Path] = None,
) -> None:
    config_path = path or COHN_CONFIG_PATH
    payload = {"credentials": {key: value.to_dict() for key, value in credentials.items()}}
    notes = None
    if config_path.exists():
        existing = yaml.safe_load(config_path.read_text()) or {}
        notes = existing.get("notes")
    if notes is None:
        notes = COHN_DEFAULT_NOTES
    payload["notes"] = notes
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def upsert_cohn_credentials(
    camera_id: str,
    credentials: CohnCredentials,
    path: Optional[Path] = None,
) -> None:
    config = load_cohn_config(path)
    config[str(camera_id)] = credentials
    save_cohn_config(config, path)


def get_cohn_credentials(
    camera_id: str,
    path: Optional[Path] = None,
) -> Optional[CohnCredentials]:
    return load_cohn_config(path).get(str(camera_id))
