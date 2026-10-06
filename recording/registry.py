from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .drive import DriveClient, load_drive_config
from .paths import registry_cache_dir
from .rigs import RigSpec


REGISTRY_SCHEMA = "transcriptions.registry"
REGISTRY_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass
class RegistryData:
    schema: str = REGISTRY_SCHEMA
    schema_version: int = REGISTRY_VERSION
    updated_at: str = field(default_factory=_now_iso)
    rigs: list[dict[str, Any]] = field(default_factory=list)
    tactitions: list[dict[str, Any]] = field(default_factory=list)
    scenes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "updated_at": self.updated_at,
            "rigs": self.rigs,
            "tactitions": self.tactitions,
            "scenes": self.scenes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegistryData":
        return cls(
            schema=str(data.get("schema") or REGISTRY_SCHEMA),
            schema_version=int(data.get("schema_version") or REGISTRY_VERSION),
            updated_at=str(data.get("updated_at") or _now_iso()),
            rigs=list(data.get("rigs") or []),
            tactitions=list(data.get("tactitions") or []),
            scenes=list(data.get("scenes") or []),
        )


@dataclass
class RegistryMeta:
    registry_file_id: Optional[str] = None
    registry_version: Optional[str] = None
    uploads_file_id: Optional[str] = None
    uploads_version: Optional[str] = None
    registry_folder_id: Optional[str] = None
    last_sync: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_file_id": self.registry_file_id,
            "registry_version": self.registry_version,
            "uploads_file_id": self.uploads_file_id,
            "uploads_version": self.uploads_version,
            "registry_folder_id": self.registry_folder_id,
            "last_sync": self.last_sync,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegistryMeta":
        return cls(
            registry_file_id=data.get("registry_file_id"),
            registry_version=data.get("registry_version"),
            uploads_file_id=data.get("uploads_file_id"),
            uploads_version=data.get("uploads_version"),
            registry_folder_id=data.get("registry_folder_id"),
            last_sync=data.get("last_sync"),
        )


class RegistryStore:
    def __init__(self) -> None:
        self.config = load_drive_config()
        self.drive = DriveClient(self.config)
        self.cache_dir = registry_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.cache_dir / "registry_meta.json"
        self.registry_path = self.cache_dir / "registry.json"
        self.uploads_path = self.cache_dir / "uploads.jsonl"

    def _load_meta(self) -> RegistryMeta:
        if not self.meta_path.exists():
            return RegistryMeta()
        return RegistryMeta.from_dict(json.loads(self.meta_path.read_text(encoding="utf-8") or "{}"))

    def _save_meta(self, meta: RegistryMeta) -> None:
        self.meta_path.write_text(json.dumps(meta.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")

    def _ensure_registry_folder(self, meta: RegistryMeta) -> str:
        if meta.registry_folder_id:
            return meta.registry_folder_id
        folder_id = self.drive.ensure_folder(self.config.root_folder_id, self.config.registry_folder)
        meta.registry_folder_id = folder_id
        self._save_meta(meta)
        return folder_id

    def _default_registry(self) -> RegistryData:
        return RegistryData(
            rigs=[
                {
                    "rig_id": "rig_gopro_default",
                    "name": "GoPro Default",
                    "device_kind": "gopro",
                    "roles": ["HEAD", "LWRIST", "RWRIST"],
                }
            ],
            tactitions=[],
            scenes=[],
        )

    def pull(self, *, force: bool = False) -> RegistryData:
        meta = self._load_meta()
        folder_id = self._ensure_registry_folder(meta)

        registry_file = None
        if meta.registry_file_id:
            registry_file = self.drive.get_file_meta(meta.registry_file_id)
        if not registry_file:
            registry_file = self.drive.find_child_by_name(folder_id, "registry.json")

        if not registry_file:
            data = self._default_registry()
            uploaded = self.drive.upload_file(
                parent_id=folder_id,
                name="registry.json",
                content_bytes=json.dumps(data.to_dict(), indent=2, sort_keys=False).encode("utf-8"),
            )
            meta.registry_file_id = uploaded.get("id")
            meta.registry_version = uploaded.get("version")
            meta.last_sync = _now_iso()
            self._save_meta(meta)
            self.registry_path.write_text(json.dumps(data.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
            return data

        registry_version = str(registry_file.get("version") or "")
        if force or registry_version != meta.registry_version or not self.registry_path.exists():
            content = self.drive.download_file(str(registry_file["id"]))
            self.registry_path.write_bytes(content)
            meta.registry_file_id = str(registry_file["id"])
            meta.registry_version = registry_version
            meta.last_sync = _now_iso()
            self._save_meta(meta)

        return self.load(_skip_pull=True)

    def load(self, *, _skip_pull: bool = False) -> RegistryData:
        if not self.registry_path.exists():
            if _skip_pull:
                return self._default_registry()
            return self.pull(force=True)
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            if _skip_pull:
                return self._default_registry()
            return self.pull(force=True)
        return RegistryData.from_dict(data)

    def save(self, data: RegistryData) -> None:
        self.registry_path.write_text(json.dumps(data.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")

    def push(self) -> RegistryData:
        meta = self._load_meta()
        folder_id = self._ensure_registry_folder(meta)
        data = self.load()
        data.updated_at = _now_iso()
        payload = json.dumps(data.to_dict(), indent=2, sort_keys=False).encode("utf-8")
        uploaded = self.drive.upload_file(
            parent_id=folder_id,
            name="registry.json",
            content_bytes=payload,
            file_id=meta.registry_file_id,
        )
        meta.registry_file_id = uploaded.get("id")
        meta.registry_version = uploaded.get("version")
        meta.last_sync = _now_iso()
        self._save_meta(meta)
        self.registry_path.write_text(payload.decode("utf-8") + "\n", encoding="utf-8")
        return data

    def registry_revision(self) -> Optional[str]:
        meta = self._load_meta()
        return meta.registry_version

    def list_rigs(self) -> list[dict[str, Any]]:
        return self.load().rigs

    def list_tactitions(self) -> list[dict[str, Any]]:
        return self.load().tactitions

    def list_scenes(self) -> list[dict[str, Any]]:
        return self.load().scenes

    def get_rig(self, rig_id: str) -> Optional[dict[str, Any]]:
        for rig in self.load().rigs:
            if rig.get("rig_id") == rig_id:
                return rig
        return None

    def get_tactition(self, tactition_id: str) -> Optional[dict[str, Any]]:
        for tactition in self.load().tactitions:
            if tactition.get("tactition_id") == tactition_id:
                return tactition
        return None

    def get_scene(self, scene_id: str) -> Optional[dict[str, Any]]:
        for scene in self.load().scenes:
            if scene.get("scene_id") == scene_id:
                return scene
        return None

    def rig_spec_from_registry(self, rig: dict[str, Any]) -> RigSpec:
        device_kind = str(rig.get("device_kind") or "unknown")
        roles = [str(role).upper() for role in (rig.get("roles") or []) if role]
        return RigSpec(
            name=str(rig.get("rig_id") or rig.get("name") or "rig"),
            device_kind=device_kind,
            roles=tuple(roles) if roles else None,
        )

    def append_uploads(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        meta = self._load_meta()
        folder_id = self._ensure_registry_folder(meta)

        uploads_file = None
        if meta.uploads_file_id:
            uploads_file = self.drive.get_file_meta(meta.uploads_file_id)
        if not uploads_file:
            uploads_file = self.drive.find_child_by_name(folder_id, "uploads.jsonl")

        if uploads_file:
            remote_version = str(uploads_file.get("version") or "")
            if remote_version != meta.uploads_version or not self.uploads_path.exists():
                content = self.drive.download_file(str(uploads_file["id"]))
                self.uploads_path.write_bytes(content)
                meta.uploads_version = remote_version
                meta.uploads_file_id = str(uploads_file["id"])
                self._save_meta(meta)

        pending = list(records)
        retries = 3
        while retries > 0:
            retries -= 1
            seen_keys = self._load_upload_keys()
            new_records = [r for r in pending if str(r.get("upload_key") or "") not in seen_keys]
            if not new_records:
                return
            tmp_path = self.uploads_path.with_suffix(self.uploads_path.suffix + ".tmp")
            try:
                existing = ""
                if self.uploads_path.exists():
                    existing = self.uploads_path.read_text(encoding="utf-8")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    if existing:
                        f.write(existing.rstrip("\n") + "\n")
                    for record in new_records:
                        line = json.dumps(record, sort_keys=False)
                        f.write(line + "\n")
                payload = tmp_path.read_bytes()
                uploaded = self.drive.upload_file(
                    parent_id=folder_id,
                    name="uploads.jsonl",
                    content_bytes=payload,
                    content_type="application/json",
                    file_id=meta.uploads_file_id,
                )
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
            tmp_path.replace(self.uploads_path)
            remote_version = str(uploaded.get("version") or "")
            meta.uploads_file_id = uploaded.get("id")
            meta.uploads_version = remote_version
            meta.last_sync = _now_iso()
            self._save_meta(meta)

            # Re-fetch to confirm no concurrent update overwrote our upload.
            updated_meta = self.drive.get_file_meta(str(meta.uploads_file_id))
            latest_version = str(updated_meta.get("version") or "")
            if latest_version == meta.uploads_version:
                return
            content = self.drive.download_file(str(meta.uploads_file_id))
            self.uploads_path.write_bytes(content)
            meta.uploads_version = latest_version
            self._save_meta(meta)
            pending = new_records

    def _load_upload_keys(self) -> set[str]:
        if not self.uploads_path.exists():
            return set()
        keys: set[str] = set()
        with open(self.uploads_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(record.get("upload_key") or "")
                if key:
                    keys.add(key)
        return keys

    def load_uploads(self, *, force: bool = False) -> list[dict[str, Any]]:
        meta = self._load_meta()
        folder_id = self._ensure_registry_folder(meta)
        uploads_file = None
        if meta.uploads_file_id:
            uploads_file = self.drive.get_file_meta(meta.uploads_file_id)
        if not uploads_file:
            uploads_file = self.drive.find_child_by_name(folder_id, "uploads.jsonl")
        if uploads_file:
            remote_version = str(uploads_file.get("version") or "")
            if force or remote_version != meta.uploads_version or not self.uploads_path.exists():
                content = self.drive.download_file(str(uploads_file["id"]))
                self.uploads_path.write_bytes(content)
                meta.uploads_version = remote_version
                meta.uploads_file_id = str(uploads_file["id"])
                meta.last_sync = _now_iso()
                self._save_meta(meta)

        records: list[dict[str, Any]] = []
        if not self.uploads_path.exists():
            return records
        with open(self.uploads_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(record)
        return records

    def append_upload(self, record: dict[str, Any]) -> None:
        self.append_uploads([record])
