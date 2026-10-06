from __future__ import annotations

import hashlib
import mimetypes
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .drive import DriveClient, load_drive_config
from .identity import load_identity
from .registry import RegistryStore
from .store import SessionStore, get_session_store


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class UploadSummary:
    session_id: str
    session_uid: str
    drive_folder_id: str
    file_count: int
    total_bytes: int


class DriveUploader:
    def __init__(
        self,
        *,
        store: Optional[SessionStore] = None,
        registry: Optional[RegistryStore] = None,
        format_version: str = "v1",
    ) -> None:
        self.store = store or get_session_store()
        self.registry = registry or RegistryStore()
        self.config = load_drive_config()
        self.client = DriveClient(self.config)
        self.format_version = format_version

    def _ensure_upload_root(self) -> str:
        uploads_root = self.client.ensure_folder(self.config.root_folder_id, self.config.uploads_folder)
        return self.client.ensure_folder(uploads_root, self.format_version)

    def _ensure_path(self, cache: dict[str, str], root_id: str, parts: list[str]) -> str:
        current_id = root_id
        current_path = ""
        for part in parts:
            current_path = f"{current_path}/{part}" if current_path else part
            if current_path in cache:
                current_id = cache[current_path]
                continue
            current_id = self.client.ensure_folder(current_id, part)
            cache[current_path] = current_id
        return current_id

    def upload_session(
        self,
        session_id: str,
        *,
        include_recordings: bool = True,
    ) -> UploadSummary:
        session = self.store.load(session_id)
        if not session:
            raise ValueError("session_not_found")
        session_uid = session.session_uid or session.session_id

        session_dir = self.store.session_dir(session.session_id)
        if not session_dir.exists():
            raise ValueError("session_dir_missing")

        upload_root = self._ensure_upload_root()
        session_folder_id = self.client.ensure_folder(upload_root, session_uid)

        folder_cache: dict[str, str] = {}
        records: list[dict[str, Any]] = []
        total_bytes = 0
        identity = load_identity()

        for path in sorted(session_dir.rglob("*")):
            if path.is_dir():
                continue
            rel_path = path.relative_to(session_dir)
            if not include_recordings and rel_path.parts and rel_path.parts[0] == "takes":
                continue
            parent_parts = list(rel_path.parts[:-1])
            parent_id = self._ensure_path(folder_cache, session_folder_id, parent_parts) if parent_parts else session_folder_id
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            existing = self.client.find_child_by_name(parent_id, path.name)
            result = self.client.upload_file_path(
                parent_id=parent_id,
                name=path.name,
                file_path=path,
                content_type=mime_type,
                file_id=str(existing["id"]) if existing else None,
            )
            size_bytes = path.stat().st_size
            total_bytes += size_bytes
            sha256 = _sha256(path)
            upload_key = f"{session_uid}:{rel_path.as_posix()}:{sha256}"
            records.append(
                {
                    "upload_id": uuid.uuid4().hex,
                    "upload_key": upload_key,
                    "uploaded_at": _now_iso(),
                    "session_id": session.session_id,
                    "session_uid": session_uid,
                    "rig": session.rig,
                    "scene_id": session.scene_id,
                    "tactition_id": session.tactition_id,
                    "format_version": self.format_version,
                    "drive_file_id": result.get("id"),
                    "drive_parent_id": parent_id,
                    "relative_path": rel_path.as_posix(),
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                    "recorder_id": identity.recorder_id,
                    "operator_id": identity.operator_id,
                    "operator_name": identity.operator_name,
                    "registry_revision": session.registry_revision,
                }
            )

        if records:
            self.registry.append_uploads(records)

        self.store.append_event(
            session.session_id,
            "session_uploaded",
            {
                "drive_folder_id": session_folder_id,
                "format_version": self.format_version,
                "file_count": len(records),
                "total_bytes": total_bytes,
            },
        )
        self.store.save(session)
        return UploadSummary(
            session_id=session.session_id,
            session_uid=session_uid,
            drive_folder_id=session_folder_id,
            file_count=len(records),
            total_bytes=total_bytes,
        )
