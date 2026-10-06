from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as OAuthCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from .paths import haptica_home


DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

# Hardcoded Drive root folder for the Haptica recorder UI.
DEFAULT_DRIVE_ROOT_FOLDER_ID = "1fPJkOorfiv49zOzf4m8l-nThbfPrBwCo"


@dataclass
class DriveConfig:
    oauth_credentials_path: Optional[Path]
    oauth_token_path: Optional[Path]
    service_account_path: Optional[Path]
    root_folder_id: str
    drive_id: Optional[str] = None
    registry_folder: str = "registry"
    uploads_folder: str = "uploads"

def drive_config_path() -> Path:
    override = os.environ.get("HAPTICA_DRIVE_CONFIG_PATH")
    if override:
        return Path(override).expanduser()
    return haptica_home() / "state" / "drive_config.json"

def default_oauth_token_path() -> Path:
    return haptica_home() / "state" / "drive_token.json"


def save_drive_config(config: DriveConfig) -> None:
    path = drive_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "oauth_credentials_path": str(config.oauth_credentials_path) if config.oauth_credentials_path else None,
        "oauth_token_path": str(config.oauth_token_path) if config.oauth_token_path else None,
        "service_account_path": str(config.service_account_path) if config.service_account_path else None,
        "root_folder_id": config.root_folder_id,
        "drive_id": config.drive_id,
        "registry_folder": config.registry_folder,
        "uploads_folder": config.uploads_folder,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _load_drive_config_file() -> Optional[DriveConfig]:
    path = drive_config_path()
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8") or "{}")
    return DriveConfig(
        oauth_credentials_path=Path(raw["oauth_credentials_path"]).expanduser() if raw.get("oauth_credentials_path") else None,
        oauth_token_path=Path(raw["oauth_token_path"]).expanduser() if raw.get("oauth_token_path") else None,
        service_account_path=Path(raw["service_account_path"]).expanduser() if raw.get("service_account_path") else None,
        root_folder_id=str(raw.get("root_folder_id") or ""),
        drive_id=str(raw.get("drive_id")) if raw.get("drive_id") else None,
        registry_folder=str(raw.get("registry_folder") or "registry"),
        uploads_folder=str(raw.get("uploads_folder") or "uploads"),
    )


def load_drive_config() -> DriveConfig:
    oauth_creds = os.environ.get("TRANSCRIPTIONS_OAUTH_CREDENTIALS", "")
    oauth_token = os.environ.get("TRANSCRIPTIONS_OAUTH_TOKEN", "")
    service_creds = os.environ.get("TRANSCRIPTIONS_DRIVE_CREDENTIALS", "")
    root_id = os.environ.get("TRANSCRIPTIONS_DRIVE_ROOT_ID", "")
    drive_id = os.environ.get("TRANSCRIPTIONS_DRIVE_ID")
    registry_folder = os.environ.get("TRANSCRIPTIONS_DRIVE_REGISTRY_FOLDER", "registry")
    uploads_folder = os.environ.get("TRANSCRIPTIONS_DRIVE_UPLOADS_FOLDER", "uploads")
    if root_id and (oauth_creds or service_creds):
        config = DriveConfig(
            oauth_credentials_path=Path(oauth_creds).expanduser() if oauth_creds else None,
            oauth_token_path=Path(oauth_token).expanduser() if oauth_token else None,
            service_account_path=Path(service_creds).expanduser() if service_creds else None,
            root_folder_id=str(root_id),
            drive_id=str(drive_id) if drive_id else None,
            registry_folder=registry_folder,
            uploads_folder=uploads_folder,
        )
        config.root_folder_id = DEFAULT_DRIVE_ROOT_FOLDER_ID
        return config

    file_config = _load_drive_config_file()
    if file_config:
        file_config.root_folder_id = DEFAULT_DRIVE_ROOT_FOLDER_ID
        if file_config.oauth_credentials_path or file_config.service_account_path:
            return file_config

    service_account_default = haptica_home() / "state" / "drive_service_account.json"
    oauth_credentials_default = haptica_home() / "state" / "drive_oauth_credentials.json"
    oauth_token_default = haptica_home() / "state" / "drive_token.json"
    if service_account_default.exists() or oauth_credentials_default.exists():
        config = DriveConfig(
            oauth_credentials_path=oauth_credentials_default if oauth_credentials_default.exists() else None,
            oauth_token_path=oauth_token_default if oauth_token_default.exists() else None,
            service_account_path=service_account_default if service_account_default.exists() else None,
            root_folder_id=DEFAULT_DRIVE_ROOT_FOLDER_ID,
            drive_id=str(drive_id) if drive_id else None,
            registry_folder=registry_folder,
            uploads_folder=uploads_folder,
        )
        config.root_folder_id = DEFAULT_DRIVE_ROOT_FOLDER_ID
        return config

    raise RuntimeError("drive_config_missing")


class DriveClient:
    def __init__(self, config: DriveConfig) -> None:
        self.config = config
        creds = self._load_credentials()
        self.service = build("drive", "v3", credentials=creds, cache_discovery=False)

    def _load_credentials(self) -> Any:
        if self.config.oauth_credentials_path:
            token_path = self.config.oauth_token_path or (haptica_home() / "state" / "drive_token.json")
            token_path.parent.mkdir(parents=True, exist_ok=True)
            creds = None
            if token_path.exists():
                creds = OAuthCredentials.from_authorized_user_file(str(token_path), DRIVE_SCOPES)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.config.oauth_credentials_path),
                        DRIVE_SCOPES,
                    )
                    creds = flow.run_local_server(port=0)
                token_path.write_text(creds.to_json(), encoding="utf-8")
            return creds
        if self.config.service_account_path:
            return service_account.Credentials.from_service_account_file(
                str(self.config.service_account_path),
                scopes=DRIVE_SCOPES,
            )
        raise RuntimeError("drive_credentials_missing")

    def _drive_kwargs(self) -> dict[str, Any]:
        if self.config.drive_id:
            return {
                "supportsAllDrives": True,
                "includeItemsFromAllDrives": True,
                "corpora": "drive",
                "driveId": self.config.drive_id,
            }
        return {"corpora": "user"}

    def find_child_by_name(self, parent_id: str, name: str, mime_type: Optional[str] = None) -> Optional[dict[str, Any]]:
        mime_clause = f" and mimeType = '{mime_type}'" if mime_type else ""
        query = f"name = '{name}' and '{parent_id}' in parents and trashed = false{mime_clause}"
        result = (
            self.service.files()
            .list(
                q=query,
                fields="files(id, name, mimeType, modifiedTime, version, md5Checksum, size)",
                **self._drive_kwargs(),
            )
            .execute()
        )
        files = result.get("files") or []
        return files[0] if files else None

    def ensure_folder(self, parent_id: str, name: str) -> str:
        existing = self.find_child_by_name(parent_id, name, mime_type="application/vnd.google-apps.folder")
        if existing:
            return str(existing["id"])
        meta = {"name": name, "parents": [parent_id], "mimeType": "application/vnd.google-apps.folder"}
        created = (
            self.service.files()
            .create(body=meta, fields="id", supportsAllDrives=bool(self.config.drive_id))
            .execute()
        )
        return str(created["id"])

    def get_file_meta(self, file_id: str) -> dict[str, Any]:
        return (
            self.service.files()
            .get(
                fileId=file_id,
                fields="id, name, mimeType, modifiedTime, version, md5Checksum, size",
                supportsAllDrives=bool(self.config.drive_id),
            )
            .execute()
        )

    def download_file(self, file_id: str) -> bytes:
        request = self.service.files().get_media(fileId=file_id, supportsAllDrives=bool(self.config.drive_id))
        return request.execute()

    def download_file_path(
        self,
        *,
        file_id: str,
        dest_path: Path,
        overwrite: bool = False,
        chunk_size: int = 10 * 1024 * 1024,
    ) -> int:
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() and not overwrite:
            try:
                meta = self.get_file_meta(file_id)
                expected_size = meta.get("size")
                if expected_size is not None:
                    local_size = int(dest_path.stat().st_size)
                    if local_size == int(expected_size):
                        return local_size
            except Exception:
                pass

        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
        if tmp_path.exists():
            tmp_path.unlink()

        request = self.service.files().get_media(fileId=file_id, supportsAllDrives=bool(self.config.drive_id))
        try:
            with open(tmp_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            tmp_path.replace(dest_path)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise

        return int(dest_path.stat().st_size)

    def upload_file(
        self,
        *,
        parent_id: str,
        name: str,
        content_bytes: bytes,
        content_type: str = "application/json",
        file_id: Optional[str] = None,
    ) -> dict[str, Any]:
        from googleapiclient.http import MediaInMemoryUpload

        media = MediaInMemoryUpload(content_bytes, mimetype=content_type, resumable=False)
        body = {"name": name, "parents": [parent_id]}
        if file_id:
            return (
                self.service.files()
                .update(
                    fileId=file_id,
                    media_body=media,
                    fields="id, name, modifiedTime, version",
                    supportsAllDrives=bool(self.config.drive_id),
                )
                .execute()
            )
        return (
            self.service.files()
            .create(
                body=body,
                media_body=media,
                fields="id, name, modifiedTime, version",
                supportsAllDrives=bool(self.config.drive_id),
            )
            .execute()
        )

    def upload_file_path(
        self,
        *,
        parent_id: str,
        name: str,
        file_path: Path,
        content_type: Optional[str] = None,
        file_id: Optional[str] = None,
    ) -> dict[str, Any]:
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(
            str(file_path),
            mimetype=content_type,
            resumable=True,
        )
        body = {"name": name, "parents": [parent_id]}
        if file_id:
            return (
                self.service.files()
                .update(
                    fileId=file_id,
                    media_body=media,
                    fields="id, name, modifiedTime, version",
                    supportsAllDrives=bool(self.config.drive_id),
                )
                .execute()
            )
        return (
            self.service.files()
            .create(
                body=body,
                media_body=media,
                fields="id, name, modifiedTime, version",
                supportsAllDrives=bool(self.config.drive_id),
            )
            .execute()
        )
