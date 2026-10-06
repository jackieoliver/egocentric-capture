from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from recording.offload import DeviceOffloader, OffloadRequest, OffloadResult

from .gopro_daemon import (
    DEFAULT_SOCKET_PATH as GOPRO_DAEMON_SOCKET_PATH,
    request_gopro_daemon,
    start_gopro_daemon,
)


@dataclass(frozen=True)
class GoProDaemonOffloader(DeviceOffloader):
    """
    Offload recordings via `gopro-daemon`.

    The daemon owns the device-specific transfer mechanism; this class is the
    device-agnostic plug-in used by `recording.IngestManager`.
    """

    socket_path: Path = GOPRO_DAEMON_SOCKET_PATH
    prefer_usb: bool = False
    allow_usb: bool = False

    def offload(self, request: OffloadRequest) -> OffloadResult:
        rec = request.recording
        if not rec.folder or not rec.file:
            return OffloadResult(ok=False, error="folder_file_required")

        dest_dir = Path(request.dest_dir).expanduser()
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / rec.file

        start_gopro_daemon(socket_path=self.socket_path)
        try:
            result = request_gopro_daemon(
                method="download_media",
                params={
                    "camera": rec.device_id,
                    "folder": rec.folder,
                    "file": rec.file,
                    "dest_path": str(dest_path),
                    "prefer_usb": bool(self.prefer_usb),
                    "allow_usb": bool(self.allow_usb),
                },
                socket_path=self.socket_path,
            )
        except Exception as exc:
            return OffloadResult(ok=False, error=str(exc))

        if not isinstance(result, dict):
            return OffloadResult(ok=False, error="invalid_daemon_response")

        if not result.get("ok"):
            return OffloadResult(ok=False, error=str(result.get("error") or "download_failed"))

        local_path = str(dest_path)
        return OffloadResult(
            ok=True,
            local_path=local_path,
            size_bytes=result.get("size_bytes"),
            error=None,
        )

    def offload_telemetry(self, request: OffloadRequest, *, dest_path: Path) -> Optional[dict]:
        rec = request.recording
        if not rec.folder or not rec.file:
            return {"ok": False, "error": "folder_file_required"}

        dest_path = Path(dest_path).expanduser()
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        start_gopro_daemon(socket_path=self.socket_path)
        try:
            result = request_gopro_daemon(
                method="download_telemetry",
                params={
                    "camera": rec.device_id,
                    "folder": rec.folder,
                    "file": rec.file,
                    "dest_path": str(dest_path),
                    "prefer_usb": bool(self.prefer_usb),
                    "allow_usb": bool(self.allow_usb),
                },
                socket_path=self.socket_path,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        if not isinstance(result, dict):
            return {"ok": False, "error": "invalid_daemon_response"}
        if not result.get("ok"):
            return {"ok": False, "error": str(result.get("error") or "download_failed")}

        return {
            "ok": True,
            "local_path": str(dest_path),
            "size_bytes": result.get("size_bytes"),
        }
