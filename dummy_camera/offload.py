from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recording.offload import DeviceOffloader, OffloadRequest, OffloadResult

from .dummy_daemon import (
    default_socket_path,
    request_dummy_daemon,
    start_dummy_daemon,
)


@dataclass(frozen=True)
class DummyDaemonOffloader(DeviceOffloader):
    """
    Offload recordings via dummy daemon.
    """

    socket_path: Path = default_socket_path()

    def offload(self, request: OffloadRequest) -> OffloadResult:
        rec = request.recording
        if not rec.folder or not rec.file:
            return OffloadResult(ok=False, error="folder_file_required")

        dest_dir = Path(request.dest_dir).expanduser()
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / rec.file

        start_dummy_daemon(socket_path=self.socket_path)
        try:
            result = request_dummy_daemon(
                method="download_media",
                params={
                    "camera": rec.device_id,
                    "folder": rec.folder,
                    "file": rec.file,
                    "dest_path": str(dest_path),
                },
                socket_path=self.socket_path,
            )
        except Exception as exc:
            return OffloadResult(ok=False, error=str(exc))

        if not isinstance(result, dict):
            return OffloadResult(ok=False, error="invalid_daemon_response")
        if not result.get("ok"):
            return OffloadResult(ok=False, error=str(result.get("error") or "download_failed"))

        return OffloadResult(
            ok=True,
            local_path=str(dest_path),
            size_bytes=result.get("size_bytes"),
            error=None,
        )

    def offload_telemetry(self, request: OffloadRequest, *, dest_path: Path) -> dict:
        _ = (request, dest_path)
        return {"ok": False, "error": "telemetry_not_supported"}
