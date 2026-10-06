from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Any, Dict

from .models import RecordingEntry


@dataclass(frozen=True)
class OffloadRequest:
    session_id: str
    take_number: int
    recording: RecordingEntry
    dest_dir: str


@dataclass(frozen=True)
class OffloadResult:
    ok: bool
    local_path: Optional[str] = None
    size_bytes: Optional[int] = None
    error: Optional[str] = None


class DeviceOffloader(Protocol):
    """
    Device-specific offloader.

    Implementations should:
    - retrieve the recording bytes (USB/WiFi/SD/etc.)
    - write into `dest_dir`
    - return the final `local_path`
    """

    def offload(self, request: OffloadRequest) -> OffloadResult: ...

    def offload_telemetry(
        self,
        request: OffloadRequest,
        *,
        dest_path: Path,
    ) -> Optional[Dict[str, Any]]: ...
