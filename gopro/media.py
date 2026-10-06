from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class LastCaptured:
    folder: Optional[str] = None
    file: Optional[str] = None
    gumi: Optional[str] = None
    media_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "folder": self.folder,
            "file": self.file,
            "gumi": self.gumi,
            "type": self.media_type,
        }


def parse_last_captured_response(data: Mapping[str, Any]) -> LastCaptured:
    folder = data.get("folder") or data.get("d")
    file = data.get("file") or data.get("n")
    gumi = data.get("gumi") or data.get("id") or data.get("media_id")
    media_type = data.get("type") or data.get("t")
    return LastCaptured(
        folder=str(folder) if folder else None,
        file=str(file) if file else None,
        gumi=str(gumi) if gumi else None,
        media_type=str(media_type) if media_type else None,
    )

