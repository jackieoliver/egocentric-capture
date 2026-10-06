from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Optional


@dataclass(frozen=True)
class SdCardStatus:
    present: Optional[bool]
    remaining_kib: Optional[int]
    remaining_bytes: Optional[int]
    remaining_gib: Optional[float]
    remaining_gb: Optional[float]
    full: Optional[bool]
    status: str


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _kib_to_bytes(value_kib: int) -> int:
    return int(value_kib) * 1024


def _bytes_to_gib(value_bytes: int) -> float:
    return float(value_bytes) / (1024.0**3)


def _bytes_to_gb(value_bytes: int) -> float:
    return float(value_bytes) / (1000.0**3)


def sd_card_status_from_values(
    *,
    present: Optional[bool],
    remaining_kib: Optional[int],
    full_threshold_kib: int = 0,
) -> SdCardStatus:
    if remaining_kib is not None and remaining_kib < 0:
        remaining_kib = None

    if present is False:
        remaining_kib = None

    remaining_bytes = _kib_to_bytes(remaining_kib) if remaining_kib is not None else None
    remaining_gib = _bytes_to_gib(remaining_bytes) if remaining_bytes is not None else None
    remaining_gb = _bytes_to_gb(remaining_bytes) if remaining_bytes is not None else None

    full: Optional[bool] = None
    if present is True and remaining_kib is not None:
        full = remaining_kib <= int(full_threshold_kib)

    if present is False:
        status_str = "missing"
    elif present is True:
        if remaining_kib is None:
            status_str = "unknown"
        elif full is True:
            status_str = "full"
        else:
            status_str = "ok"
    else:
        status_str = "unknown"

    return SdCardStatus(
        present=present,
        remaining_kib=remaining_kib,
        remaining_bytes=remaining_bytes,
        remaining_gib=remaining_gib,
        remaining_gb=remaining_gb,
        full=full,
        status=status_str,
    )


def parse_sd_card_status(
    status: Mapping[str, Any],
    *,
    full_threshold_kib: int = 0,
) -> SdCardStatus:
    """
    Parse GoPro `/gopro/camera/state` SD card fields.

    - `status["33"]` is SD present flag (0 => present)
    - `status["54"]` is free space in KiB (per existing usage in this repo)
    """
    present_raw = _safe_int(status.get("33"))
    present: Optional[bool] = (present_raw == 0) if present_raw is not None else None
    remaining_kib = _safe_int(status.get("54"))
    return sd_card_status_from_values(
        present=present,
        remaining_kib=remaining_kib,
        full_threshold_kib=full_threshold_kib,
    )


def sd_extra_from_health(health: Mapping[str, Any]) -> dict[str, Any]:
    sd = health.get("sd") if isinstance(health.get("sd"), Mapping) else {}
    remaining_gib = sd.get("remaining_gib")
    remaining_gb = sd.get("remaining_gb")
    remaining_kib = sd.get("remaining_kib")
    flat_remaining_gib = health.get("sd_space_remaining_gib")
    if flat_remaining_gib is None and health.get("sd_space_remaining_gb_decimal") is None:
        flat_remaining_gib = health.get("sd_space_remaining_gb")
    return {
        "sd_status": sd.get("status") or health.get("sd_status"),
        "sd_present": sd.get("present") if "present" in sd else health.get("sd_present"),
        "sd_full": sd.get("full") if "full" in sd else health.get("sd_full"),
        "sd_remaining_kib": remaining_kib if remaining_kib is not None else health.get("sd_space_remaining_kib"),
        "sd_remaining_gib": remaining_gib if remaining_gib is not None else flat_remaining_gib,
        "sd_remaining_gb": remaining_gb if remaining_gb is not None else health.get("sd_space_remaining_gb_decimal"),
    }


def format_sd_health_cell_rich(health: Mapping[str, Any]) -> str:
    sd = health.get("sd") if isinstance(health.get("sd"), Mapping) else {}
    status = sd.get("status") or health.get("sd_status")
    if status == "missing":
        return "[red]missing[/red]"
    if status == "full":
        return "[bold red]FULL[/bold red]"

    remaining_gib = sd.get("remaining_gib")
    if not isinstance(remaining_gib, (int, float)):
        remaining_gib = health.get("sd_space_remaining_gib")
    if not isinstance(remaining_gib, (int, float)) and health.get("sd_space_remaining_gb_decimal") is None:
        remaining_gib = health.get("sd_space_remaining_gb")
    if isinstance(remaining_gib, (int, float)):
        return f"{float(remaining_gib):.1f} GiB"

    present = sd.get("present") if "present" in sd else health.get("sd_present")
    if present is True:
        return "present"
    if present is False:
        return "[red]missing[/red]"
    return "-"


def format_sd_card_status_rich(sd: SdCardStatus) -> str:
    if sd.status == "missing":
        return "missing"
    if sd.status == "full":
        return "[bold red]FULL[/bold red]"
    if sd.status == "ok":
        if sd.remaining_gib is not None:
            return f"{sd.remaining_gib:.1f} GiB free"
        return "present"
    return "unknown"


def format_sd_from_extra_rich(extra: Mapping[str, Any], *, fallback_remaining_gib: Optional[float] = None) -> str:
    status = extra.get("sd_status")
    if status == "missing":
        return "[red]missing[/red]"
    if status == "full":
        return "[bold red]FULL[/bold red]"

    remaining_gib = extra.get("sd_remaining_gib")
    if not isinstance(remaining_gib, (int, float)):
        remaining_gib = fallback_remaining_gib
    if isinstance(remaining_gib, (int, float)):
        return f"{float(remaining_gib):.1f} GiB"
    return "??"
