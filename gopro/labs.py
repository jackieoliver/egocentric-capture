from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LabsFeature:
    """A GoPro Labs feature that can be enabled via QR code."""
    name: str
    command: str
    description: str
    category: str = "setup"


# Key features for egocentric data pipeline
SETUP_FEATURES: list[LabsFeature] = [
    LabsFeature(
        name="BLE Pairing (Quik)",
        command="!PA",
        description="Start BLE pairing (Quik pairing mode).",
        category="ble",
    ),
    LabsFeature(
        name="BLE Pairing (Remote)",
        command="!PR",
        description="Start BLE remote pairing mode.",
        category="ble",
    ),
    LabsFeature(
        name="BLE Pairing Cancel",
        command="!PS",
        description="Cancel BLE pairing mode.",
        category="ble",
    ),
    LabsFeature(
        name="QRDR - QR Detection During Recording",
        command="*QRDR=1",
        description="Scan and store QR codes while recording. Essential for task markers.",
        category="critical",
    ),
    LabsFeature(
        name="GPS Time Sync",
        command="!MSYNC",
        description="Synchronize camera time to GPS satellites (±1 frame accuracy).",
        category="sync",
    ),
    LabsFeature(
        name="Wake on USB Power",
        command="*WAKE=1",
        description="Automatically wake camera when USB power connected.",
        category="power",
    ),
    LabsFeature(
        name="Boot to Recording Ready",
        command="*BOOT=1",
        description="Boot directly to last used mode, ready to record.",
        category="power",
    ),
    LabsFeature(
        name="Disable Front LCD",
        command="dF0",
        description="Disable front LCD to save power and reduce distractions.",
        category="power",
    ),
    LabsFeature(
        name="Disable All LEDs",
        command="dL0",
        description="Turn off all LED indicators for stealth/discretion.",
        category="power",
    ),
    LabsFeature(
        name="Quiet Beeps",
        command="dB2",
        description="Reduce beep volume to 70% (keeps QR scan confirmation).",
        category="power",
    ),
]

# Camera handle features (for multi-camera setups)
HANDLE_FEATURES: list[LabsFeature] = [
    LabsFeature(
        name=f"Camera Handle {i}",
        command=f"*HNDL={i}",
        description=f"Assign camera ID {i} for multi-camera identification in metadata.",
        category="handle",
    )
    for i in range(1, 9)
]

# Predefined Haptica camera names for multi-camera rigs
HAPTICA_NAMES = {
    "head": "Haptica-Head",
    "wrist_left": "Haptica-WristL",
    "wrist_right": "Haptica-WristR",
    "chest": "Haptica-Chest",
    "shoulder": "Haptica-Shoulder",
}


_COLOR_PROFILE_MAP = {
    "natural": "cN",
    "standard": "cN",
    "natural/standard": "cN",
    "vibrant": "cG",
    "flat": "cF",
    "log": "cF",
}

_WHITE_BALANCE_CODES = {
    2300: "w23",
    2800: "w28",
    3200: "w32",
    4000: "w40",
    4500: "w45",
    5000: "w50",
    5500: "w55",
    6000: "w60",
    6500: "w65",
}

_ISO_STEPS = [100, 200, 400, 800, 1600, 3200, 6400]
_ISO_MIN_PREFIX = "M"
_ISO_MAX_PREFIX = "i"

_SHUTTER_ANGLES = [360, 180, 90, 45, 22, 10, 5, 2]


def build_profile_labs_command(
    profile: Dict[str, Any],
    *,
    profile_name: Optional[str] = None,
    handle_id: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """
    Build a GoPro Labs QR command string for settings not covered by OpenGoPro.
    Returns (command, labels) where labels describe the included settings.
    """
    commands: List[str] = []
    labels: List[str] = []

    owner_name = _format_owner_name(profile_name)
    if owner_name:
        commands.append(f'*OWNR="{owner_name}"')
        labels.append(f"owner:{owner_name}")

    if handle_id:
        commands.append(f"*HNDL={handle_id}")
        labels.append(f"handle:{handle_id}")

    protune_cmds: List[str] = []

    color_cmd = _protune_color(profile.get("color_profile"))
    if color_cmd:
        protune_cmds.append(color_cmd)
        labels.append(f"color:{color_cmd}")

    wb_cmd = _white_balance(profile.get("white_balance_kelvin"))
    if wb_cmd:
        protune_cmds.append(wb_cmd)
        labels.append(f"wb:{wb_cmd}")

    iso_min_cmd = _iso_value(profile.get("iso_min"), prefix=_ISO_MIN_PREFIX)
    if iso_min_cmd:
        protune_cmds.append(iso_min_cmd)
        labels.append(f"iso_min:{iso_min_cmd}")

    iso_max_cmd = _iso_value(profile.get("iso_max"), prefix=_ISO_MAX_PREFIX)
    if iso_max_cmd:
        protune_cmds.append(iso_max_cmd)
        labels.append(f"iso_max:{iso_max_cmd}")

    sharp_cmd = _sharpness(profile.get("sharpness"))
    if sharp_cmd:
        protune_cmds.append(sharp_cmd)
        labels.append(f"sharpness:{sharp_cmd}")

    shutter_cmd = _shutter_lock(profile.get("shutter_target"), profile.get("fps"))
    if shutter_cmd:
        protune_cmds.append(shutter_cmd)
        labels.append(f"shutter:{shutter_cmd}")

    if protune_cmds:
        protune_cmds.insert(0, "t")
        commands.append("".join(protune_cmds))

    denoise_cmd = _noise_reduction(profile.get("denoise"))
    if denoise_cmd:
        commands.append(denoise_cmd)
        labels.append(f"denoise:{denoise_cmd}")

    return "".join(commands), labels


def _format_owner_name(profile_name: Optional[str]) -> Optional[str]:
    if not profile_name:
        return None
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", profile_name).strip("-_")
    if not safe:
        return None
    return f"HAPTICA-{safe.replace('_', '-').upper()}"


def _protune_color(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return _COLOR_PROFILE_MAP.get(text)


def _white_balance(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"auto", "aw", "awb"}:
        return "wA"
    if text in {"native", "raw"}:
        return "wN"

    kelvin = _coerce_int(value)
    if not kelvin:
        return None
    nearest = _nearest_value(kelvin, list(_WHITE_BALANCE_CODES.keys()))
    return _WHITE_BALANCE_CODES.get(nearest)


def _iso_value(value: Any, *, prefix: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"auto"}:
        return None
    iso = _coerce_int(value)
    if not iso:
        return None
    nearest = _nearest_value(iso, _ISO_STEPS)
    return f"{prefix}{int(nearest / 100)}"


def _sharpness(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"low", "l"}:
        return "sL"
    if text in {"medium", "med", "m"}:
        return "sM"
    if text in {"high", "h"}:
        return "sH"
    return None


def _noise_reduction(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    level: Optional[int] = None
    if text.isdigit():
        level = int(text)
    elif text in {"off", "none", "zero", "min"}:
        level = 1
    elif text in {"low", "l"}:
        level = 25
    elif text in {"medium", "med", "m"}:
        level = 50
    elif text in {"high", "h", "max"}:
        level = 100

    if level is None:
        return None
    level = max(1, min(level, 100))
    return f"*NR01={level}"


def _shutter_lock(shutter_target: Any, fps_value: Any) -> Optional[str]:
    if shutter_target is None or fps_value is None:
        return None

    target_text = None
    if isinstance(shutter_target, dict):
        target_text = shutter_target.get("preferred") or shutter_target.get("high_light")
    else:
        target_text = shutter_target

    if not target_text:
        return None

    match = re.search(r"1\s*/\s*(\d+)", str(target_text))
    if not match:
        return None

    shutter_den = _coerce_int(match.group(1))
    fps = _coerce_int(fps_value)
    if not shutter_den or not fps:
        return None

    angle = 360.0 * float(fps) / float(shutter_den)
    nearest = _nearest_value(angle, _SHUTTER_ANGLES)
    return f"S{int(nearest)}"


def _nearest_value(value: float, options: List[int]) -> int:
    return min(options, key=lambda opt: abs(opt - value))


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None
