from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import os
from pathlib import Path
import ssl
import time
from typing import Any, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from .config import load_cohn_config
from .media import LastCaptured, parse_last_captured_response


def _read_timeout(env_key: str, default: float) -> float:
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


DEFAULT_HTTP_TIMEOUT = _read_timeout("GOPRO_HTTP_TIMEOUT", 10.0)
DEFAULT_CONTROL_TIMEOUT = _read_timeout("GOPRO_CONTROL_TIMEOUT", 3.0)
DEFAULT_MEDIA_TIMEOUT = _read_timeout("GOPRO_MEDIA_TIMEOUT", 5.0)
CLIENT_CACHE_TTL_SECONDS = _read_timeout("GOPRO_HTTP_CLIENT_CACHE_TTL", 120.0)
MEDIA_LIST_TTL_SECONDS = _read_timeout("GOPRO_MEDIA_LIST_TTL", 2.0)


@dataclass
class _ClientCacheEntry:
    client: httpx.Client
    last_used: float


_CLIENT_CACHE: dict[tuple, _ClientCacheEntry] = {}


@dataclass
class _MediaListCacheEntry:
    data: dict[str, Any]
    fetched_at: float


_MEDIA_LIST_CACHE: dict[tuple, _MediaListCacheEntry] = {}


def _http_cache_enabled() -> bool:
    return os.environ.get("GOPRO_HTTP_CLIENT_CACHE", "1").strip().lower() not in ("0", "false", "no")


def _client_cache_key(
    camera: "CameraInventoryRecord",
    transport: str,
    timeout: float,
    auth: Optional[httpx.Auth],
    verify_key: str,
) -> tuple:
    auth_key: Optional[tuple[str, str]] = None
    if isinstance(auth, tuple) and len(auth) == 2:
        auth_key = (str(auth[0]), str(auth[1]))
    return (camera.serial_number, transport, float(timeout), auth_key, verify_key)


def _prune_client_cache(now: float) -> None:
    if CLIENT_CACHE_TTL_SECONDS <= 0:
        return
    for key, entry in list(_CLIENT_CACHE.items()):
        if now - entry.last_used > CLIENT_CACHE_TTL_SECONDS:
            try:
                entry.client.close()
            except Exception:
                pass
            _CLIENT_CACHE.pop(key, None)


def _mark_client_managed(client: httpx.Client, managed: bool) -> httpx.Client:
    setattr(client, "_haptica_managed", managed)
    return client


def close_http_client(client: httpx.Client) -> None:
    if getattr(client, "_haptica_managed", False):
        return
    client.close()


def _get_or_create_client(
    cache_key: tuple,
    *,
    timeout: float,
    verify: httpx.VerifyTypes,
    auth: Optional[httpx.Auth],
) -> httpx.Client:
    if not _http_cache_enabled():
        return _mark_client_managed(_create_ipv4_client(timeout=timeout, verify=verify, auth=auth), False)
    now = time.monotonic()
    _prune_client_cache(now)
    entry = _CLIENT_CACHE.get(cache_key)
    if entry:
        entry.last_used = now
        return _mark_client_managed(entry.client, True)
    client = _create_ipv4_client(timeout=timeout, verify=verify, auth=auth)
    _CLIENT_CACHE[cache_key] = _ClientCacheEntry(client=client, last_used=now)
    return _mark_client_managed(client, True)


def _create_ipv4_client(
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    verify: httpx.VerifyTypes = True,
    auth: Optional[httpx.Auth] = None,
) -> httpx.Client:
    """Create an HTTP client that forces IPv4 connections."""
    # Force IPv4 by binding to 0.0.0.0 (prevents IPv6 attempts)
    transport = httpx.HTTPTransport(local_address="0.0.0.0", verify=verify)
    return httpx.Client(timeout=timeout, transport=transport, auth=auth)


def _create_cohn_ssl_context(certificate: str) -> Optional[ssl.SSLContext]:
    if not certificate.strip():
        return None
    context = ssl.create_default_context()
    context.check_hostname = False
    try:
        context.load_verify_locations(cadata=certificate)
    except Exception:
        return None
    return context


def _get_cohn_credentials(camera: "CameraInventoryRecord") -> Optional[object]:
    credentials = load_cohn_config()
    if not credentials:
        return None
    candidate_ids = [
        camera.provisioning.cohn_credential_id,
        camera.serial_number,
        camera.short_id,
    ]
    for candidate in candidate_ids:
        if candidate and candidate in credentials:
            return credentials[candidate]
    return None

if TYPE_CHECKING:
    from .state import CameraInventoryRecord


GOPRO_USB_PORT = 8080
GOPRO_COHN_PORT = 443


def _cohn_base_url(camera: "CameraInventoryRecord") -> Optional[str]:
    if not camera.provisioning.cohn_enabled:
        return None
    if not camera.wifi.ip:
        return None
    port = camera.wifi.port or GOPRO_COHN_PORT
    return f"https://{camera.wifi.ip}:{port}"


def _create_http_client_for_base_url(
    camera: "CameraInventoryRecord",
    base_url: str,
    *,
    timeout: float,
    transport: str,
) -> httpx.Client:
    verify: httpx.VerifyTypes = transport != "cohn"
    auth: Optional[httpx.Auth] = None
    verify_key = "default"
    if transport == "cohn":
        creds = _get_cohn_credentials(camera)
        certificate = ""
        if creds:
            if getattr(creds, "username", None) and getattr(creds, "password", None):
                auth = (creds.username, creds.password)
            certificate = str(getattr(creds, "certificate", "") or "")
            ssl_context = _create_cohn_ssl_context(certificate)
            verify = ssl_context if ssl_context else False
        else:
            verify = False
        verify_key = certificate.strip() or "insecure"
    cache_key = _client_cache_key(camera, transport, timeout, auth, verify_key)
    return _get_or_create_client(cache_key, timeout=timeout, verify=verify, auth=auth)


def create_http_client_for_control(
    camera: "CameraInventoryRecord",
    timeout: float = DEFAULT_CONTROL_TIMEOUT,
    *,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> Tuple[Optional[httpx.Client], Optional[str], Optional[str]]:
    """
    Create an HTTP client for "control plane" calls, even if wifi.available is stale.

    For COHN, we treat `cohn_enabled + wifi.ip` as sufficient to attempt HTTPS.
    USB remains opt-in.
    """
    if prefer_usb and allow_usb and camera.usb.ip:
        base_url = f"http://{camera.usb.ip}:{GOPRO_USB_PORT}"
        return _create_http_client_for_base_url(camera, base_url, timeout=timeout, transport="usb"), base_url, "usb"

    cohn_url = _cohn_base_url(camera)
    if cohn_url:
        return _create_http_client_for_base_url(camera, cohn_url, timeout=timeout, transport="cohn"), cohn_url, "cohn"

    if allow_usb and camera.usb.ip:
        base_url = f"http://{camera.usb.ip}:{GOPRO_USB_PORT}"
        return _create_http_client_for_base_url(camera, base_url, timeout=timeout, transport="usb"), base_url, "usb"

    return None, None, None


# =============================================================================
# Transport Selection (COHN primary, USB opt-in)
# =============================================================================

def get_http_base_url(
    camera: "CameraInventoryRecord",
    *,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the best available HTTP base URL for a camera.

    Tries COHN first by default. USB is opt-in.

    Returns:
        (base_url, transport) - e.g. ("http://192.0.2.10:8080", "usb")
        (None, None) if no HTTP path available
    """
    # Try USB first (if allowed)
    if prefer_usb and allow_usb and camera.usb.available and camera.usb.ip:
        return f"http://{camera.usb.ip}:{GOPRO_USB_PORT}", "usb"

    # Try COHN (WiFi) - requires COHN to be enabled and have IP
    if camera.provisioning.cohn_enabled and camera.wifi.available and camera.wifi.ip:
        port = camera.wifi.port or GOPRO_COHN_PORT
        return f"https://{camera.wifi.ip}:{port}", "cohn"

    # Fallback to USB if allowed
    if not prefer_usb and allow_usb and camera.usb.available and camera.usb.ip:
        return f"http://{camera.usb.ip}:{GOPRO_USB_PORT}", "usb"

    return None, None


def create_http_client(
    camera: "CameraInventoryRecord",
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    *,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> Tuple[Optional[httpx.Client], Optional[str], Optional[str]]:
    """
    Create an HTTP client configured for the best available transport.

    Returns:
        (client, base_url, transport) or (None, None, None) if unavailable

    Note: Caller is responsible for closing the client.
    """
    base_url, transport = get_http_base_url(camera, prefer_usb=prefer_usb, allow_usb=allow_usb)
    if not base_url:
        return None, None, None
    client = _create_http_client_for_base_url(camera, base_url, timeout=timeout, transport=transport)
    return client, base_url, transport


def query_camera(
    camera: "CameraInventoryRecord",
    *,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> Tuple[Optional["GoProDevice"], Optional[str]]:
    """
    Query camera state using best available transport.

    Returns:
        (device, transport) or (None, None) if unreachable
    """
    client, base_url, transport = create_http_client(
        camera,
        prefer_usb=prefer_usb,
        allow_usb=allow_usb,
    )
    if not client:
        return None, None

    try:
        info_resp = client.get(f"{base_url}/gopro/camera/info")
        info_resp.raise_for_status()
        info = info_resp.json()

        state_resp = client.get(f"{base_url}/gopro/camera/state")
        state_resp.raise_for_status()
        state = state_resp.json()["status"]

        battery = _safe_int(state.get("70"))
        sd_present_raw = _safe_int(state.get("33"))
        photos_remaining = _safe_int(state.get("34"))
        video_minutes_remaining = _safe_int(state.get("35"))
        sd_space_remaining_kb = _safe_int(state.get("54"))

        device = GoProDevice(
            serial_number=info.get("serial_number", ""),
            model_name=info.get("model_name", ""),
            model_number=info.get("model_number", ""),
            firmware_version=info.get("firmware_version", ""),
            camera_name=state.get("30", info.get("ap_ssid", "")),
            ap_ssid=info.get("ap_ssid", ""),
            ap_mac_addr=info.get("ap_mac_addr", ""),
            ip_address=base_url,  # Store the URL we used
            battery_percent=battery or 0,
            sd_card_present=(sd_present_raw or 0) == 0,
            photos_remaining=photos_remaining or 0,
            video_minutes_remaining=video_minutes_remaining or 0,
            sd_space_remaining_kb=sd_space_remaining_kb or 0,
            gps_lock=bool(state.get("68", 0)),
            is_recording=bool(state.get("10", 0)),
            discovered_at=datetime.now().isoformat(),
        )
        return device, transport
    except Exception:
        return None, None
    finally:
        close_http_client(client)


def shutter_control(
    camera: "CameraInventoryRecord",
    action: str,
    *,
    timeout: float = DEFAULT_CONTROL_TIMEOUT,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> Tuple[bool, str, Optional[str]]:
    """
    Start or stop recording.

    Args:
        camera: Camera record
        action: "start" or "stop"

    Returns:
        (success, message, transport)
    """
    client, base_url, transport = create_http_client_for_control(
        camera,
        timeout=timeout,
        prefer_usb=prefer_usb,
        allow_usb=allow_usb,
    )
    if not client:
        return False, "No HTTP connection available", None

    try:
        # For USB transport, need to open control before recording
        if transport == "usb" and action == "start":
            # Open USB control
            client.get(f"{base_url}/gopro/camera/control/wired_usb?p=1")
            # Set video preset (id=1000 = Standard video mode)
            client.get(f"{base_url}/gopro/camera/presets/set_group?id=1000")

        resp = client.get(f"{base_url}/gopro/camera/shutter/{action}")
        if resp.status_code == 200:
            # For USB transport, close control after stopping
            if transport == "usb" and action == "stop":
                client.get(f"{base_url}/gopro/camera/control/wired_usb?p=0")
            return True, "OK", transport
        else:
            try:
                error = resp.json().get("error", resp.text)
            except Exception:
                error = resp.text
            return False, str(error), transport
    except Exception as e:
        return False, str(e), transport
    finally:
        close_http_client(client)


def apply_setting_to_camera(
    camera: "CameraInventoryRecord",
    setting_id: int,
    option: int,
    *,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Apply a setting using best available transport.

    Returns:
        (success, transport)
    """
    client, base_url, transport = create_http_client(
        camera,
        prefer_usb=prefer_usb,
        allow_usb=allow_usb,
    )
    if not client:
        return False, None

    try:
        resp = client.get(
            f"{base_url}/gopro/camera/setting",
            params={"setting": setting_id, "option": option},
        )
        return resp.status_code == 200, transport
    except Exception:
        return False, transport
    finally:
        close_http_client(client)


def locate_camera(
    camera: "CameraInventoryRecord",
    on: bool,
    *,
    beep_volume_option: Optional[int] = 100,
    timeout: float = 2.0,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> Tuple[bool, str, Optional[str]]:
    """
    Trigger the legacy gpControl "locate" beep.

    Notes:
    - This is not an OpenGoPro v2 command; it uses the older `/gp/gpControl/...` endpoint.
    - Works over COHN if the endpoint is exposed on the camera's COHN interface.
    - Falls back to `http://<ip>/gp/gpControl/...` when the COHN base URL doesn't serve it.

    Returns:
        (success, message, transport)
    """
    client, base_url, transport = create_http_client_for_control(
        camera,
        timeout=timeout,
        prefer_usb=prefer_usb,
        allow_usb=allow_usb,
    )
    if not client or not base_url:
        return False, "No HTTP connection available", None

    try:
        if on and beep_volume_option is not None:
            try:
                client.get(
                    f"{base_url}/gopro/camera/setting",
                    params={"setting": 216, "option": int(beep_volume_option)},
                )
            except Exception:
                pass

        urls: list[str] = [f"{base_url}/gp/gpControl/command/system/locate"]
        parsed = urlparse(base_url)
        if parsed.hostname:
            urls.append(f"http://{parsed.hostname}/gp/gpControl/command/system/locate")

        last_error: Optional[str] = None
        for url in urls:
            try:
                resp = client.get(url, params={"p": 1 if on else 0})
                if resp.status_code == 200:
                    return True, "OK", transport
                last_error = f"http_{resp.status_code}"
            except Exception as exc:
                last_error = str(exc)
                continue
        return False, last_error or "failed", transport
    finally:
        close_http_client(client)


@dataclass
class GoProDevice:
    serial_number: str
    model_name: str
    model_number: str
    firmware_version: str
    camera_name: str
    ap_ssid: str
    ap_mac_addr: str
    ip_address: str
    battery_percent: int
    sd_card_present: bool
    photos_remaining: int
    video_minutes_remaining: int
    sd_space_remaining_kb: int
    gps_lock: bool
    is_recording: bool
    discovered_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def get_last_captured(
    camera: "CameraInventoryRecord",
    *,
    timeout: float = DEFAULT_MEDIA_TIMEOUT,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> LastCaptured:
    """
    Get the last captured media file from a camera.

    Returns:
        `LastCaptured` (folder/file/gumi if available).
    """
    client, base_url, transport = create_http_client(
        camera,
        timeout=timeout,
        prefer_usb=prefer_usb,
        allow_usb=allow_usb,
    )
    if not client:
        return LastCaptured()

    try:
        resp = client.get(f"{base_url}/gopro/media/last_captured")
        if resp.status_code == 200:
            return parse_last_captured_response(resp.json() or {})
        return LastCaptured()
    except Exception:
        return LastCaptured()
    finally:
        close_http_client(client)


def _media_list_cache_key(camera: "CameraInventoryRecord", transport: Optional[str]) -> tuple:
    return (camera.serial_number, transport or "unknown")


def _fetch_media_list(
    client: httpx.Client,
    base_url: str,
    cache_key: tuple,
) -> Optional[dict[str, Any]]:
    resp = client.get(f"{base_url}/gopro/media/list")
    if resp.status_code != 200:
        return None
    data = resp.json()
    _MEDIA_LIST_CACHE[cache_key] = _MediaListCacheEntry(data=data, fetched_at=time.monotonic())
    return data


def _get_media_list(
    client: httpx.Client,
    base_url: str,
    cache_key: tuple,
    *,
    force_refresh: bool = False,
) -> Optional[dict[str, Any]]:
    if not force_refresh:
        entry = _MEDIA_LIST_CACHE.get(cache_key)
        if entry:
            age = time.monotonic() - entry.fetched_at
            if MEDIA_LIST_TTL_SECONDS <= 0 or age <= MEDIA_LIST_TTL_SECONDS:
                return entry.data
    return _fetch_media_list(client, base_url, cache_key)


def _find_media_info(data: dict[str, Any], folder: str, file: str) -> Optional[dict[str, Any]]:
    for media_folder in data.get("media", []):
        if media_folder.get("d") == folder:
            for f in media_folder.get("fs", []):
                if f.get("n") == file:
                    info: dict[str, Any] = {
                        "s": int(f.get("s", 0)),
                        "dur": f.get("dur"),
                    }
                    for key in ("mod", "cre", "cts", "ts"):
                        if key in f:
                            info[key] = f.get(key)
                    return info
    return None


def get_media_info(
    camera: "CameraInventoryRecord",
    folder: str,
    file: str,
    *,
    timeout: float = DEFAULT_MEDIA_TIMEOUT,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> Optional[dict]:
    """
    Get detailed metadata for a media file.

    Returns:
        Dict with 's' (size in bytes) and optionally 'dur' (duration), or None
    """
    client, base_url, transport = create_http_client(
        camera,
        timeout=timeout,
        prefer_usb=prefer_usb,
        allow_usb=allow_usb,
    )
    if not client:
        return None

    try:
        # Use media list to find file info (more reliable than /gopro/media/info)
        cache_key = _media_list_cache_key(camera, transport)
        data = _get_media_list(client, base_url, cache_key, force_refresh=False)
        if data:
            info = _find_media_info(data, folder, file)
            if info:
                return info
        data = _get_media_list(client, base_url, cache_key, force_refresh=True)
        if data:
            return _find_media_info(data, folder, file)
        return None
    except Exception:
        return None
    finally:
        close_http_client(client)


def get_camera_state(
    camera: "CameraInventoryRecord",
    *,
    timeout: float = DEFAULT_CONTROL_TIMEOUT,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Fetch camera state status map.

    Returns:
        (status_dict, transport) or (None, None) if unavailable.
    """
    client, base_url, transport = create_http_client_for_control(
        camera,
        timeout=timeout,
        prefer_usb=prefer_usb,
        allow_usb=allow_usb,
    )
    if not client:
        return None, None

    try:
        resp = client.get(f"{base_url}/gopro/camera/state")
        if resp.status_code != 200:
            return None, transport
        data = resp.json()
        return data.get("status", {}), transport
    except Exception:
        return None, transport
    finally:
        close_http_client(client)


def start_stream(
    camera: "CameraInventoryRecord",
    port: int,
    *,
    timeout: float = DEFAULT_CONTROL_TIMEOUT,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> Tuple[bool, str, Optional[str]]:
    """
    Start the UDP preview stream over COHN.

    Returns:
        (success, message, transport)
    """
    client, base_url, transport = create_http_client_for_control(
        camera,
        timeout=timeout,
        prefer_usb=prefer_usb,
        allow_usb=allow_usb,
    )
    if not client or not base_url:
        return False, "no_http", None

    try:
        resp = client.get(f"{base_url}/gopro/camera/stream/start", params={"port": int(port)})
        if resp.status_code == 200:
            return True, "OK", transport
        return False, f"http_{resp.status_code}", transport
    except Exception as exc:
        return False, str(exc), transport
    finally:
        close_http_client(client)


def stop_stream(
    camera: "CameraInventoryRecord",
    *,
    timeout: float = DEFAULT_CONTROL_TIMEOUT,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> Tuple[bool, str, Optional[str]]:
    """
    Stop the UDP preview stream.

    Returns:
        (success, message, transport)
    """
    client, base_url, transport = create_http_client_for_control(
        camera,
        timeout=timeout,
        prefer_usb=prefer_usb,
        allow_usb=allow_usb,
    )
    if not client or not base_url:
        return False, "no_http", None

    try:
        resp = client.get(f"{base_url}/gopro/camera/stream/stop")
        if resp.status_code == 200:
            return True, "OK", transport
        return False, f"http_{resp.status_code}", transport
    except Exception as exc:
        return False, str(exc), transport
    finally:
        close_http_client(client)


def download_media(
    camera: "CameraInventoryRecord",
    folder: str,
    file: str,
    dest_path: Path,
    *,
    prefer_usb: bool = False,
    allow_usb: bool = False,
    timeout: float = 600.0,
) -> tuple[bool, Optional[int], Optional[str], Optional[str]]:
    """
    Download a media file off the camera to `dest_path`.

    This is the device-specific "mechanism" used by the device-agnostic ingest layer.

    Returns:
        (ok, size_bytes, transport, error_message)
    """
    dest_path = Path(dest_path)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".partial")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    expected_size: Optional[int] = None
    try:
        # Best-effort skip if already present and non-empty.
        if dest_path.exists() and dest_path.is_file():
            existing_size = dest_path.stat().st_size
            if existing_size > 0:
                return True, existing_size, "local", None
    except Exception:
        pass

    client, base_url, transport = create_http_client_for_control(
        camera,
        timeout=timeout,
        prefer_usb=prefer_usb,
        allow_usb=allow_usb,
    )
    if not client or not base_url:
        return False, None, None, "http_unavailable"

    url = f"{base_url}/videos/DCIM/{folder}/{file}"
    written = 0
    try:
        with client.stream("GET", url, follow_redirects=True) as resp:
            if resp.status_code != 200:
                return False, None, transport, f"http_status:{resp.status_code}"
            content_length = resp.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                expected_size = int(content_length)

            with open(tmp_path, "wb") as out:
                for chunk in resp.iter_bytes():
                    if not chunk:
                        continue
                    out.write(chunk)
                    written += len(chunk)

        os.replace(tmp_path, dest_path)
        size_bytes = dest_path.stat().st_size
        if expected_size is not None and size_bytes != expected_size:
            try:
                dest_path.unlink(missing_ok=True)
            except Exception:
                pass
            return False, size_bytes, transport, "size_mismatch"
        return True, size_bytes, transport, None
    except Exception as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False, written or None, transport, str(exc)
    finally:
        close_http_client(client)


def download_telemetry(
    camera: "CameraInventoryRecord",
    folder: str,
    file: str,
    dest_path: Path,
    *,
    prefer_usb: bool = False,
    allow_usb: bool = False,
    timeout: float = 600.0,
) -> tuple[bool, Optional[int], Optional[str], Optional[str]]:
    """
    Download telemetry (GPMF) data for a media file.

    Returns:
        (ok, size_bytes, transport, error_message)
    """
    dest_path = Path(dest_path)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".partial")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if dest_path.exists() and dest_path.is_file():
            existing_size = dest_path.stat().st_size
            if existing_size > 0:
                return True, existing_size, "local", None
    except Exception:
        pass

    client, base_url, transport = create_http_client_for_control(
        camera,
        timeout=timeout,
        prefer_usb=prefer_usb,
        allow_usb=allow_usb,
    )
    if not client or not base_url:
        return False, None, None, "http_unavailable"

    media_path = f"{folder}/{file}"
    endpoints = ["gopro/media/gpmf", "gopro/media/telemetry"]
    try:
        for idx, endpoint in enumerate(endpoints):
            expected_size: Optional[int] = None
            written = 0
            url = f"{base_url}/{endpoint}"
            try:
                with client.stream("GET", url, params={"path": media_path}, follow_redirects=True) as resp:
                    if resp.status_code != 200:
                        if resp.status_code in {400, 404} and idx + 1 < len(endpoints):
                            continue
                        return False, None, transport, f"http_status:{resp.status_code}"
                    content_length = resp.headers.get("Content-Length")
                    if content_length and content_length.isdigit():
                        expected_size = int(content_length)

                    with open(tmp_path, "wb") as out:
                        for chunk in resp.iter_bytes():
                            if not chunk:
                                continue
                            out.write(chunk)
                            written += len(chunk)

                os.replace(tmp_path, dest_path)
                size_bytes = dest_path.stat().st_size
                if expected_size is not None and size_bytes != expected_size:
                    try:
                        dest_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return False, size_bytes, transport, "size_mismatch"
                return True, size_bytes, transport, None
            except Exception as exc:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                if idx + 1 < len(endpoints):
                    continue
                return False, written or None, transport, str(exc)
        return False, None, transport, "telemetry_unavailable"
    finally:
        close_http_client(client)


SETTING_VIDEO_RESOLUTION = 2
SETTING_VIDEO_FPS = 3
SETTING_VIDEO_ASPECT = 108
SETTING_VIDEO_LENS = 121
SETTING_HYPERSMOOTH = 135
SETTING_BIT_DEPTH = 183
SETTING_VIDEO_FRAMING = 232

RESOLUTION_OPTIONS = {
    ("4K", "16:9"): 1,
    ("4K", "4:3"): 112,
    ("2.7K", "16:9"): 4,
    ("2.7K", "4:3"): 111,
}

FRAMING_OPTIONS = {
    "4:3": 0,
    "16:9": 1,
    "8:7": 3,
    "9:16": 4,
}

FPS_OPTIONS = {
    120: 1,
}

LENS_OPTIONS = {
    "Wide": 0,
    "Ultra Wide": 13,
}

BIT_DEPTH_OPTIONS = {
    8: 0,
    10: 2,
}

HYPERSMOOTH_OPTIONS = {
    "off": 0,
    "on": 1,
}


def _apply_setting_url(client: httpx.Client, base_url: str, setting_id: int, option: int) -> bool:
    try:
        resp = client.get(
            f"{base_url}/gopro/camera/setting",
            params={"setting": setting_id, "option": option},
        )
        return resp.status_code == 200
    except Exception:
        return False


def apply_profile_settings_camera(
    camera: "CameraInventoryRecord",
    profile: dict,
    *,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> tuple[list[str], list[str], Optional[str]]:
    client, base_url, transport = create_http_client(
        camera,
        prefer_usb=prefer_usb,
        allow_usb=allow_usb,
    )
    if not client:
        return [], [], None

    applied: list[str] = []
    skipped: list[str] = []

    resolution = profile.get("resolution")
    framing = profile.get("framing")
    fps = profile.get("fps")
    bit_depth = profile.get("bit_depth")
    stabilization = str(profile.get("stabilization", "")).lower()

    try:
        if resolution and framing:
            option = RESOLUTION_OPTIONS.get((str(resolution), str(framing)))
            if option is not None and _apply_setting_url(client, base_url, SETTING_VIDEO_RESOLUTION, option):
                applied.append("video_resolution")
            else:
                skipped.append("video_resolution")
        else:
            skipped.append("video_resolution")

        if framing:
            framing_option = FRAMING_OPTIONS.get(str(framing))
            if framing_option is not None and _apply_setting_url(
                client, base_url, SETTING_VIDEO_FRAMING, framing_option
            ):
                applied.append("video_framing")
            else:
                skipped.append("video_framing")

        if isinstance(fps, int):
            fps_option = FPS_OPTIONS.get(fps)
            if fps_option is not None and _apply_setting_url(client, base_url, SETTING_VIDEO_FPS, fps_option):
                applied.append("fps")
            else:
                skipped.append("fps")
        else:
            skipped.append("fps")

        lens_setting = str(profile.get("lens_setting", "Wide"))
        lens_option = LENS_OPTIONS.get(lens_setting)
        if lens_option is not None and _apply_setting_url(client, base_url, SETTING_VIDEO_LENS, lens_option):
            applied.append("video_lens")
        else:
            skipped.append("video_lens")

        if isinstance(bit_depth, int):
            bit_depth_option = BIT_DEPTH_OPTIONS.get(bit_depth)
            if bit_depth_option is not None and _apply_setting_url(
                client, base_url, SETTING_BIT_DEPTH, bit_depth_option
            ):
                applied.append("bit_depth")
            else:
                skipped.append("bit_depth")
        else:
            skipped.append("bit_depth")

        if stabilization:
            hs_option = HYPERSMOOTH_OPTIONS.get(stabilization)
            if hs_option is not None and _apply_setting_url(
                client, base_url, SETTING_HYPERSMOOTH, hs_option
            ):
                applied.append("hypersmooth")
            else:
                skipped.append("hypersmooth")
    except Exception:
        return [], [], transport
    finally:
        close_http_client(client)

    unsupported = [
        "color_profile",
        "white_balance_kelvin",
        "iso_min",
        "iso_max",
        "sharpness",
        "denoise",
        "shutter_target",
    ]
    skipped.extend([item for item in unsupported if item not in skipped])

    return applied, skipped, transport


# Reverse mappings for validation
RESOLUTION_REVERSE = {v: k for k, v in RESOLUTION_OPTIONS.items()}  # option -> (res, framing)
FRAMING_REVERSE = {v: k for k, v in FRAMING_OPTIONS.items()}  # option -> framing
FPS_REVERSE = {v: k for k, v in FPS_OPTIONS.items()}  # option -> fps
LENS_REVERSE = {v: k for k, v in LENS_OPTIONS.items()}  # option -> lens
BIT_DEPTH_REVERSE = {v: k for k, v in BIT_DEPTH_OPTIONS.items()}  # option -> bit_depth
HYPERSMOOTH_REVERSE = {v: k for k, v in HYPERSMOOTH_OPTIONS.items()}  # option -> stabilization


@dataclass
class SettingValidation:
    """Result of validating a single setting."""
    name: str
    expected: Any
    actual: Any
    matches: bool


def validate_profile_settings(
    camera: "CameraInventoryRecord",
    profile: dict,
    *,
    prefer_usb: bool = False,
    allow_usb: bool = False,
) -> Tuple[List[SettingValidation], Optional[str]]:
    """
    Validate that camera settings match the expected profile.

    Returns:
        (validations, transport) - List of validation results and transport used
    """
    client, base_url, transport = create_http_client(
        camera,
        prefer_usb=prefer_usb,
        allow_usb=allow_usb,
    )
    if not client:
        return [], None

    validations = []

    try:
        resp = client.get(f"{base_url}/gopro/camera/state")
        resp.raise_for_status()
        state = resp.json()
        settings = state.get("settings", {})

        # Resolution
        expected_res = profile.get("resolution")
        expected_framing = str(profile.get("framing", ""))
        if expected_res:
            current_res_option = settings.get(str(SETTING_VIDEO_RESOLUTION))
            expected_option = RESOLUTION_OPTIONS.get((str(expected_res), expected_framing))
            actual_res = RESOLUTION_REVERSE.get(current_res_option, f"unknown({current_res_option})")
            matches = current_res_option == expected_option
            validations.append(SettingValidation(
                name="resolution",
                expected=f"{expected_res} {expected_framing}",
                actual=f"{actual_res[0]} {actual_res[1]}" if isinstance(actual_res, tuple) else str(actual_res),
                matches=matches,
            ))

        # FPS
        expected_fps = profile.get("fps")
        if expected_fps:
            current_fps_option = settings.get(str(SETTING_VIDEO_FPS))
            expected_fps_option = FPS_OPTIONS.get(expected_fps)
            actual_fps = FPS_REVERSE.get(current_fps_option, f"unknown({current_fps_option})")
            matches = current_fps_option == expected_fps_option
            validations.append(SettingValidation(
                name="fps",
                expected=str(expected_fps),
                actual=str(actual_fps),
                matches=matches,
            ))

        # Lens
        expected_lens = profile.get("lens_setting")
        if expected_lens:
            current_lens_option = settings.get(str(SETTING_VIDEO_LENS))
            expected_lens_option = LENS_OPTIONS.get(expected_lens)
            actual_lens = LENS_REVERSE.get(current_lens_option, f"unknown({current_lens_option})")
            matches = current_lens_option == expected_lens_option
            validations.append(SettingValidation(
                name="lens",
                expected=str(expected_lens),
                actual=str(actual_lens),
                matches=matches,
            ))

        # Bit depth
        expected_bit_depth = profile.get("bit_depth")
        if expected_bit_depth:
            current_bd_option = settings.get(str(SETTING_BIT_DEPTH))
            expected_bd_option = BIT_DEPTH_OPTIONS.get(expected_bit_depth)
            actual_bd = BIT_DEPTH_REVERSE.get(current_bd_option, f"unknown({current_bd_option})")
            matches = current_bd_option == expected_bd_option
            validations.append(SettingValidation(
                name="bit_depth",
                expected=str(expected_bit_depth),
                actual=str(actual_bd),
                matches=matches,
            ))

        # Stabilization
        expected_stab = str(profile.get("stabilization", "")).lower()
        if expected_stab:
            current_hs_option = settings.get(str(SETTING_HYPERSMOOTH))
            expected_hs_option = HYPERSMOOTH_OPTIONS.get(expected_stab)
            actual_stab = HYPERSMOOTH_REVERSE.get(current_hs_option, f"unknown({current_hs_option})")
            matches = current_hs_option == expected_hs_option
            validations.append(SettingValidation(
                name="stabilization",
                expected=expected_stab,
                actual=str(actual_stab),
                matches=matches,
            ))

        return validations, transport

    except Exception:
        return [], transport
    finally:
        close_http_client(client)
