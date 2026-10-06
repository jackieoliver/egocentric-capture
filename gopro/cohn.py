from __future__ import annotations

from typing import Optional

from .ble_proto import cohn_pb2
from .config import COHN_DEFAULT_USERNAME, CohnCredentials, upsert_cohn_credentials
from .state import CameraInventoryRecord


def _extract_status_fields(status: object) -> tuple[
    Optional[bool],
    Optional[object],
    Optional[object],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
]:
    if isinstance(status, dict):
        enabled = status.get("enabled")
        status_value = status.get("status")
        state_value = status.get("state")
        username = status.get("username")
        password = status.get("password")
        ipaddress = status.get("ipaddress")
        ssid = status.get("ssid")
    else:
        enabled = getattr(status, "enabled", None)
        status_value = getattr(status, "status", None)
        state_value = getattr(status, "state", None)
        username = getattr(status, "username", None)
        password = getattr(status, "password", None)
        ipaddress = getattr(status, "ipaddress", None)
        ssid = getattr(status, "ssid", None)
    return enabled, status_value, state_value, username, password, ipaddress, ssid


def is_cohn_ready(status: Optional[object]) -> bool:
    if status is None:
        return False

    enabled, status_value, state_value, _, _, ipaddress, _ = _extract_status_fields(status)
    if not ipaddress:
        return False

    if isinstance(enabled, bool) and enabled:
        return True

    if isinstance(status_value, str) and "PROVISIONED" in status_value:
        return True
    if isinstance(status_value, int) and status_value == 1:
        return True

    if state_value is not None:
        try:
            if int(state_value) == int(cohn_pb2.EnumCOHNNetworkState.COHN_STATE_NetworkConnected):
                return True
        except (TypeError, ValueError):
            pass
        if isinstance(state_value, str) and "NetworkConnected" in state_value:
            return True

    return False


def store_cohn_credentials(
    camera: CameraInventoryRecord,
    status: Optional[object],
    certificate: Optional[str],
) -> None:
    if status is None:
        return

    _, _, _, username, password, ipaddress, ssid = _extract_status_fields(status)

    if ipaddress:
        camera.wifi.available = True
        camera.wifi.ip = ipaddress
    if ssid:
        camera.provisioning.wifi_ssid = ssid

    if password:
        credential_id = camera.serial_number
        creds = CohnCredentials(
            username=username or COHN_DEFAULT_USERNAME,
            password=password,
            certificate=certificate or "",
        )
        upsert_cohn_credentials(credential_id, creds)
        camera.provisioning.cohn_credential_id = credential_id
