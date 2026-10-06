from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def haptica_home() -> Path:
    """
    Writable base directory for all local state.

    Defaults to the repository root for developer workflows.
    Override with `HAPTICA_HOME` when running from an app bundle or when you
    want state outside the repo.
    """
    override = os.environ.get("HAPTICA_HOME")
    if override:
        return Path(override).expanduser()
    return repo_root()


def sessions_root() -> Path:
    """
    Canonical session store root.

    Defaults to `<HAPTICA_HOME>/transcriptions`, override with `HAPTICA_SESSIONS_DIR`.
    """
    override = os.environ.get("HAPTICA_SESSIONS_DIR")
    if override:
        return Path(override).expanduser()
    return haptica_home() / "transcriptions"


def active_session_path() -> Path:
    """
    Pointer to the active session id.

    Defaults to `<HAPTICA_HOME>/state/active_session`, override with `HAPTICA_ACTIVE_SESSION_PATH`.
    """
    override = os.environ.get("HAPTICA_ACTIVE_SESSION_PATH")
    if override:
        return Path(override).expanduser()
    return haptica_home() / "state" / "active_session"


def registry_cache_dir() -> Path:
    """
    Local cache for the Drive-backed registry.

    Defaults to `<HAPTICA_HOME>/state/registry_cache`, override with `TRANSCRIPTIONS_REGISTRY_CACHE_DIR`.
    """
    override = os.environ.get("TRANSCRIPTIONS_REGISTRY_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return haptica_home() / "state" / "registry_cache"
