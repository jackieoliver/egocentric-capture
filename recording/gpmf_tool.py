from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .paths import haptica_home, repo_root


@dataclass(frozen=True)
class GpmfTool:
    bin_path: Path
    version: Optional[str] = None


def gpmf_parser_repo() -> Path:
    return repo_root() / "vendor" / "gopro" / "gpmf-parser"


def gpmf_extract_bin() -> Path:
    override = os.environ.get("TRANSCRIPTIONS_GPMF_EXTRACT_BIN")
    if override:
        return Path(override).expanduser()
    return haptica_home() / "state" / "bin" / "transcriptions_gpmf_extract"


def ensure_gpmf_extract_tool(*, force: bool = False) -> GpmfTool:
    bin_path = gpmf_extract_bin()
    if bin_path.exists() and not force:
        return GpmfTool(bin_path=bin_path, version=_tool_version(bin_path))

    src_repo = gpmf_parser_repo()
    include_dir = src_repo
    demo_dir = src_repo / "demo"
    tool_src = repo_root() / "recording" / "tools" / "gpmf_extract.c"
    if not tool_src.exists():
        raise RuntimeError("gpmf_tool_source_missing")
    if not (include_dir / "GPMF_parser.h").exists():
        raise RuntimeError("gpmf_parser_submodule_missing")

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = bin_path.with_suffix(".tmp")
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)

    sources = [
        tool_src,
        src_repo / "GPMF_parser.c",
        src_repo / "GPMF_utils.c",
        demo_dir / "GPMF_mp4reader.c",
    ]
    for src in sources:
        if not src.exists():
            raise RuntimeError(f"gpmf_build_missing:{src}")

    cc = os.environ.get("CC") or "cc"
    cmd = [
        cc,
        "-O2",
        "-std=c11",
        "-D_GNU_SOURCE",  # Required for strdup, strtok_r, fseeko
        "-I",
        str(include_dir),
        "-I",
        str(demo_dir),
        "-o",
        str(tmp_path),
        *[str(p) for p in sources],
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip() or "gpmf_build_failed")
    tmp_path.replace(bin_path)
    try:
        bin_path.chmod(0o755)
    except Exception:
        pass
    return GpmfTool(bin_path=bin_path, version=_tool_version(bin_path))


def _tool_version(bin_path: Path) -> Optional[str]:
    try:
        result = subprocess.run([str(bin_path), "--help"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return "v1"
    except Exception:
        return None
    return None

