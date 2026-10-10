"""Data collectors for the Ollama dashboard: GPU, log, env parsing, Ollama API."""
from __future__ import annotations

import os
import re
from pathlib import Path

_ENV_RE = re.compile(r"(OLLAMA_[A-Z_]+):([^\s\]]*)")
_ENV_MARKER = "OLLAMA_FLASH_ATTENTION:"


def parse_gpu_csv(text: str) -> dict | None:
    """Parse one nvidia-smi CSV line (name, util, mem_used, mem_total, temp, power, power_limit)."""
    fields = [f.strip() for f in text.split(",")]
    if len(fields) < 7:
        return None

    def _int(v: str) -> int:
        return int(float(v))

    def _float_or_none(v: str) -> float | None:
        return None if v.startswith("[") else float(v)

    return {
        "name": fields[0],
        "util": _int(fields[1]),
        "mem_used": _int(fields[2]),
        "mem_total": _int(fields[3]),
        "temp": _int(fields[4]),
        "power": _float_or_none(fields[5]),
        "power_limit": _float_or_none(fields[6]),
    }


def tail_log(path: Path, n: int) -> list[str] | None:
    """Return the last n lines of the file, or None if it does not exist."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    return text.splitlines()[-n:]


def parse_env_from_log(lines: list[str]) -> dict[str, str]:
    """Extract OLLAMA_* settings from the last server-config line Ollama logged."""
    for line in reversed(lines):
        if _ENV_MARKER in line:
            return dict(_ENV_RE.findall(line))
    return {}


def default_log_path() -> Path:
    override = os.environ.get("OLLAMA_LOG_PATH")
    if override:
        return Path(override)
    return Path(os.environ["LOCALAPPDATA"]) / "Ollama" / "server.log"
