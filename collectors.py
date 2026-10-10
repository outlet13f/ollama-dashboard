"""Data collectors for the Ollama dashboard: GPU, log, env parsing, Ollama API."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import AsyncIterator

import httpx

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


_BLOCK = 64 * 1024


def tail_log(path: Path, n: int, exclude: str | None = None) -> list[str] | None:
    """Return the last n lines of the file, reading backward in blocks.

    With ``exclude``, lines containing that substring are skipped and the scan
    continues further back until n matching lines are found or the file starts.
    None if the file does not exist.
    """
    try:
        f = path.open("rb")
    except FileNotFoundError:
        return None
    found: list[str] = []
    with f:
        f.seek(0, 2)
        pos = f.tell()
        buf = b""
        first = True
        while pos > 0 and len(found) < n:
            step = min(_BLOCK, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
            if first:
                first = False
                if buf.endswith(b"\n"):
                    buf = buf[:-1]
            parts = buf.split(b"\n")
            buf = parts[0]  # may still be incomplete; completed by the next block
            for raw in reversed(parts[1:]):
                line = raw.decode("utf-8", errors="replace").rstrip("\r")
                if exclude and exclude in line:
                    continue
                found.append(line)
                if len(found) >= n:
                    break
        if len(found) < n and pos == 0 and (buf or not found):
            line = buf.decode("utf-8", errors="replace").rstrip("\r")
            if buf and not (exclude and exclude in line):
                found.append(line)
    return list(reversed(found))


def parse_env_from_log(lines: list[str]) -> dict[str, str]:
    """Extract OLLAMA_* settings from the last server-config line Ollama logged."""
    for line in reversed(lines):
        if _ENV_MARKER in line:
            return dict(_ENV_RE.findall(line))
    return {}


class EnvTracker:
    """Tracks Ollama's startup config line in server.log incrementally.

    Each read() scans only the bytes appended since the previous call, so the
    config line is found even when thousands of request-log lines follow it.
    A shrinking file (log rotation) resets the cache.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._offset = 0
        self._env: dict[str, str] = {}

    def read(self) -> dict[str, str]:
        try:
            size = self.path.stat().st_size
        except OSError:
            self._offset, self._env = 0, {}
            return {}
        if size < self._offset:
            self._offset, self._env = 0, {}
        if size == self._offset:
            return self._env
        with self.path.open("rb") as f:
            f.seek(self._offset)
            data = f.read(size - self._offset)
        end = data.rfind(b"\n")
        if end < 0:
            return self._env
        self._offset += end + 1
        env = parse_env_from_log(data[:end].decode("utf-8", errors="replace").splitlines())
        if env:
            self._env = env
        return self._env


def default_log_path() -> Path:
    override = os.environ.get("OLLAMA_LOG_PATH")
    if override:
        return Path(override)
    return Path(os.environ["LOCALAPPDATA"]) / "Ollama" / "server.log"


# ---------------------------------------------------------------------------
# nvidia-smi
# ---------------------------------------------------------------------------
_NVSMI_CMD = [
    "nvidia-smi",
    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
    "--format=csv,noheader,nounits",
]


def read_gpu(timeout: float = 3.0) -> dict | None:
    """Run nvidia-smi and parse the first GPU line. None if unavailable."""
    try:
        proc = subprocess.run(_NVSMI_CMD, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return parse_gpu_csv(proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else "")


# ---------------------------------------------------------------------------
# Ollama API client
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_PORT = "11434"
_WILDCARD_HOSTS = {"0.0.0.0", "::", ""}


def normalize_host(value: str | None) -> str:
    """Turn an OLLAMA_HOST value into a connectable base URL.

    Ollama's own OLLAMA_HOST is a *bind* address, so wildcard hosts become
    127.0.0.1 and a missing port becomes 11434.
    """
    v = (value or "").strip()
    scheme = "http"
    if "://" in v:
        scheme, v = v.split("://", 1)
    v = v.rstrip("/")
    if v.startswith("["):
        host, _, rest = v[1:].partition("]")
        port = rest.lstrip(":")
    elif v.count(":") == 1:
        host, _, port = v.partition(":")
    else:
        host, port = v, ""
    if host in _WILDCARD_HOSTS:
        host = "127.0.0.1"
    return f"{scheme}://{host}:{port or DEFAULT_OLLAMA_PORT}"


class OllamaError(Exception):
    """Ollama returned a non-2xx response."""


class OllamaClient:
    def __init__(self, base_url: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = normalize_host(base_url or os.environ.get("OLLAMA_HOST"))
        self._client = httpx.AsyncClient(base_url=self.base_url, transport=transport, timeout=3.0)
        self._stream_timeout = httpx.Timeout(connect=3.0, read=None, write=3.0, pool=3.0)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get_json(self, path: str) -> dict | None:
        try:
            resp = await self._client.get(path)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError):
            return None

    async def version(self) -> str | None:
        data = await self._get_json("/api/version")
        return data.get("version") if data else None

    async def ps(self) -> list[dict]:
        data = await self._get_json("/api/ps")
        return data.get("models", []) if data else []

    async def tags(self) -> list[dict]:
        data = await self._get_json("/api/tags")
        return data.get("models", []) if data else []

    async def _generate_keep_alive(self, model: str, keep_alive: int) -> None:
        try:
            resp = await self._client.post("/api/generate", json={"model": model, "keep_alive": keep_alive},
                                           timeout=httpx.Timeout(connect=3.0, read=120.0, write=3.0, pool=3.0))
        except httpx.HTTPError as e:
            raise OllamaError(str(e)) from e
        if resp.status_code >= 300:
            raise OllamaError(_error_text(resp))

    async def load(self, model: str) -> None:
        await self._generate_keep_alive(model, -1)

    async def unload(self, model: str) -> None:
        await self._generate_keep_alive(model, 0)

    async def chat_stream(self, model: str, prompt: str) -> AsyncIterator[str]:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}
        try:
            async with self._client.stream("POST", "/api/chat", json=payload,
                                           timeout=self._stream_timeout) as resp:
                if resp.status_code >= 300:
                    await resp.aread()
                    yield json.dumps({"error": _error_text(resp)})
                    return
                async for line in resp.aiter_lines():
                    if line:
                        yield line
        except httpx.HTTPError as e:
            yield json.dumps({"error": str(e) or type(e).__name__})


def _error_text(resp: httpx.Response) -> str:
    try:
        return resp.json().get("error") or resp.text
    except ValueError:
        return resp.text or f"HTTP {resp.status_code}"
