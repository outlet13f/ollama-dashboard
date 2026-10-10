from pathlib import Path

from collectors import parse_gpu_csv, tail_log, parse_env_from_log, default_log_path

SAMPLE = "NVIDIA GeForce RTX 3080, 1, 8234, 10240, 39, 37.48, 320.00"


def test_parse_gpu_csv_ok():
    assert parse_gpu_csv(SAMPLE) == {
        "name": "NVIDIA GeForce RTX 3080", "util": 1, "mem_used": 8234,
        "mem_total": 10240, "temp": 39, "power": 37.48, "power_limit": 320.0,
    }


def test_parse_gpu_csv_na_power():
    g = parse_gpu_csv("NVIDIA T400, 0, 100, 2048, 30, [N/A], [Not Supported]")
    assert g["power"] is None and g["power_limit"] is None and g["mem_total"] == 2048


def test_parse_gpu_csv_empty_and_short():
    assert parse_gpu_csv("") is None
    assert parse_gpu_csv("a, b, c") is None


def test_tail_log_shorter_than_n(tmp_path):
    p = tmp_path / "s.log"
    p.write_text("a\nb\n", encoding="utf-8")
    assert tail_log(p, 100) == ["a", "b"]


def test_tail_log_longer_than_n(tmp_path):
    p = tmp_path / "s.log"
    p.write_text("\n".join(str(i) for i in range(50)), encoding="utf-8")
    assert tail_log(p, 3) == ["47", "48", "49"]


def test_tail_log_missing(tmp_path):
    assert tail_log(tmp_path / "nope.log", 10) is None


def test_tail_log_invalid_utf8(tmp_path):
    p = tmp_path / "s.log"
    p.write_bytes(b"ok\n\xff\xfe bad\n")
    lines = tail_log(p, 10)
    assert len(lines) == 2 and lines[0] == "ok"


ENV_LINE = (
    'time=1 level=INFO msg="server config" env="map[OLLAMA_FLASH_ATTENTION:true '
    'OLLAMA_KEEP_ALIVE:2562047h47m16.854775807s OLLAMA_KV_CACHE_TYPE:q8_0 '
    'OLLAMA_ORIGINS:[http://localhost]]"'
)


def test_parse_env_picks_last_config_line():
    old = ENV_LINE.replace("q8_0", "f16")
    env = parse_env_from_log(["junk", old, "junk", ENV_LINE, "tail"])
    assert env["OLLAMA_FLASH_ATTENTION"] == "true"
    assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert env["OLLAMA_KEEP_ALIVE"].startswith("2562047h")


def test_parse_env_none():
    assert parse_env_from_log(["nothing here"]) == {}


def test_default_log_path_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_LOG_PATH", r"C:\x\y.log")
    assert default_log_path() == Path(r"C:\x\y.log")


# ---- Task 2: OllamaClient / read_gpu ----
import json
import subprocess

import httpx
import pytest

from collectors import OllamaClient, OllamaError, read_gpu


def make_client(handler):
    return OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_version_ok():
    c = make_client(lambda r: httpx.Response(200, json={"version": "0.32.13"}))
    assert await c.version() == "0.32.13"


@pytest.mark.asyncio
async def test_version_offline_returns_none():
    def h(r):
        raise httpx.ConnectError("refused")
    assert await make_client(h).version() is None


@pytest.mark.asyncio
async def test_version_timeout_returns_none():
    def h(r):
        raise httpx.ReadTimeout("slow")
    assert await make_client(h).version() is None


@pytest.mark.asyncio
async def test_ps_and_tags():
    def h(r):
        if r.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "a"}]})
        if r.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "a"}, {"name": "b"}]})
        return httpx.Response(404)
    c = make_client(h)
    assert [m["name"] for m in await c.ps()] == ["a"]
    assert len(await c.tags()) == 2


@pytest.mark.asyncio
async def test_load_and_unload_send_keep_alive():
    seen = []

    def h(r):
        seen.append(json.loads(r.content))
        return httpx.Response(200, json={"done": True})
    c = make_client(h)
    await c.load("m")
    await c.unload("m")
    assert seen[0] == {"model": "m", "keep_alive": -1}
    assert seen[1] == {"model": "m", "keep_alive": 0}


@pytest.mark.asyncio
async def test_load_error_raises():
    c = make_client(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(OllamaError):
        await c.load("nope")


@pytest.mark.asyncio
async def test_chat_stream_passes_lines_through():
    body = (b'{"message":{"content":"hi"},"done":false}\n'
            b'{"done":true,"eval_count":2,"eval_duration":1000}\n')
    c = make_client(lambda r: httpx.Response(200, content=body))
    lines = [l async for l in c.chat_stream("m", "p")]
    assert len(lines) == 2 and json.loads(lines[1])["done"] is True


@pytest.mark.asyncio
async def test_chat_stream_error_chunk_on_404():
    c = make_client(lambda r: httpx.Response(404, json={"error": "model not found"}))
    lines = [l async for l in c.chat_stream("nope", "p")]
    assert len(lines) == 1 and "error" in json.loads(lines[0])


def test_read_gpu_missing_binary(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr(subprocess, "run", boom)
    assert read_gpu() is None


def test_read_gpu_parses(monkeypatch):
    class R:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 3080, 5, 1000, 10240, 40, 50.0, 320.0\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    assert read_gpu()["util"] == 5
