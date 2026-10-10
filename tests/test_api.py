import json

import httpx
from fastapi.testclient import TestClient

from collectors import OllamaClient
from app import create_app

ENV_LINE = ('msg="server config" env="map[OLLAMA_FLASH_ATTENTION:true '
            'OLLAMA_KEEP_ALIVE:-1 OLLAMA_KV_CACHE_TYPE:q8_0]"')

CHAT_BODY = (
    b'{"message":{"content":"hi"},"done":false}\n'
    b'{"done":true,"eval_count":2,"eval_duration":1000000000,"prompt_eval_count":3,'
    b'"prompt_eval_duration":500000000,"load_duration":10,"total_duration":20}\n'
)


def ollama_ok(r):
    p = r.url.path
    if p == "/api/version":
        return httpx.Response(200, json={"version": "0.32.13"})
    if p == "/api/ps":
        return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b", "size_vram": 1}]})
    if p == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b"}, {"name": "ornith:latest"}]})
    if p == "/api/generate":
        body = json.loads(r.content)
        if body["model"] == "nope":
            return httpx.Response(404, json={"error": "model 'nope' not found"})
        return httpx.Response(200, json={"done": True})
    if p == "/api/chat":
        body = json.loads(r.content)
        if body["model"] == "nope":
            return httpx.Response(404, json={"error": "model 'nope' not found"})
        return httpx.Response(200, content=CHAT_BODY)
    return httpx.Response(404)


def ollama_down(r):
    raise httpx.ConnectError("refused")


GPU_FIXED = {"name": "RTX", "util": 1, "mem_used": 1, "mem_total": 2, "temp": 3, "power": 4.0, "power_limit": 5.0}


def make(tmp_path, handler=ollama_ok, gpu=lambda: GPU_FIXED, write_log=True):
    log = tmp_path / "server.log"
    if write_log:
        log.write_text("boot\n" + ENV_LINE + "\n" + "\n".join(f"line{i}" for i in range(200)), encoding="utf-8")
    client = OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler))
    app = create_app(client, gpu_reader=gpu, log_path=log)
    return TestClient(app)


def test_status_online(tmp_path):
    d = make(tmp_path).get("/api/status").json()
    assert d["ollama"] == {"online": True, "version": "0.32.13"}
    assert d["env"]["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert d["gpu"]["name"] == "RTX"
    assert [m["name"] for m in d["loaded"]] == ["qwen3.5:9b"]
    assert len(d["models"]) == 2


def test_status_offline_keeps_other_fields(tmp_path):
    d = make(tmp_path, handler=ollama_down).get("/api/status").json()
    assert d["ollama"] == {"online": False, "version": None}
    assert d["loaded"] == [] and d["models"] == []
    assert d["gpu"]["name"] == "RTX" and d["env"]["OLLAMA_FLASH_ATTENTION"] == "true"


def test_status_gpu_unavailable(tmp_path):
    d = make(tmp_path, gpu=lambda: None).get("/api/status").json()
    assert d["gpu"] is None and d["ollama"]["online"] is True


def test_log_tail_default_and_custom(tmp_path):
    c = make(tmp_path)
    assert len(c.get("/api/log").json()["lines"]) == 100
    r = c.get("/api/log?lines=5").json()
    assert r["lines"] == [f"line{i}" for i in range(195, 200)] and r["path"].endswith("server.log")


def test_log_missing_file(tmp_path):
    assert make(tmp_path, write_log=False).get("/api/log").json()["lines"] is None


def test_log_lines_out_of_range(tmp_path):
    c = make(tmp_path)
    assert c.get("/api/log?lines=0").status_code == 422
    assert c.get("/api/log?lines=5000").status_code == 422
