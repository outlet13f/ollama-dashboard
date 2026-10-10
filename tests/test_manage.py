"""Model management: pull / delete / show."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import create_app
from collectors import OllamaClient, OllamaError

PULL_BODY = (
    b'{"status":"pulling manifest"}\n'
    b'{"status":"pulling abc","digest":"sha256:abc","total":1000,"completed":500}\n'
    b'{"status":"success"}\n'
)
SHOW_BODY = {"license": "MIT", "modelfile": "FROM x", "template": "{{ .Prompt }}",
             "details": {"family": "qwen35"}, "model_info": {"qwen35.context_length": 262144},
             "capabilities": ["completion", "vision"], "modified_at": "2026-10-01T00:00:00Z"}


def handler(r):
    p, m = r.url.path, r.method
    body = json.loads(r.content) if r.content else {}
    if p == "/api/pull" and m == "POST":
        if body.get("model") == "nope":
            return httpx.Response(404, json={"error": "pull model manifest: file does not exist"})
        return httpx.Response(200, content=PULL_BODY)
    if p == "/api/delete" and m == "DELETE":
        if body.get("model") == "nope":
            return httpx.Response(404, json={"error": "model 'nope' not found"})
        return httpx.Response(200)
    if p == "/api/show" and m == "POST":
        if body.get("model") == "nope":
            return httpx.Response(404, json={"error": "model 'nope' not found"})
        return httpx.Response(200, json=SHOW_BODY)
    return httpx.Response(404, json={"error": "unexpected " + m + " " + p})


def make_client():
    return OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler))


# ---- collectors ----
@pytest.mark.asyncio
async def test_pull_stream_passes_progress_lines():
    lines = [json.loads(l) async for l in make_client().pull_stream("qwen3.5:0.8b")]
    assert lines[1]["completed"] == 500 and lines[-1]["status"] == "success"


@pytest.mark.asyncio
async def test_pull_stream_error_chunk():
    lines = [json.loads(l) async for l in make_client().pull_stream("nope")]
    assert len(lines) == 1 and "does not exist" in lines[0]["error"]


@pytest.mark.asyncio
async def test_delete_sends_delete_method():
    seen = []

    def h(r):
        seen.append((r.method, r.url.path, json.loads(r.content)))
        return httpx.Response(200)
    await OllamaClient("http://ollama.test", transport=httpx.MockTransport(h)).delete("m")
    assert seen == [("DELETE", "/api/delete", {"model": "m"})]


@pytest.mark.asyncio
async def test_delete_unknown_raises():
    with pytest.raises(OllamaError):
        await make_client().delete("nope")


@pytest.mark.asyncio
async def test_show_returns_details():
    d = await make_client().show("qwen3.5:9b")
    assert d["model_info"]["qwen35.context_length"] == 262144


@pytest.mark.asyncio
async def test_show_unknown_raises():
    with pytest.raises(OllamaError):
        await make_client().show("nope")


# ---- api ----
def api(tmp_path):
    app = create_app(make_client(), gpu_reader=lambda: None, log_path=tmp_path / "server.log")
    return TestClient(app, base_url="http://127.0.0.1")


def test_api_pull_streams_ndjson(tmp_path):
    with api(tmp_path).stream("POST", "/api/pull", json={"model": "qwen3.5:0.8b"}) as r:
        assert r.headers["content-type"].startswith("application/x-ndjson")
        lines = [json.loads(l) for l in r.iter_lines() if l]
    assert lines[-1]["status"] == "success"


def test_api_pull_error_chunk(tmp_path):
    with api(tmp_path).stream("POST", "/api/pull", json={"model": "nope"}) as r:
        lines = [json.loads(l) for l in r.iter_lines() if l]
    assert len(lines) == 1 and "error" in lines[0]


def test_api_delete_ok_and_502(tmp_path):
    c = api(tmp_path)
    assert c.post("/api/delete", json={"model": "x"}).json() == {"ok": True}
    r = c.post("/api/delete", json={"model": "nope"})
    assert r.status_code == 502 and "not found" in r.json()["detail"]


def test_api_show_ok_and_502(tmp_path):
    c = api(tmp_path)
    assert c.get("/api/show", params={"model": "qwen3.5:9b"}).json()["capabilities"] == ["completion", "vision"]
    assert c.get("/api/show", params={"model": "nope"}).status_code == 502
    assert c.get("/api/show").status_code == 422
