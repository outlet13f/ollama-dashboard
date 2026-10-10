# Ollama Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 로컬 Ollama 서버와 GPU 상태를 보여주고, 모델 로드/언로드와 속도 측정용 테스트 프롬프트를 제공하는 단일 페이지 웹 대시보드.

**Architecture:** FastAPI 앱 하나가 네 개의 수집기(Ollama HTTP API, nvidia-smi, server.log tail, 로그에서 환경변수 파싱)를 묶어 JSON으로 내보내고, 테스트 프롬프트는 Ollama의 NDJSON 스트림을 그대로 중계한다. 프론트는 빌드 없는 바닐라 JS 한 파일이 2초 간격으로 폴링한다. 수집기는 라우트와 분리되어 GPU나 Ollama 없이 단위 테스트된다.

**Tech Stack:** Python 3.14, FastAPI, uvicorn, httpx (AsyncClient + MockTransport), pytest, pytest-asyncio. 프론트는 HTML/CSS/JS 인라인, 외부 CDN 없음.

**Spec:** `docs/superpowers/specs/2026-10-11-ollama-dashboard-design.md`

## Global Constraints

- 서버는 `127.0.0.1`에만 바인딩. 기본 포트 `8585`. `--port`, `--no-browser` 옵션.
- CORS 미설정. 외부 CDN 없음. 프론트 빌드 도구 없음.
- 의존성은 `fastapi`, `uvicorn`, `httpx`만. dev 의존성 `pytest`, `pytest-asyncio`.
- Ollama 기본 주소 `http://127.0.0.1:11434`. 환경변수 `OLLAMA_HOST`가 있으면 그 값 사용.
- 로그 경로 기본값 `%LOCALAPPDATA%\Ollama\server.log`, 환경변수 `OLLAMA_LOG_PATH`로 덮어씀.
- nvidia-smi 명령: `nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits`, 타임아웃 3초.
- Ollama 상태 조회 타임아웃 3초. chat 스트림은 connect 3초, read 무제한.
- `/api/log`의 `lines`는 1~1000. 기본 100.
- 테스트 실행 명령: `python -m pytest -q`.

## Review Focus

1. nvidia-smi가 전력 값에 `[N/A]`나 `[Not Supported]`를 내는 GPU에서도 파서가 죽지 않고 해당 필드만 `null`이어야 한다. → Task 1 `test_parse_gpu_csv_na_power`
2. server.log에 UTF-8이 아닌 바이트가 섞여 있어도 tail이 예외 없이 읽혀야 한다. → Task 1 `test_tail_log_invalid_utf8`
3. Ollama가 응답은 하지만 3초 넘게 느릴 때 `/api/status`가 매달리지 않고 `online=false`로 돌아와야 폴링이 쌓이지 않는다. → Task 2 `test_version_timeout_returns_none`
4. 존재하지 않는 모델로 `/api/chat`을 보내면 Ollama가 404 JSON을 돌려주는데, 스트림이 `{"error": ...}` 청크 하나로 끝나야 화면에 오류가 보인다. → Task 4 `test_chat_unknown_model_emits_error_chunk`
5. `/api/log?lines=0`이나 `lines=5000`처럼 범위를 벗어난 값은 422로 거부해야 한다. → Task 3 `test_log_lines_out_of_range`

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `collectors.py` | 순수 파서 3개(`parse_gpu_csv`, `tail_log`, `parse_env_from_log`), 서브프로세스 래퍼 `read_gpu`, 경로 헬퍼 `default_log_path`, `OllamaClient` |
| `app.py` | `create_app(...)` 팩토리와 라우트, `main()` CLI |
| `static/index.html` | 화면 전체 |
| `tests/test_collectors.py` | 파서와 OllamaClient 테스트 |
| `tests/test_api.py` | 엔드포인트 테스트 |
| `requirements.txt`, `requirements-dev.txt`, `README.md` | |

---

### Task 1: 프로젝트 골격과 순수 파서

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `collectors.py`, `tests/__init__.py`, `tests/test_collectors.py`

**Interfaces:**
- Produces:
  - `parse_gpu_csv(text: str) -> dict | None` — 반환 키: `name: str, util: int, mem_used: int, mem_total: int, temp: int, power: float | None, power_limit: float | None`. 빈 문자열이거나 필드가 7개 미만이면 `None`.
  - `tail_log(path: Path, n: int) -> list[str] | None` — 마지막 n줄(개행 제거). 파일 없으면 `None`.
  - `parse_env_from_log(lines: list[str]) -> dict[str, str]` — `OLLAMA_FLASH_ATTENTION:`을 포함한 **마지막** 줄에서 정규식 `(OLLAMA_[A-Z_]+):([^\s\]]*)`로 모든 쌍 추출. 없으면 `{}`.
  - `default_log_path() -> Path` — `OLLAMA_LOG_PATH` 환경변수가 있으면 그 값, 없으면 `Path(os.environ["LOCALAPPDATA"]) / "Ollama" / "server.log"`.

- [ ] **Step 1: 의존성 파일 작성**

`requirements.txt`: `fastapi`, `uvicorn`, `httpx` 세 줄. `requirements-dev.txt`: `-r requirements.txt`, `pytest`, `pytest-asyncio`.

- [ ] **Step 2: 가상환경 생성 및 설치**

Run: `python -m venv .venv; .\.venv\Scripts\pip install -r requirements-dev.txt`
Expected: 오류 없이 종료. 이후 모든 `python`/`pytest` 명령은 `.\.venv\Scripts\` 아래 것을 사용.

- [ ] **Step 3: 실패하는 테스트 작성** (`tests/test_collectors.py`)

```python
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
    p = tmp_path / "s.log"; p.write_text("a\nb\n", encoding="utf-8")
    assert tail_log(p, 100) == ["a", "b"]

def test_tail_log_longer_than_n(tmp_path):
    p = tmp_path / "s.log"; p.write_text("\n".join(str(i) for i in range(50)), encoding="utf-8")
    assert tail_log(p, 3) == ["47", "48", "49"]

def test_tail_log_missing(tmp_path):
    assert tail_log(tmp_path / "nope.log", 10) is None

def test_tail_log_invalid_utf8(tmp_path):
    p = tmp_path / "s.log"; p.write_bytes(b"ok\n\xff\xfe bad\n")
    lines = tail_log(p, 10)
    assert len(lines) == 2 and lines[0] == "ok"

ENV_LINE = 'time=1 level=INFO msg="server config" env="map[OLLAMA_FLASH_ATTENTION:true OLLAMA_KEEP_ALIVE:2562047h47m16.854775807s OLLAMA_KV_CACHE_TYPE:q8_0 OLLAMA_ORIGINS:[http://localhost]]"'

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
```

- [ ] **Step 4: 실패 확인**

Run: `python -m pytest tests/test_collectors.py -q`
Expected: `ImportError` 또는 `ModuleNotFoundError: collectors`

- [ ] **Step 5: `collectors.py`에 네 함수 구현**

`parse_gpu_csv`: 쉼표로 나눠 strip. 정수 필드는 `int(float(x))`, 전력 두 필드는 `[`로 시작하면 `None`. `tail_log`: `path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]`.

- [ ] **Step 6: 통과 확인**

Run: `python -m pytest tests/test_collectors.py -q`
Expected: `10 passed`

- [ ] **Step 7: Commit**

```bash
git add requirements.txt requirements-dev.txt collectors.py tests/
git commit -m "feat: add gpu/log/env parsers with tests"
```

---

### Task 2: OllamaClient와 read_gpu

**Files:**
- Modify: `collectors.py`
- Modify: `tests/test_collectors.py`

**Interfaces:**
- Produces:
  - `class OllamaError(Exception)`
  - `class OllamaClient:`
    - `__init__(self, base_url: str | None = None, transport: httpx.AsyncBaseTransport | None = None)` — `base_url`이 None이면 `OLLAMA_HOST` 환경변수, 그것도 없으면 `http://127.0.0.1:11434`. 내부에 `httpx.AsyncClient(timeout=3.0)` 하나.
    - `async version(self) -> str | None` — 실패(연결 오류, 타임아웃, 비정상 상태코드) 시 `None`.
    - `async ps(self) -> list[dict]` — `/api/ps`의 `models`. 실패 시 `[]`.
    - `async tags(self) -> list[dict]` — `/api/tags`의 `models`. 실패 시 `[]`.
    - `async load(self, model: str) -> None` — POST `/api/generate` `{"model": model, "keep_alive": -1}`. 비2xx면 `OllamaError(응답 본문)`.
    - `async unload(self, model: str) -> None` — 위와 같되 `keep_alive: 0`.
    - `async chat_stream(self, model: str, prompt: str) -> AsyncIterator[str]` — POST `/api/chat` `{"model", "messages":[{"role":"user","content":prompt}], "stream": true}`, 타임아웃 `httpx.Timeout(connect=3.0, read=None, write=3.0, pool=3.0)`. 응답 줄을 그대로 yield. 비2xx나 예외면 `json.dumps({"error": <메시지>})` 한 줄을 yield하고 종료.
    - `async aclose(self) -> None`
  - `read_gpu(timeout: float = 3.0) -> dict | None` — `subprocess.run(..., capture_output=True, text=True, timeout=timeout)` 후 `parse_gpu_csv(stdout.strip())`. `FileNotFoundError`, `TimeoutExpired`, 비0 종료코드면 `None`.

- [ ] **Step 1: 실패하는 테스트 추가**

```python
import json, httpx, pytest
from collectors import OllamaClient, OllamaError, read_gpu

def make_client(handler):
    return OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler))

@pytest.mark.asyncio
async def test_version_ok():
    c = make_client(lambda r: httpx.Response(200, json={"version": "0.32.13"}))
    assert await c.version() == "0.32.13"

@pytest.mark.asyncio
async def test_version_offline_returns_none():
    def h(r): raise httpx.ConnectError("refused")
    assert await make_client(h).version() is None

@pytest.mark.asyncio
async def test_version_timeout_returns_none():
    def h(r): raise httpx.ReadTimeout("slow")
    assert await make_client(h).version() is None

@pytest.mark.asyncio
async def test_ps_and_tags():
    def h(r):
        if r.url.path == "/api/ps": return httpx.Response(200, json={"models": [{"name": "a"}]})
        if r.url.path == "/api/tags": return httpx.Response(200, json={"models": [{"name": "a"}, {"name": "b"}]})
        return httpx.Response(404)
    c = make_client(h)
    assert [m["name"] for m in await c.ps()] == ["a"]
    assert len(await c.tags()) == 2

@pytest.mark.asyncio
async def test_load_and_unload_send_keep_alive():
    seen = []
    def h(r):
        seen.append(json.loads(r.content)); return httpx.Response(200, json={"done": True})
    c = make_client(h)
    await c.load("m"); await c.unload("m")
    assert seen[0] == {"model": "m", "keep_alive": -1}
    assert seen[1] == {"model": "m", "keep_alive": 0}

@pytest.mark.asyncio
async def test_load_error_raises():
    c = make_client(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(OllamaError):
        await c.load("nope")

@pytest.mark.asyncio
async def test_chat_stream_passes_lines_through():
    body = b'{"message":{"content":"hi"},"done":false}\n{"done":true,"eval_count":2,"eval_duration":1000}\n'
    c = make_client(lambda r: httpx.Response(200, content=body))
    lines = [l async for l in c.chat_stream("m", "p")]
    assert len(lines) == 2 and json.loads(lines[1])["done"] is True

@pytest.mark.asyncio
async def test_chat_stream_error_chunk_on_404():
    c = make_client(lambda r: httpx.Response(404, json={"error": "model not found"}))
    lines = [l async for l in c.chat_stream("nope", "p")]
    assert len(lines) == 1 and "error" in json.loads(lines[0])

def test_read_gpu_missing_binary(monkeypatch):
    import subprocess
    def boom(*a, **k): raise FileNotFoundError
    monkeypatch.setattr(subprocess, "run", boom)
    assert read_gpu() is None

def test_read_gpu_parses(monkeypatch):
    import subprocess
    class R: returncode = 0; stdout = "NVIDIA GeForce RTX 3080, 5, 1000, 10240, 40, 50.0, 320.0\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    assert read_gpu()["util"] == 5
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_collectors.py -q`
Expected: `ImportError: cannot import name 'OllamaClient'`

- [ ] **Step 3: `OllamaClient`, `OllamaError`, `read_gpu` 구현**

`chat_stream`은 `async with self._client.stream("POST", ...) as resp:` 후 상태코드 확인, `async for line in resp.aiter_lines(): if line: yield line`. 예외는 `httpx.HTTPError` 전체를 잡는다.

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest tests/test_collectors.py -q`
Expected: `20 passed`

- [ ] **Step 5: Commit**

```bash
git add collectors.py tests/test_collectors.py
git commit -m "feat: add OllamaClient and nvidia-smi reader"
```

---

### Task 3: `create_app`과 조회 엔드포인트

**Files:**
- Create: `app.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: Task 1, 2의 모든 함수와 `OllamaClient`.
- Produces:
  - `create_app(client: OllamaClient, gpu_reader: Callable[[], dict | None] = read_gpu, log_path: Path | None = None) -> FastAPI` — `log_path`가 None이면 `default_log_path()`.
  - `GET /api/status` → `{"ollama": {"online": bool, "version": str | None}, "env": dict, "gpu": dict | None, "loaded": list, "models": list}`. `online`은 `version()`이 None이 아닐 때 True. 네 수집기는 `asyncio.gather`로 동시 호출, `gpu_reader`와 로그 읽기는 `asyncio.to_thread`.
  - `GET /api/log?lines=100` → `{"path": str, "lines": list[str] | None}`. `lines`는 `Query(100, ge=1, le=1000)`.

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/test_api.py`)

```python
import json, httpx, pytest
from fastapi.testclient import TestClient
from collectors import OllamaClient
from app import create_app

ENV_LINE = 'msg="server config" env="map[OLLAMA_FLASH_ATTENTION:true OLLAMA_KEEP_ALIVE:-1 OLLAMA_KV_CACHE_TYPE:q8_0]"'

def ollama_ok(r):
    p = r.url.path
    if p == "/api/version": return httpx.Response(200, json={"version": "0.32.13"})
    if p == "/api/ps": return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b", "size_vram": 1}]})
    if p == "/api/tags": return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b"}, {"name": "ornith:latest"}]})
    if p in ("/api/generate",):
        body = json.loads(r.content)
        if body["model"] == "nope": return httpx.Response(404, json={"error": "model 'nope' not found"})
        return httpx.Response(200, json={"done": True})
    if p == "/api/chat":
        body = json.loads(r.content)
        if body["model"] == "nope": return httpx.Response(404, json={"error": "model 'nope' not found"})
        return httpx.Response(200, content=b'{"message":{"content":"hi"},"done":false}\n{"done":true,"eval_count":2,"eval_duration":1000000000,"prompt_eval_count":3,"prompt_eval_duration":500000000,"load_duration":10,"total_duration":20}\n')
    return httpx.Response(404)

def ollama_down(r): raise httpx.ConnectError("refused")

def make(tmp_path, handler=ollama_ok, gpu=lambda: {"name": "RTX", "util": 1, "mem_used": 1, "mem_total": 2, "temp": 3, "power": 4.0, "power_limit": 5.0}, write_log=True):
    log = tmp_path / "server.log"
    if write_log: log.write_text("boot\n" + ENV_LINE + "\n" + "\n".join(f"line{i}" for i in range(200)), encoding="utf-8")
    app = create_app(OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler)), gpu_reader=gpu, log_path=log)
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
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_api.py -q`
Expected: `ModuleNotFoundError: app`

- [ ] **Step 3: `app.py`에 `create_app`과 두 GET 라우트 구현**

`/api/status`에서 `gather(..., return_exceptions=True)` 결과 중 예외는 각 필드의 기본값(`None`, `[]`, `{}`)으로 치환.

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest -q`
Expected: `26 passed`

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_api.py
git commit -m "feat: add status and log endpoints"
```

---

### Task 4: 조작 엔드포인트 (load / unload / chat)

**Files:**
- Modify: `app.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: Task 3의 `create_app`, Task 2의 `OllamaClient.load/unload/chat_stream`, `OllamaError`.
- Produces:
  - `POST /api/load` body `{"model": str}` → 200 `{"ok": true}`, `OllamaError`면 502 `{"detail": str}`.
  - `POST /api/unload` 동일.
  - `POST /api/chat` body `{"model": str, "prompt": str}` → `StreamingResponse(media_type="application/x-ndjson")`, 각 줄 뒤에 `\n`.
  - 요청 모델: `class ModelReq(BaseModel): model: str`, `class ChatReq(BaseModel): model: str; prompt: str`.

- [ ] **Step 1: 실패하는 테스트 추가**

```python
def test_load_ok_and_unload_ok(tmp_path):
    c = make(tmp_path)
    assert c.post("/api/load", json={"model": "qwen3.5:9b"}).json() == {"ok": True}
    assert c.post("/api/unload", json={"model": "qwen3.5:9b"}).json() == {"ok": True}

def test_load_unknown_model_502(tmp_path):
    r = make(tmp_path).post("/api/load", json={"model": "nope"})
    assert r.status_code == 502 and "not found" in r.json()["detail"]

def test_load_missing_body_422(tmp_path):
    assert make(tmp_path).post("/api/load", json={}).status_code == 422

def test_chat_streams_ndjson(tmp_path):
    with make(tmp_path).stream("POST", "/api/chat", json={"model": "qwen3.5:9b", "prompt": "hi"}) as r:
        assert r.headers["content-type"].startswith("application/x-ndjson")
        lines = [json.loads(l) for l in r.iter_lines() if l]
    assert lines[0]["message"]["content"] == "hi"
    assert lines[-1]["done"] is True and lines[-1]["eval_count"] == 2

def test_chat_unknown_model_emits_error_chunk(tmp_path):
    with make(tmp_path).stream("POST", "/api/chat", json={"model": "nope", "prompt": "hi"}) as r:
        lines = [json.loads(l) for l in r.iter_lines() if l]
    assert len(lines) == 1 and "not found" in lines[0]["error"]
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_api.py -q`
Expected: 새 테스트 5개가 404로 실패

- [ ] **Step 3: 세 POST 라우트 구현**

- [ ] **Step 4: 통과 확인**

Run: `python -m pytest -q`
Expected: `31 passed`

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_api.py
git commit -m "feat: add load/unload/chat endpoints"
```

---

### Task 5: 프론트엔드, 루트 라우트, CLI

**Files:**
- Create: `static/index.html`, `README.md`
- Modify: `app.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: Task 3, 4의 모든 엔드포인트.
- Produces:
  - `GET /` → `static/index.html` (`FileResponse`). 경로는 `Path(__file__).parent / "static" / "index.html"`.
  - `main(argv: list[str] | None = None) -> None` — argparse: `--port` (기본 8585), `--no-browser`. `OllamaClient()`로 `create_app` 생성, `--no-browser`가 아니면 `threading.Timer(1.0, webbrowser.open, [url]).start()`, `uvicorn.run(app, host="127.0.0.1", port=port)`. `if __name__ == "__main__": main()`.

- [ ] **Step 1: 실패하는 테스트 추가**

```python
def test_root_serves_html(tmp_path):
    r = make(tmp_path).get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Ollama" in r.text
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_api.py::test_root_serves_html -q`
Expected: 404

- [ ] **Step 3: `static/index.html` 작성**

스펙 1장의 화면 구성을 그대로 구현한다. 다크 테마, 시스템 폰트, 카드 6개. JS 함수와 동작:

- `pollStatus()`: 2초마다 `GET /api/status`. 상단바(점 색: online이면 초록, 아니면 빨강 + "오프라인"), 환경변수 칩 3개(`OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_KEEP_ALIVE`; 값 없으면 "–"), GPU 카드(`gpu`가 null이면 "GPU 정보 없음"; 사용률 바, VRAM 바 `mem_used/mem_total` MiB → GB 소수 1자리, 온도, 전력 `power`가 null이면 "–"), 로드된 모델 카드(`loaded` 각 항목: name, `details.parameter_size`, `details.quantization_level`, `size_vram` GB, GPU 비율 `round(size_vram/size*100)`%, `context_length`, 언로드 버튼), 보유 모델 카드(`models` 각 항목: name, size GB, `details.family`; loaded에 있으면 "로드됨" 배지, 없으면 로드 버튼). 모델 select 옵션도 `models`로 갱신(현재 선택 유지).
- `pollLog()`: 5초마다 `GET /api/log?lines=100`. 체크박스 "자동갱신" 꺼지면 중단. 필터 입력값을 포함하는 줄만 표시. `lines`가 null이면 `"로그 파일을 찾을 수 없음: " + path`.
- `loadModel(name)` / `unloadModel(name)`: POST 후 실패하면 버튼 옆 `<span class="err">`에 `detail` 표시. 성공하면 즉시 `pollStatus()`.
- `sendPrompt()`: `fetch("/api/chat", {method:"POST", body: JSON})` → `res.body.getReader()`로 읽어 `\n` 기준 분리, 각 JSON의 `message.content`를 응답 영역에 누적. `error` 키가 있으면 빨간 글씨로 표시. `done: true` 청크에서 통계 줄 갱신: `로드 {load_duration/1e9 toFixed(2)}s │ 프롬프트 {prompt_eval_count}tok {prompt_eval_count/(prompt_eval_duration/1e9) toFixed(1)}tok/s │ 생성 {eval_count}tok {eval_count/(eval_duration/1e9) toFixed(1)}tok/s │ 총 {total_duration/1e9 toFixed(2)}s`. 전송 중에는 버튼 비활성.

- [ ] **Step 4: `app.py`에 `GET /` 라우트와 `main()` 추가**

- [ ] **Step 5: 통과 확인**

Run: `python -m pytest -q`
Expected: `32 passed`

- [ ] **Step 6: README 작성**

설치/실행 명령(스펙 6장), 옵션 2개, 환경변수 `OLLAMA_HOST`, `OLLAMA_LOG_PATH`, 테스트 명령.

- [ ] **Step 7: Commit**

```bash
git add static/index.html app.py README.md tests/test_api.py
git commit -m "feat: add dashboard UI, root route, and CLI entry"
```

---

### Task 6: 실제 Ollama 대상 수동 검증

**Files:** 없음 (검증만)

- [ ] **Step 1: 서버 실행**

Run: `.\.venv\Scripts\python app.py --no-browser`
Expected: `Uvicorn running on http://127.0.0.1:8585`

- [ ] **Step 2: API 확인**

Run (다른 터미널): `Invoke-RestMethod http://127.0.0.1:8585/api/status | ConvertTo-Json -Depth 3`
Expected: `ollama.online: true`, `env.OLLAMA_KV_CACHE_TYPE: q8_0`, `gpu.name`에 RTX 3080, `models`에 qwen3.5:9b.

- [ ] **Step 3: 브라우저 확인**

`http://127.0.0.1:8585` 열어 카드 6개가 채워지는지, 언로드 → `ollama ps`에서 사라지는지 → 로드 → 다시 나타나는지, 테스트 프롬프트 "안녕하세요"로 스트리밍과 통계 줄이 나오는지 확인. 문제가 있으면 해당 Task로 돌아가 수정하고 테스트 추가.

- [ ] **Step 4: 수정 사항이 있었다면 Commit**

```bash
git add -A
git commit -m "fix: adjustments from manual verification"
```
