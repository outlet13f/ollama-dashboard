# Ollama Dashboard 설계

작성일: 2026-10-11

## 목적

로컬 PC(Windows 11, RTX 3080 10GB)에서 돌아가는 Ollama와 GPU 상태를 브라우저 한 화면에서 확인하고, 모델을 로드/언로드하며, 테스트 프롬프트로 응답 속도를 측정한다. 본인 PC에서 혼자 쓰는 용도이며 인증이나 외부 공개는 없다.

## 범위

포함:
- 모니터링: Ollama 온라인 여부와 버전, 서버 시작 환경변수(FLASH_ATTENTION, KV_CACHE_TYPE, KEEP_ALIVE), GPU 사용률/VRAM/온도/전력, 로드된 모델과 VRAM 점유, 보유 모델 목록
- 테스트 프롬프트: 모델 선택 후 프롬프트 전송, 스트리밍 응답, 로드 시간과 prompt/eval 토큰 수 및 tokens/sec 표시
- 모델 로드(웜업)/언로드 버튼
- server.log 마지막 N줄 보기, 필터, 자동갱신 토글

제외:
- 모델 pull/delete
- 멀티턴 채팅 이력 저장
- 인증, HTTPS, 외부 바인딩
- 프론트 빌드 도구, 프레임워크

## 구조

```
ollama-dashboard/
  app.py            FastAPI 라우트. 수집기를 호출해 JSON/스트림으로 응답
  collectors.py     데이터 수집기: OllamaClient, read_gpu, tail_log, parse_env_from_log
  static/index.html 단일 페이지. CSS/JS 인라인, 바닐라 JS
  tests/
    test_collectors.py
    test_api.py
  requirements.txt  fastapi, uvicorn, httpx (+ pytest는 dev)
  README.md
```

실행: `python app.py` → `127.0.0.1:8585`에 바인딩하고 기본 브라우저를 연다. 포트는 `--port`, 브라우저 자동 열기는 `--no-browser`로 제어한다.

### collectors.py

| 함수/클래스 | 입력 | 출력 | 의존 |
|---|---|---|---|
| `OllamaClient(base_url)` | | | httpx.AsyncClient |
| `.version()` | | `str` 또는 `None`(오프라인) | GET /api/version |
| `.ps()` | | 로드된 모델 리스트 | GET /api/ps |
| `.tags()` | | 보유 모델 리스트 | GET /api/tags |
| `.load(model)` | model | 성공 여부 | POST /api/generate `{model, keep_alive: -1}` |
| `.unload(model)` | model | 성공 여부 | POST /api/generate `{model, keep_alive: 0}` |
| `.chat_stream(model, prompt)` | | NDJSON 라인 async iterator | POST /api/chat `stream: true` |
| `read_gpu()` | | `{name, util, mem_used, mem_total, temp, power, power_limit}` 또는 `None` | nvidia-smi 서브프로세스 |
| `parse_gpu_csv(text)` | nvidia-smi CSV 한 줄 | 위 dict | 순수 함수 |
| `tail_log(path, n)` | 경로, 줄 수 | 문자열 리스트. 파일 없으면 `None` | 파일 I/O |
| `parse_env_from_log(lines)` | 로그 줄들 | `{OLLAMA_FLASH_ATTENTION, OLLAMA_KV_CACHE_TYPE, OLLAMA_KEEP_ALIVE, ...}` | 순수 함수 |

nvidia-smi 호출: `nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits`. 타임아웃 3초.

로그 경로 기본값: `%LOCALAPPDATA%\Ollama\server.log`. 환경변수 `OLLAMA_LOG_PATH`로 덮어쓸 수 있다.

환경변수 파싱: server.log에서 `OLLAMA_FLASH_ATTENTION:` 같은 토큰이 들어 있는 **마지막** 줄을 찾아 `KEY:VALUE` 쌍을 추출한다. Ollama는 시작할 때 전체 설정을 한 줄로 기록하므로 마지막 줄이 현재 실행 중인 서버의 설정이다. PC의 환경변수를 직접 읽지 않는 이유는, 서버가 재시작 전 값으로 떠 있을 수 있기 때문이다.

### app.py 엔드포인트

| 메서드 | 경로 | 요청 | 응답 |
|---|---|---|---|
| GET | `/` | | static/index.html |
| GET | `/api/status` | | `{ollama: {online, version}, env: {...}, gpu: {...} \| null, loaded: [...], models: [...]}` |
| GET | `/api/log?lines=100` | lines 1~1000 | `{path, lines: [...] \| null}` |
| POST | `/api/load` | `{model}` | `{ok: true}` 또는 502 |
| POST | `/api/unload` | `{model}` | `{ok: true}` 또는 502 |
| POST | `/api/chat` | `{model, prompt}` | `application/x-ndjson` 스트림. Ollama 청크를 그대로 중계. `done: true` 청크에 `load_duration, prompt_eval_count, prompt_eval_duration, eval_count, eval_duration, total_duration` 포함 |

`/api/status`는 네 수집기를 `asyncio.gather`로 동시에 호출한다. 하나가 실패해도 나머지는 정상 반환한다.

### static/index.html

- 2초마다 `/api/status` 폴링. 5초마다 `/api/log` 폴링(토글로 끌 수 있음).
- 카드: 상단 상태바, GPU, 로드된 모델, 보유 모델, 테스트 프롬프트, 서버 로그.
- 보유 모델 목록에서 로드된 모델은 배지로 표시. 각 행에 로드/언로드 버튼.
- 테스트 프롬프트: `fetch` + `ReadableStream`으로 NDJSON을 줄 단위로 읽어 응답 영역에 누적. `done` 청크에서 통계 계산: `tok/s = count / (duration_ns / 1e9)`.
- 다크 테마, 시스템 폰트, 외부 CDN 없음.

## 오류 처리

- Ollama 오프라인: `ollama.online=false`, `loaded`/`models`는 빈 배열. 화면은 빨간 점과 "오프라인". 폴링은 계속한다.
- nvidia-smi 실패/부재/타임아웃: `gpu=null`. 화면은 "GPU 정보 없음".
- 로그 파일 없음: `lines=null`. 화면은 경로와 함께 "로그 파일을 찾을 수 없음".
- `/api/load`, `/api/unload` 실패: 502와 메시지. 화면은 버튼 옆에 오류 표시.
- `/api/chat` 중 Ollama 오류: 스트림에 `{error: "..."}` 청크를 보내고 종료. 화면은 응답 영역에 빨간 글씨로 표시.

## 테스트

- `test_collectors.py`: `parse_gpu_csv`(정상 줄, 빈 문자열, 필드 부족), `tail_log`(n보다 짧은 파일, 긴 파일, 없는 파일), `parse_env_from_log`(설정 줄 여러 개일 때 마지막 것 선택, 없을 때 빈 dict).
- `test_api.py`: FastAPI `TestClient`로 각 엔드포인트. Ollama는 `httpx.MockTransport`로 대체. `/api/status`에서 Ollama 오프라인 시 `online=false`이고 나머지 필드가 유지되는지, `/api/chat`이 NDJSON 청크를 그대로 통과시키는지 확인. `read_gpu`는 monkeypatch로 고정값 반환.
- 수동: 실제 Ollama 대상으로 실행해 qwen3.5:9b 언로드 → 로드 → 프롬프트 전송까지 확인.

## 보안

`127.0.0.1`에만 바인딩한다. CORS 설정은 두지 않는다(같은 출처에서만 호출).
