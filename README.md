# Ollama Dashboard

로컬 Ollama 서버와 NVIDIA GPU 상태를 브라우저 한 화면에서 확인하는 대시보드.

- Ollama 온라인 여부, 버전, 서버 시작 환경변수 (FLASH_ATTENTION / KV_CACHE_TYPE / KEEP_ALIVE)
- GPU 사용률, VRAM, 온도, 전력 (nvidia-smi)
- CPU 사용률, RAM, 모델 저장 디스크 여유 공간 (psutil)
- 로드된 모델과 VRAM 점유, 로드/언로드 버튼
- 보유 모델 관리: 받기(진행률·속도·취소), 상세 정보(컨텍스트, capabilities, 파라미터, 라이선스, 템플릿), 삭제
- 테스트 프롬프트: 스트리밍 응답(thinking 포함)과 tokens/sec, 로드 시간
- server.log 마지막 100줄, 요청 로그 숨기기, 필터, 자동갱신

## 설치 및 실행

```powershell
cd C:\Users\casi0\Workspace\ollama-dashboard
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

`http://127.0.0.1:8585`가 브라우저에서 자동으로 열립니다. `127.0.0.1`에만 바인딩되므로 외부에서는 접근할 수 없습니다.

| 옵션 | 설명 |
|---|---|
| `--port N` | 포트 변경 (기본 8585) |
| `--no-browser` | 시작할 때 브라우저를 열지 않음 |

| 환경변수 | 설명 |
|---|---|
| `OLLAMA_HOST` | Ollama 주소 (기본 `http://127.0.0.1:11434`) |
| `OLLAMA_LOG_PATH` | server.log 경로 (기본 `%LOCALAPPDATA%\Ollama\server.log`) |

## 테스트

```powershell
.\.venv\Scripts\pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q
```
