"""Ollama dashboard: FastAPI routes and CLI entry point."""
from __future__ import annotations

import argparse
import asyncio
import threading
import webbrowser
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from collectors import EnvTracker, OllamaClient, OllamaError, default_log_path, read_gpu, tail_log

INDEX_HTML = Path(__file__).parent / "static" / "index.html"
DEFAULT_PORT = 8585


class ModelReq(BaseModel):
    model: str


class ChatReq(BaseModel):
    model: str
    prompt: str


def create_app(client: OllamaClient,
               gpu_reader: Callable[[], dict | None] = read_gpu,
               log_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Ollama Dashboard")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    log_file = log_path or default_log_path()
    env_tracker = EnvTracker(log_file)

    async def _safe(coro, default):
        try:
            return await coro
        except Exception:
            return default

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(INDEX_HTML, media_type="text/html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    @app.get("/api/status")
    async def status():
        version, loaded, models, gpu, env = await asyncio.gather(
            _safe(client.version(), None),
            _safe(client.ps(), []),
            _safe(client.tags(), []),
            _safe(asyncio.to_thread(gpu_reader), None),
            _safe(asyncio.to_thread(env_tracker.read), {}),
        )
        return {
            "ollama": {"online": version is not None, "version": version},
            "env": env,
            "gpu": gpu,
            "loaded": loaded,
            "models": models,
        }

    @app.get("/api/log")
    async def log(lines: int = Query(100, ge=1, le=1000),
                  exclude: str | None = Query(None, max_length=100)):
        result = await asyncio.to_thread(tail_log, log_file, lines, exclude or None)
        return {"path": str(log_file), "lines": result}

    @app.post("/api/load")
    async def load(req: ModelReq):
        try:
            await client.load(req.model)
        except OllamaError as e:
            raise HTTPException(status_code=502, detail=str(e))
        return {"ok": True}

    @app.post("/api/unload")
    async def unload(req: ModelReq):
        try:
            await client.unload(req.model)
        except OllamaError as e:
            raise HTTPException(status_code=502, detail=str(e))
        return {"ok": True}

    @app.post("/api/chat")
    async def chat(req: ChatReq):
        async def gen():
            async for line in client.chat_stream(req.model, req.prompt):
                yield line + "\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Ollama / GPU status dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser on start")
    args = parser.parse_args(argv)

    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(create_app(OllamaClient()), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
