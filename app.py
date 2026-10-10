"""Ollama dashboard: FastAPI routes and CLI entry point."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from collectors import OllamaClient, OllamaError, default_log_path, parse_env_from_log, read_gpu, tail_log

ENV_SCAN_LINES = 2000  # how far back to look for Ollama's startup config line


class ModelReq(BaseModel):
    model: str


class ChatReq(BaseModel):
    model: str
    prompt: str


def create_app(client: OllamaClient,
               gpu_reader: Callable[[], dict | None] = read_gpu,
               log_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Ollama Dashboard")
    log_file = log_path or default_log_path()

    async def _safe(coro, default):
        try:
            return await coro
        except Exception:
            return default

    @app.get("/api/status")
    async def status():
        version, loaded, models, gpu, log_lines = await asyncio.gather(
            _safe(client.version(), None),
            _safe(client.ps(), []),
            _safe(client.tags(), []),
            _safe(asyncio.to_thread(gpu_reader), None),
            _safe(asyncio.to_thread(tail_log, log_file, ENV_SCAN_LINES), None),
        )
        return {
            "ollama": {"online": version is not None, "version": version},
            "env": parse_env_from_log(log_lines or []),
            "gpu": gpu,
            "loaded": loaded,
            "models": models,
        }

    @app.get("/api/log")
    async def log(lines: int = Query(100, ge=1, le=1000)):
        result = await asyncio.to_thread(tail_log, log_file, lines)
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
