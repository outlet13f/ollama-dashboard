"""CPU / RAM / disk collector and its exposure in /api/status."""
import httpx
from fastapi.testclient import TestClient

from app import create_app
from collectors import OllamaClient, read_system, models_dir


def test_read_system_shape_and_ranges():
    s = read_system()
    assert set(s) >= {"cpu_name", "cpu_percent", "cpu_count", "ram_used", "ram_total", "disk_path", "disk_free", "disk_total"}
    assert 0 <= s["cpu_percent"] <= 100 and s["cpu_count"] >= 1
    assert 0 < s["ram_used"] <= s["ram_total"]
    assert 0 <= s["disk_free"] <= s["disk_total"]
    assert isinstance(s["cpu_name"], str) and s["cpu_name"]


def test_models_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    assert models_dir() == tmp_path


def test_status_includes_system(tmp_path):
    client = OllamaClient("http://ollama.test", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    fake = {"cpu_name": "X", "cpu_percent": 5.0, "cpu_count": 8, "ram_used": 1, "ram_total": 2,
            "disk_path": "C:/m", "disk_free": 3, "disk_total": 4}
    app = create_app(client, gpu_reader=lambda: None, system_reader=lambda: fake, log_path=tmp_path / "s.log")
    d = TestClient(app, base_url="http://127.0.0.1").get("/api/status").json()
    assert d["system"] == fake


def test_status_system_failure_gives_null(tmp_path):
    client = OllamaClient("http://ollama.test",
                          transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"version": "1", "models": []})))

    def boom():
        raise RuntimeError("psutil broke")
    app = create_app(client, gpu_reader=lambda: None, system_reader=boom, log_path=tmp_path / "s.log")
    d = TestClient(app, base_url="http://127.0.0.1").get("/api/status").json()
    assert d["system"] is None and d["ollama"]["online"] is True
