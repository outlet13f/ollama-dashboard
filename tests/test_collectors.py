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
