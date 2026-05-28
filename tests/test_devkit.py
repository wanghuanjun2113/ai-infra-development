from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

import devkit


def test_config_loads():
    cfg = devkit.config(ROOT)
    assert cfg["current_framework"] == "xllm"
    assert "xllm" in devkit.enabled_frameworks(cfg)


def test_relative_path_resolution():
    assert devkit.path(ROOT, "code") == ROOT / "code"
