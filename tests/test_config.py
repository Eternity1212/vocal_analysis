from __future__ import annotations

from src.utils.config import load_yaml_config


def test_load_yaml_config_returns_dict() -> None:
    config = load_yaml_config("configs/mfcc.yaml")
    assert "mfcc" in config
