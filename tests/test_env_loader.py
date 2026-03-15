"""
Kiểm tra nạp biến môi trường — load_env, get_env.

Đảm bảo load_env không gây lỗi khi không có .env và get_env trả về default.
"""

import os
import sys
from pathlib import Path

import pytest

# Đảm bảo import từ src
src = Path(__file__).resolve().parent.parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from calm.utils.env_loader import get_env, load_env


def test_load_env_returns_bool() -> None:
    """load_env() phải trả về bool (True nếu đã nạp file .env)."""
    result = load_env()
    assert isinstance(result, bool)


def test_load_env_nonexistent_file_returns_false() -> None:
    """load_env với file không tồn tại trả về False."""
    result = load_env(Path("/nonexistent/.env"))
    assert result is False


def test_get_env_missing_key_returns_default() -> None:
    """get_env với key không tồn tại trả về default."""
    assert get_env("CALM_TEST_NONEXISTENT_KEY", "default") == "default"
    assert get_env("CALM_TEST_NONEXISTENT_KEY") is None


def test_get_env_existing_key() -> None:
    """get_env với key có trong env trả về giá trị."""
    os.environ["CALM_TEST_TMP"] = "value123"
    try:
        assert get_env("CALM_TEST_TMP") == "value123"
        assert get_env("CALM_TEST_TMP", "other") == "value123"
    finally:
        os.environ.pop("CALM_TEST_TMP", None)
