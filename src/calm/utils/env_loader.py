"""
Mô-đun nạp biến môi trường từ file .env.

Đảm bảo OPENAI_API_KEY hoặc OPENROUTER_API_KEY được nạp trước khi tạo
ChatOpenRouter/ChatOpenAI. Hỗ trợ .env ở thư mục gốc dự án hoặc thư mục hiện tại.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_env(env_file: str | Path | None = None) -> bool:
    """
    Nạp biến môi trường từ file .env nếu có.

    Tìm file .env theo thứ tự: tham số env_file, CALM_ROOT/.env,
    thư mục hiện tại, thư mục cha (để chạy từ examples/ hoặc tests/).

    Tham số:
        env_file: Đường dẫn tới file .env. Nếu None thì tự tìm.

    Trả về:
        True nếu đã nạp được file .env, False nếu không tìm thấy hoặc
        python-dotenv chưa cài.

    Ví dụ:
        load_env()  # Tự tìm .env
        load_env(".env")  # Dùng file .env tại CWD
    """
    try:
        from dotenv import load_dotenv as _load_dotenv
    except ImportError:
        return False

    if env_file is not None:
        path = Path(env_file)
        if path.is_file():
            _load_dotenv(path, override=False)
            return True
        return False

    # Thư mục gốc dự án (chứa pyproject.toml hoặc calm/)
    cwd = Path.cwd()
    for candidate in [cwd, cwd.parent]:
        env_path = candidate / ".env"
        if env_path.is_file():
            _load_dotenv(env_path, override=False)
            return True
    if "CALM_ROOT" in os.environ:
        env_path = Path(os.environ["CALM_ROOT"]) / ".env"
        if env_path.is_file():
            _load_dotenv(env_path, override=False)
            return True
    return False


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Lấy giá trị biến môi trường sau khi đã gọi load_env().

    Tham số:
        key: Tên biến (ví dụ OPENAI_API_KEY, OPENROUTER_API_KEY).
        default: Giá trị mặc định nếu biến không tồn tại.

    Trả về:
        Giá trị chuỗi của biến hoặc default.
    """
    return os.environ.get(key, default)
