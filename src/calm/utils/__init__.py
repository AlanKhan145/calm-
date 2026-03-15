"""
Mô-đun tiện ích CALM.

Chứa env_loader để nạp biến môi trường từ file .env một cách an toàn.
"""

from calm.utils.env_loader import load_env, get_env

__all__ = ["load_env", "get_env"]
