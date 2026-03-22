"""
Time utils — chuẩn hóa time_range, mặc định ngày hôm nay khi không chỉ định.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any


def resolve_time_range(
    time_range: dict[str, Any] | None = None,
    default_today: bool = True,
) -> dict[str, str]:
    """
    Chuẩn hóa time_range thành {start, end} ISO format.

    - date: "2024-03-21" → start=end=2024-03-21
    - start, end có sẵn → giữ nguyên
    - today / hôm nay / rỗng + default_today → ngày hôm nay
    """
    tr = time_range or {}
    today = date.today().isoformat()

    # Đã có start, end hợp lệ
    start = tr.get("start", "").strip()
    end = tr.get("end", "").strip()
    if start and end:
        return {"start": start, "end": end}
    if start:
        return {"start": start, "end": start}

    # Single date
    d = tr.get("date", "").strip() or tr.get("day", "").strip()
    if d:
        return {"start": d, "end": d}

    # Default: hôm nay
    if default_today:
        return {"start": today, "end": today}
    return {"start": "", "end": ""}
