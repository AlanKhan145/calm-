"""
Time utils — chuẩn hóa time_range thành ISO + semantics rõ ràng.

Hỗ trợ:
- hôm nay / hôm qua / ngày mai
- 7 ngày tới / 30 ngày tới / next 7 days
- tuần này / tuần sau / this week / next week
- tháng này / tháng sau / this month / next month
- ngày tuyệt đối: 2026-03-25, 25/03/2026, 2026/03/25
- khoảng ngày tuyệt đối: 2026-03-25 đến 2026-04-01
- khoảng năm: 2020-2025

Output chuẩn:
{
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD",
    "granularity": "day|week|month|year",
    "horizon_days": int,
    "is_forecast_window": bool,
}
"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Tuple, Union


TimeRangeInput = Union[None, str, Dict[str, Any]]


def resolve_time_range(
    time_range: TimeRangeInput = None,
    default_today: bool = True,
    reference_date: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Chuẩn hóa time_range thành dict ISO + semantics rõ ràng.

    Hỗ trợ input:
    - None
    - str tự nhiên: "7 ngày tới", "tuần sau", "2020-2025"
    - dict:
        {
            "start": "...",
            "end": "...",
        }
        {
            "date": "...",
        }
        {
            "text": "7 ngày tới"
        }

    horizon_days:
    - độ dài cửa sổ thời gian tính theo số ngày, bao gồm cả start và end

    is_forecast_window:
    - True khi cửa sổ là tương lai hoặc do cụm forecast-like sinh ra
    """
    ref = reference_date or date.today()

    # 1) None -> default
    if time_range is None:
        if default_today:
            return _build_result(ref, ref, granularity="day", reference_date=ref, force_forecast=False)
        return _build_empty_result()

    # 2) str -> parse text
    if isinstance(time_range, str):
        parsed = parse_time_text(time_range, reference_date=ref)
        if parsed:
            return parsed
        if default_today:
            return _build_result(ref, ref, granularity="day", reference_date=ref, force_forecast=False)
        return _build_empty_result()

    # 3) dict -> direct fields or text fields
    if isinstance(time_range, dict):
        direct = _resolve_time_range_dict(time_range, reference_date=ref)
        if direct:
            return direct

        raw_text = (
            _get_first_non_empty(
                time_range,
                ["text", "raw_text", "query_time", "natural_language", "value"],
            )
            or ""
        ).strip()

        if raw_text:
            parsed = parse_time_text(raw_text, reference_date=ref)
            if parsed:
                return parsed

        if default_today:
            return _build_result(ref, ref, granularity="day", reference_date=ref, force_forecast=False)
        return _build_empty_result()

    # 4) fallback
    if default_today:
        return _build_result(ref, ref, granularity="day", reference_date=ref, force_forecast=False)
    return _build_empty_result()


def parse_time_text(
    text: str,
    reference_date: Optional[date] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parse text thời gian tự nhiên thành time_range chuẩn.
    """
    if not text or not str(text).strip():
        return None

    ref = reference_date or date.today()
    raw = str(text).strip()
    norm = _normalize_time_text(raw)

    # ─────────────────────────────────────────
    # 1) Relative day
    # ─────────────────────────────────────────
    if norm in {"hôm nay", "hom nay", "today"}:
        return _build_result(ref, ref, granularity="day", reference_date=ref, force_forecast=False)

    if norm in {"hôm qua", "hom qua", "yesterday"}:
        d = ref - timedelta(days=1)
        return _build_result(d, d, granularity="day", reference_date=ref, force_forecast=False)

    if norm in {"ngày mai", "ngay mai", "tomorrow"}:
        d = ref + timedelta(days=1)
        return _build_result(d, d, granularity="day", reference_date=ref, force_forecast=True)

    # ─────────────────────────────────────────
    # 2) Next N days / N ngày tới
    # ─────────────────────────────────────────
    match = re.search(r"\b(\d{1,3})\s*(ngày tới|ngay toi|days?|day)\b", norm)
    if not match:
        match = re.search(r"\bnext\s+(\d{1,3})\s+days?\b", norm)
    if match:
        n_days = int(match.group(1))
        if n_days <= 0:
            n_days = 1
        start = ref
        end = ref + timedelta(days=n_days - 1)
        return _build_result(start, end, granularity="day", reference_date=ref, force_forecast=True)

    # ─────────────────────────────────────────
    # 3) This week / next week
    # ─────────────────────────────────────────
    if norm in {"tuần này", "tuan nay", "this week"}:
        start, end = _week_bounds(ref)
        return _build_result(start, end, granularity="week", reference_date=ref, force_forecast=False)

    if norm in {"tuần sau", "tuan sau", "next week"}:
        next_week_ref = ref + timedelta(days=7)
        start, end = _week_bounds(next_week_ref)
        return _build_result(start, end, granularity="week", reference_date=ref, force_forecast=True)

    # ─────────────────────────────────────────
    # 4) This month / next month
    # ─────────────────────────────────────────
    if norm in {"tháng này", "thang nay", "this month"}:
        start, end = _month_bounds(ref.year, ref.month)
        return _build_result(start, end, granularity="month", reference_date=ref, force_forecast=False)

    if norm in {"tháng sau", "thang sau", "next month"}:
        year, month = _shift_month(ref.year, ref.month, delta=1)
        start, end = _month_bounds(year, month)
        return _build_result(start, end, granularity="month", reference_date=ref, force_forecast=True)

    # ─────────────────────────────────────────
    # 5) Absolute date range
    #    Ví dụ:
    #    2026-03-25 đến 2026-04-01
    #    25/03/2026 - 01/04/2026
    #    from 2026-03-25 to 2026-04-01
    # ─────────────────────────────────────────
    date_range = _extract_absolute_date_range(norm)
    if date_range:
        start, end = date_range
        return _build_result(start, end, granularity="day", reference_date=ref, force_forecast=None)

    # ─────────────────────────────────────────
    # 6) Year range
    #    Ví dụ: 2020-2025, 2020 đến 2025
    # ─────────────────────────────────────────
    year_range = _extract_year_range(norm)
    if year_range:
        start_year, end_year = year_range
        start = date(start_year, 1, 1)
        end = date(end_year, 12, 31)
        return _build_result(start, end, granularity="year", reference_date=ref, force_forecast=None)

    # ─────────────────────────────────────────
    # 7) Single absolute date
    # ─────────────────────────────────────────
    single_date = _parse_date_string(norm)
    if single_date:
        return _build_result(single_date, single_date, granularity="day", reference_date=ref, force_forecast=None)

    # ─────────────────────────────────────────
    # 8) Single year
    # ─────────────────────────────────────────
    year_match = re.fullmatch(r"(19|20)\d{2}", norm)
    if year_match:
        year = int(norm)
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        return _build_result(start, end, granularity="year", reference_date=ref, force_forecast=None)

    return None


# ─────────────────────────────────────────
# Dict resolver
# ─────────────────────────────────────────

def _resolve_time_range_dict(
    tr: Dict[str, Any],
    reference_date: date,
) -> Optional[Dict[str, Any]]:
    """
    Hỗ trợ input dict cũ + input dict mới.
    """
    # start / end
    start_raw = str(tr.get("start", "") or tr.get("start_date", "")).strip()
    end_raw = str(tr.get("end", "") or tr.get("end_date", "")).strip()

    if start_raw or end_raw:
        start = _parse_date_string(start_raw) if start_raw else None
        end = _parse_date_string(end_raw) if end_raw else None

        if start and end:
            granularity = _infer_granularity_from_range(start, end, tr.get("granularity"))
            return _build_result(start, end, granularity=granularity, reference_date=reference_date, force_forecast=None)
        if start:
            granularity = _normalize_granularity(tr.get("granularity")) or "day"
            return _build_result(start, start, granularity=granularity, reference_date=reference_date, force_forecast=None)
        if end:
            granularity = _normalize_granularity(tr.get("granularity")) or "day"
            return _build_result(end, end, granularity=granularity, reference_date=reference_date, force_forecast=None)

    # single day/date
    single_raw = str(tr.get("date", "") or tr.get("day", "")).strip()
    if single_raw:
        single = _parse_date_string(single_raw)
        if single:
            granularity = _normalize_granularity(tr.get("granularity")) or "day"
            return _build_result(single, single, granularity=granularity, reference_date=reference_date, force_forecast=None)

    # year range dict
    year_start = tr.get("year_start")
    year_end = tr.get("year_end")
    if year_start and year_end:
        try:
            ys = int(year_start)
            ye = int(year_end)
            start = date(min(ys, ye), 1, 1)
            end = date(max(ys, ye), 12, 31)
            return _build_result(start, end, granularity="year", reference_date=reference_date, force_forecast=None)
        except Exception:
            pass

    # already has semantics? normalize lightly if possible
    if tr.get("granularity") and start_raw and end_raw:
        start = _parse_date_string(start_raw)
        end = _parse_date_string(end_raw)
        if start and end:
            return _build_result(
                start,
                end,
                granularity=_normalize_granularity(tr.get("granularity")) or "day",
                reference_date=reference_date,
                force_forecast=tr.get("is_forecast_window"),
            )

    return None


# ─────────────────────────────────────────
# Extractors / parsers
# ─────────────────────────────────────────

def _extract_absolute_date_range(text: str) -> Optional[Tuple[date, date]]:
    """
    Tách khoảng ngày tuyệt đối.
    """
    patterns = [
        r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s*(?:-|–|—|to|đến)\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})\s*(?:-|–|—|to|đến)\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})",
        r"between\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\s+and\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})",
        r"from\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\s+to\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        start = _parse_date_string(match.group(1))
        end = _parse_date_string(match.group(2))
        if start and end:
            if start <= end:
                return start, end
            return end, start
    return None


def _extract_year_range(text: str) -> Optional[Tuple[int, int]]:
    """
    Tách khoảng năm như 2020-2025.
    """
    match = re.search(r"\b((?:19|20)\d{2})\s*(?:-|–|—|to|đến)\s*((?:19|20)\d{2})\b", text)
    if not match:
        return None

    y1 = int(match.group(1))
    y2 = int(match.group(2))
    if y1 <= y2:
        return y1, y2
    return y2, y1


def _parse_date_string(value: str) -> Optional[date]:
    """
    Hỗ trợ:
    - YYYY-MM-DD
    - YYYY/MM/DD
    - DD/MM/YYYY
    - DD-MM-YYYY
    """
    if not value:
        return None

    text = str(value).strip()
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────
# Builders
# ─────────────────────────────────────────

def _build_result(
    start: date,
    end: date,
    granularity: str,
    reference_date: date,
    force_forecast: Optional[bool],
) -> Dict[str, Any]:
    """
    Chuẩn hóa output cuối cùng.
    """
    if end < start:
        start, end = end, start

    if force_forecast is None:
        is_forecast_window = start > reference_date
    else:
        is_forecast_window = bool(force_forecast)

    horizon_days = (end - start).days + 1

    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "granularity": _normalize_granularity(granularity) or "day",
        "horizon_days": horizon_days,
        "is_forecast_window": is_forecast_window,
    }


def _build_empty_result() -> Dict[str, Any]:
    return {
        "start": "",
        "end": "",
        "granularity": "day",
        "horizon_days": 0,
        "is_forecast_window": False,
    }


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def _week_bounds(d: date) -> Tuple[date, date]:
    start = d - timedelta(days=d.weekday())  # Monday
    end = start + timedelta(days=6)
    return start, end


def _month_bounds(year: int, month: int) -> Tuple[date, date]:
    last_day = monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last_day)


def _shift_month(year: int, month: int, delta: int) -> Tuple[int, int]:
    total = (year * 12 + (month - 1)) + delta
    new_year = total // 12
    new_month = (total % 12) + 1
    return new_year, new_month


def _normalize_time_text(text: str) -> str:
    out = str(text or "").strip().lower()
    out = re.sub(r"\s+", " ", out)
    return out


def _normalize_granularity(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    mapping = {
        "day": "day",
        "date": "day",
        "daily": "day",
        "week": "week",
        "weekly": "week",
        "month": "month",
        "monthly": "month",
        "year": "year",
        "yearly": "year",
        "annual": "year",
    }
    return mapping.get(text)


def _infer_granularity_from_range(start: date, end: date, explicit: Any = None) -> str:
    explicit_norm = _normalize_granularity(explicit)
    if explicit_norm:
        return explicit_norm

    if start == end:
        return "day"

    # full year
    if start.month == 1 and start.day == 1 and end.month == 12 and end.day == 31:
        if start.year != end.year or (end - start).days >= 364:
            return "year"

    # full month
    last_day = monthrange(start.year, start.month)[1]
    if (
        start.day == 1
        and start.year == end.year
        and start.month == end.month
        and end.day == last_day
    ):
        return "month"

    # exact week window
    if start.weekday() == 0 and (end - start).days == 6:
        return "week"

    return "day"


def _get_first_non_empty(data: Dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""