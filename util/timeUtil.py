# time_util.py
# -*- coding: utf-8 -*-
# Canonical time helpers: everything is normalized to Asia/Taipei, no naive datetimes.
# Store as epoch seconds (ints). Convert to/from Taipei only through these helpers.

from __future__ import annotations

from typing import Optional, Union
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

TAIPEI = ZoneInfo("Asia/Taipei")
UTC = ZoneInfo("UTC")

EpochLike = Union[int, float, str, datetime, pd.Timestamp, None]

def _normalize_epoch_seconds(x: Union[int, float]) -> int:
    """Heuristic: if value looks like milliseconds (> 1e12), divide by 1000."""
    return int(x / 1000) if x > 10**12 else int(x)

def to_taipei_ts(value: EpochLike) -> Optional[pd.Timestamp]:
    """
    Convert any input to a tz-aware pandas.Timestamp in Asia/Taipei.
    - Numbers: interpret as epoch seconds (or ms), i.e., UTC instants → convert to Taipei.
    - Strings: parse as UTC (unless offset provided), then convert to Taipei.
    - Naive datetime/Timestamp: treat as Taipei wall time.
    """
    if value is None:
        return None

    # Pandas Timestamp
    if isinstance(value, pd.Timestamp):
        return value.tz_localize(TAIPEI) if value.tzinfo is None else value.tz_convert(TAIPEI)

    # Python datetime
    if isinstance(value, datetime):
        return pd.Timestamp(value, tz=TAIPEI) if value.tzinfo is None else pd.Timestamp(value).tz_convert(TAIPEI)

    # Numeric epoch
    if isinstance(value, (int, float)):
        sec = _normalize_epoch_seconds(value)
        return pd.to_datetime(sec, unit="s", utc=True).tz_convert(TAIPEI)

    # Strings / others: parse as UTC instant, then convert
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if ts is pd.NaT:
        return None
    return ts.tz_convert(TAIPEI)

def to_epoch_seconds(value: EpochLike) -> Optional[int]:
    """
    Convert any input to integer epoch seconds (UTC), using Taipei as the reference zone
    for naive values. Always returns the same absolute instant in seconds.
    """
    ts = to_taipei_ts(value)
    if ts is None:
        return None
    return int(ts.tz_convert(UTC).timestamp())

def now_taipei_ts() -> pd.Timestamp:
    """Current Taipei time as tz-aware Timestamp."""
    return pd.Timestamp.now(tz=TAIPEI)

def now_epoch() -> int:
    """Current time as epoch seconds, normalized via Taipei zone."""
    return to_epoch_seconds(now_taipei_ts())  # type: ignore[arg-type]

def days_ago_epoch(days: int) -> int:
    """Epoch seconds for (now in Taipei) - days."""
    return to_epoch_seconds(now_taipei_ts() - pd.Timedelta(days=days))  # type: ignore[arg-type]

def clamp_future_epoch(epoch_sec: int, grace_secs: int = 86400) -> int:
    """
    If epoch is too far in the future (beyond grace_secs), clamp to now.
    Returns the original if within the grace window.
    """
    if epoch_sec is None:
        return epoch_sec
    now_sec = now_epoch()
    return now_sec if epoch_sec > now_sec + grace_secs else epoch_sec

def fmt_taipei(epoch_sec: int) -> str:
    """Human-readable helper: epoch → 'YYYY-MM-DD HH:MM:SS+08:00' in Taipei."""
    if epoch_sec is None:
        return "None"
    return pd.to_datetime(int(epoch_sec), unit="s", utc=True).tz_convert(TAIPEI).strftime("%Y-%m-%d %H:%M:%S%z")