from __future__ import annotations
import builtins
from datetime import datetime
from typing import Any


def printo(*args: Any, count: int | None = None, **kwargs: Any) -> None:
    """Print wrapper that aliases the built-in print, with an optional count parameter for suppressing output."""
    builtins.print(*args, **kwargs)


def timenum(time_str: str) -> int:
    """Convert a time string to minutes since midnight. Supports HH:MM, HH:MM:SS, and H:MM AM/PM formats."""
    formats = ["%H:%M", "%H:%M:%S", "%I:%M %p"]
    for fmt in formats:
        try:
            time_obj = datetime.strptime(time_str, fmt).time()
            return time_obj.hour * 60 + time_obj.minute
        except ValueError:
            continue
    raise ValueError("Invalid time format. Supported formats: 'HH:MM', 'HH:MM:SS', 'H:MM AM/PM'.")
