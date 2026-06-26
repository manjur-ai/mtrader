import builtins
from datetime import datetime


def printo(*args, count=None, **kwargs):
    builtins.print(*args, **kwargs)


def timenum(time_str):
    formats = ["%H:%M", "%H:%M:%S", "%I:%M %p"]
    for fmt in formats:
        try:
            time_obj = datetime.strptime(time_str, fmt).time()
            return time_obj.hour * 60 + time_obj.minute
        except ValueError:
            continue
    raise ValueError("Invalid time format. Supported formats: 'HH:MM', 'HH:MM:SS', 'H:MM AM/PM'.")
