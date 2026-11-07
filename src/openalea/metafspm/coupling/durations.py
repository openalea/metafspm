"""
Times and durations, given with their units and kept in seconds, the scene's one internal unit::

    duration("12 h")                    43200.0     (terms "<number> <unit>", units s, min, h, d, summed: "1 d 12 h")
    duration(3600)                      3600.0      (a number: seconds)
    duration("h")                       3600.0      (a bare unit: one of it, e.g. a forcing's time unit)
    start_date("2026-05-10 00:00")      datetime(2026, 5, 10, 0, 0)     (ISO 8601, UTC; a datetime as it is)
    instant("36 h")                     129600.0    (a scene time: a number, a duration since time 0, or a date)
    instant("2026-05-12 06:00", start)  194400.0    (a date: its time since the scene's start)
    day_and_hour(start, 48600.)         (130.0, 13.5)   (the calendar a light model's sun reads)

Every argument of a time or a duration (the Scene's time_step, start and events, the recorder's periods, a
component's every, a forcing's time unit, a pulse's times, the plants' emergence times) is read by these.
"""
import math
import re
from datetime import datetime, timedelta, timezone

UNITS = {"s": 1., "min": 60., "h": 3600., "d": 86400.}
_TERM = re.compile(r"\s*(?P<number>[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)?\s*(?P<unit>min|s|h|d)(?![A-Za-z])")
_DATE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) or (
        hasattr(value, "dtype") and getattr(value, "ndim", 1) == 0 and value.dtype.kind in "iuf")


def duration(value, name: str = "duration") -> float:
    """*value* in seconds: a number (seconds), or a text of terms "<number> <unit>" summed (units s, min, h, d:
    "12 h", "1 d 12 h", "30 min"; a bare unit is one of it). Negative durations raise. *name*: the argument's, for
    the messages."""
    if _is_number(value):
        seconds = float(value)
    elif isinstance(value, str):
        text, position, seconds = value.strip(), 0, 0.
        if not text:
            raise ValueError(f"{name}: an empty duration (give e.g. '1 h', '1 d 12 h', or a number of seconds)")
        while position < len(text):
            term = _TERM.match(text, position)
            if term is None:
                raise ValueError(f"{name}: '{value}' is not a duration: terms '<number> <unit>', units "
                                 f"{', '.join(UNITS)} (e.g. '12 h', '1 d 12 h'), or a number of seconds")
            number = term.group("number")
            seconds += (1. if number is None else float(number)) * UNITS[term.group("unit")]
            position = term.end()
            while position < len(text) and text[position] in " ,+":
                position += 1
    elif isinstance(value, timedelta):
        seconds = value.total_seconds()
    else:
        raise TypeError(f"{name}: a duration is a text with its unit ('12 h', '1 d') or a number of seconds, got "
                        f"{value!r}")
    if not math.isfinite(seconds) or seconds < 0.:
        raise ValueError(f"{name}: {value!r} is not a duration (a finite, non-negative time)")
    return seconds


def positive_duration(value, name: str = "duration") -> float:
    """duration(*value*), which must be more than 0 (a time step, a period, an interval)."""
    seconds = duration(value, name)
    if not seconds > 0.:
        raise ValueError(f"{name}: {value!r} must be longer than 0")
    return seconds


def start_date(value, name: str = "start") -> datetime:
    """The calendar date of a scene's time 0: an ISO 8601 text ("2026-05-10 00:00", "2026-05-10T06:30", "2026-05-10")
    or a datetime; read as UTC (a time zone given is converted to UTC). The light models' sun reads it."""
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.strip())
        except ValueError:
            raise ValueError(f"{name}: '{value}' is not a date: give it as '2026-05-10 00:00' (ISO 8601, UTC)") \
                from None
    else:
        raise TypeError(f"{name}: a date, '2026-05-10 00:00' (ISO 8601, UTC) or a datetime, got {value!r}")
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def is_date(value) -> bool:
    """Whether *value* is a date (a datetime, or a text beginning with YYYY-MM-DD) rather than a duration."""
    return isinstance(value, datetime) or (isinstance(value, str) and bool(_DATE.match(value)))


def instant(value, start: datetime = None, name: str = "time") -> float:
    """A scene time (s since time 0): a number (seconds), a duration since time 0 ("36 h"), or a date (its time
    since *start*, the scene's start_date, which a date needs)."""
    if is_date(value):
        if start is None:
            raise ValueError(f"{name}: the date {value!r} needs the Scene's start= (its time 0's date)")
        return (start_date(value, name) - start).total_seconds()
    return duration(value, name)


def day_and_hour(start: datetime, time: float) -> tuple:
    """(day of year, hour of the day), floats, at scene time *time* (s) from the date *start* of time 0: what a light
    model's sun position reads (Component.calendar)."""
    midnight = datetime(start.year, start.month, start.day)
    of_day = (start - midnight).total_seconds() + float(time)
    days = math.floor(of_day / 86400.)
    day = (midnight + timedelta(days=days)).timetuple().tm_yday
    return float(day), (of_day - days * 86400.) / 3600.


def label(seconds: float) -> str:
    """The unit label of a time unit of *seconds* ("s", "min", "h", "d"), for files that name their time unit."""
    for unit, value in UNITS.items():
        if float(seconds) == value:
            return unit
    raise ValueError(f"a time unit of {seconds} s has no label: one of {UNITS}")
