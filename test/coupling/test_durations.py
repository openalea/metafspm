"""Times and durations given with their units (coupling.durations), kept in seconds."""
from datetime import datetime

import pytest

from openalea.metafspm.coupling.durations import (day_and_hour, duration, instant, label, positive_duration,
                                                  start_date)


def test_a_duration_is_read_with_its_unit_or_in_seconds():
    assert duration("12 h") == 43200. and duration("1 d 12 h") == 129600. and duration("30 min") == 1800.
    assert duration("90 s") == 90. and duration("1.5 h") == 5400. and duration("h") == 3600.
    assert duration("2d") == 172800. and duration("1 d, 6 h") == 108000. and duration(3600) == 3600.
    assert duration(0.) == 0. and label(86400.) == "d"
    for wrong in ("12", "1 week", "", "h h x", "-1 h"):
        with pytest.raises(ValueError):
            duration(wrong)
    with pytest.raises(ValueError):
        duration(-1.)
    with pytest.raises(ValueError):
        positive_duration("0 h")
    with pytest.raises(TypeError):
        duration([1.])


def test_a_start_is_a_date_and_an_instant_a_scene_time():
    start = start_date("2026-05-10 06:00")
    assert start == datetime(2026, 5, 10, 6) and start_date("2026-05-10") == datetime(2026, 5, 10)
    assert start_date("2026-05-10T08:00+02:00") == start                 # converted to UTC
    assert instant("36 h") == 129600. and instant(7.) == 7.
    assert instant("2026-05-11 06:00", start) == 86400.
    with pytest.raises(ValueError, match="start"):
        instant("2026-05-11 06:00")
    with pytest.raises(ValueError, match="ISO"):
        start_date("10 May 2026")


def test_the_calendar_is_the_day_of_year_and_the_hour():
    """day_and_hour gives the former (day of year, hour) calendar: day 130 of 2026 is May 10."""
    start = start_date("2026-05-10 06:00")
    assert day_and_hour(start, 0.) == (130., 6.)
    assert day_and_hour(start, 7.5 * 3600.) == (130., 13.5)
    assert day_and_hour(start, 20. * 3600.) == (131., 2.)
    assert day_and_hour(start_date("2026-12-31 23:00"), 7200.) == (1., 1.)       # into the next year
