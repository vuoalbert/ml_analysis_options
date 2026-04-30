import pandas as pd
import pandas_market_calendars as mcal

NYSE = mcal.get_calendar("NYSE")


def rth_index(start: str, end: str, tz: str = "America/New_York") -> pd.DatetimeIndex:
    """Return the exact minute timestamps that NYSE was open, in the given tz."""
    sched = NYSE.schedule(start_date=start, end_date=end)
    # minute-level expansion, label=right means bar closes at the stamp
    idx = mcal.date_range(sched, frequency="1min")
    idx = idx.tz_convert(tz)
    return idx


def is_fomc_day(dates: pd.DatetimeIndex) -> pd.Series:
    # Hard-coded FOMC announcement dates for the 2023-2026 window.
    fomc = pd.to_datetime([
        # 2023
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
        "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        # 2026 (scheduled)
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    ]).date
    fomc_set = set(fomc)
    s = pd.Series(False, index=dates)
    s[:] = [d.date() in fomc_set for d in dates]
    return s


def is_zero_dte(dates: pd.DatetimeIndex) -> pd.Series:
    # SPY 0DTE exists on M/T/W/Th/F (since 2023). Approximation: weekday != holiday.
    wd = dates.weekday
    return pd.Series(wd < 5, index=dates)


def minutes_into_session(stamps: pd.DatetimeIndex) -> pd.Series:
    """Minutes since 09:30 local for each stamp; -1 if outside RTH."""
    local = stamps.tz_convert("America/New_York")
    mins = local.hour * 60 + local.minute - (9 * 60 + 30)
    out = pd.Series(mins, index=stamps)
    out[(mins < 0) | (mins > 390)] = -1
    return out
