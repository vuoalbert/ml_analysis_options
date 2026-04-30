"""Walk-forward fold iterator.

For a minute-indexed DataFrame covering multiple months, yield:
    (train_idx, val_idx, test_idx)
monthly folds where:
    train = last `train_months` full months ending before val
    val   = 1 month preceding test (used for early stopping and threshold tuning)
    test  = 1 untouched month
"""
from __future__ import annotations

import pandas as pd


def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz=ts.tz)


def _months_range(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    start = month_start(index[0])
    end = month_start(index[-1])
    out = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = (cur + pd.offsets.MonthBegin(1)).tz_convert(index.tz)
    return out


def fold_iter(index: pd.DatetimeIndex, train_months: int, val_months: int = 1, test_months: int = 1):
    months = _months_range(index)
    # need train_months + val_months months before the first test
    for i in range(train_months + val_months, len(months) - test_months + 1):
        test_start = months[i]
        test_end = months[i + test_months] if i + test_months < len(months) else (index[-1] + pd.Timedelta(seconds=1))
        val_start = months[i - val_months]
        val_end = test_start
        train_end = val_start
        train_start = months[i - val_months - train_months]

        train_mask = (index >= train_start) & (index < train_end)
        val_mask = (index >= val_start) & (index < val_end)
        test_mask = (index >= test_start) & (index < test_end)
        yield {
            "train": (train_start, train_end),
            "val": (val_start, val_end),
            "test": (test_start, test_end),
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        }


def tuning_split(index: pd.DatetimeIndex, frac: float = 0.5, val_frac: float = 0.25):
    """Initial hyperparam-tuning split: first `frac` of data, internally split train/val."""
    n = len(index)
    cutoff = int(n * frac)
    sub = index[:cutoff]
    val_n = int(len(sub) * val_frac)
    train_idx = sub[:-val_n]
    val_idx = sub[-val_n:]
    return train_idx, val_idx
