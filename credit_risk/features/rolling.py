from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

Agg = Literal["mean", "sum", "min", "max", "std", "any_ge"]


@dataclass(frozen=True)
class RollingSpec:
    """Specification for a single rolling feature."""

    source_col: str
    window: int
    agg: Agg
    out_col: str
    min_periods: Optional[int] = None
    threshold: Optional[float] = None


def _validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")


def add_rolling_features(
    df: pd.DataFrame,
    *,
    group_cols: Union[str, Sequence[str]],
    date_col: str,
    specs: Sequence[RollingSpec],
    sort: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Add rolling features to an entity-time panel (e.g., account-month)."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    group_cols = list(group_cols)

    if validate:
        _validate_required_columns(df, [*group_cols, date_col])
        out_cols = [s.out_col for s in specs]
        dup = [c for c in out_cols if out_cols.count(c) > 1]
        if dup:
            raise ValueError(f"Duplicate out_col names in specs: {sorted(set(dup))}")
        for s in specs:
            if s.agg == "any_ge" and s.threshold is None:
                raise ValueError("RollingSpec.threshold must be set when agg='any_ge'")

    out = df.sort_values([*group_cols, date_col]).copy() if sort else df.copy()
    g = out.groupby(group_cols, sort=False)

    for s in specs:
        if validate:
            _validate_required_columns(out, [s.source_col])

        mp = s.min_periods if s.min_periods is not None else s.window
        roll = g[s.source_col].rolling(window=s.window, min_periods=mp)

        if s.agg == "mean":
            out[s.out_col] = roll.mean().reset_index(level=group_cols, drop=True)
        elif s.agg == "sum":
            out[s.out_col] = roll.sum().reset_index(level=group_cols, drop=True)
        elif s.agg == "min":
            out[s.out_col] = roll.min().reset_index(level=group_cols, drop=True)
        elif s.agg == "max":
            out[s.out_col] = roll.max().reset_index(level=group_cols, drop=True)
        elif s.agg == "std":
            out[s.out_col] = roll.std().reset_index(level=group_cols, drop=True)
        elif s.agg == "any_ge":
            thr = float(s.threshold)  # validated above
            out[s.out_col] = (
                roll.apply(lambda x: int(np.any(np.asarray(x) >= thr)), raw=False)
                .reset_index(level=group_cols, drop=True)
            )
        else:
            raise ValueError(f"Unsupported agg: {s.agg}")

    return out
