
# ==================== deal with missings ================
import numpy as np
import pandas as pd
import re

def impute_by_month_group_mean(
    df: pd.DataFrame,
    cols: list[str],
    month_col: str,
    invalid_below: float = -7000,
    extra_invalid_values: list[float | int] | None = None,
    inplace: bool = False,
):
    """
    Impute selected columns using group (month) mean.

    Rules (per column):
      1) Treat any value containing letters (a-z/A-Z) as invalid -> set to NaN
      2) Treat numeric values < invalid_below as invalid -> set to NaN
      3) Treat values in extra_invalid_values as invalid -> set to NaN
      4) Convert remaining values to numeric
      5) Impute NaNs using the mean within each month group
      6) If a month group mean is NaN (all invalid), fallback to overall column mean
         If still NaN, fill with 0.0

    Returns:
      df_out, report_df
        - df_out: dataframe with imputed columns
        - report_df: per-column summary counts (invalid letters, invalid below, etc.)
    """
    df_out = df if inplace else df.copy()

    extra_invalid_values = extra_invalid_values or []

    # Ensure month column is datetime (grouping key)
    df_out[month_col] = pd.to_datetime(df_out[month_col])

    report_rows = []

    letter_pat = re.compile(r"[A-Za-z]")

    for c in cols:
        s_raw = df_out[c]

        # Work on string view for letter detection
        s_str = s_raw.astype(str)

        has_letters = s_str.str.contains(letter_pat, na=False)

        # Convert to numeric (non-numeric -> NaN)
        s_num = pd.to_numeric(s_raw, errors="coerce")

        invalid_low = s_num < invalid_below
        invalid_extra = s_num.isin(extra_invalid_values)

        # Mark invalids as NaN (letters OR low OR extra OR existing NaN)
        invalid_mask = has_letters | invalid_low | invalid_extra | s_num.isna()
        s_clean = s_num.mask(invalid_mask, np.nan)

        # Group mean by month
        group_mean = s_clean.groupby(df_out[month_col]).transform("mean")

        # Overall mean fallback
        overall_mean = float(np.nanmean(s_clean.to_numpy())) if np.isfinite(np.nanmean(s_clean.to_numpy())) else np.nan

        # Fill: first by group mean, then overall mean, then 0
        s_imputed = s_clean.fillna(group_mean)
        if np.isfinite(overall_mean):
            s_imputed = s_imputed.fillna(overall_mean)
        s_imputed = s_imputed.fillna(0.0)

        df_out[c] = s_imputed

        report_rows.append({
            "column": c,
            "n_rows": int(len(df_out)),
            "n_letters_replaced": int(has_letters.sum()),
            "n_below_threshold_replaced": int(invalid_low.fillna(False).sum()),
            "n_extra_invalid_replaced": int(invalid_extra.fillna(False).sum()),
            "n_non_numeric_replaced": int(s_num.isna().sum()),
            "n_total_imputed": int((s_clean.isna()).sum()),
            "overall_mean_used_if_needed": overall_mean,
        })

    report_df = pd.DataFrame(report_rows)
    return df_out, report_df


# ====== New feature creation -- change/trend -----
import numpy as np
import pandas as pd

def add_change_features(
    df: pd.DataFrame,
    specs: list[dict],
    *,
    invalid_values: tuple[float | int, ...] = (-9999, 999),
    default_min_valid: float | None = None,
    default_max_valid: float | None = None,
    clip_pct: float | None = 5.0,   # +/-500%
    clip_sym: float | None = 2.0,   # +/-200%
    clip_log: float | None = 5.0,
    denom_mode: str = "signed",     # "signed" (shift/app), "abs" (shift/|app|)
    denom_zero: str = "nan",        # "nan" or "eps"
    eps: float = 1e-12,
    suffix_abs: str = "_shift",
    suffix_pct: str = "_pct_change",
    suffix_sym: str = "_sym_pct_change",
    suffix_log: str = "_log_change",
    suffix_dir: str = "_dir",
    suffix_valid: str = "_chg_valid",
    inplace: bool = False,
):
    """
    Create change-related features from (current, application, shift) with per-feature valid ranges.

    specs: list[dict], each dict:
      required:
        - name: base name for new features
        - curr: current value col
        - app:  application value col
      optional:
        - shift: absolute change col (curr - app). If missing, computed.
        - min_valid: values below are treated as missing (NaN)
        - max_valid: values above are treated as missing (NaN)
        - invalid_values: override default invalid_values for this feature group (e.g., (-9999, 999, 9999))
        - denom_mode: override denom_mode per feature ("signed" or "abs")

    denom_mode:
      - "signed": pct = (curr - app) / app        (keeps sign of denominator; supports negative app values)
      - "abs":    pct = (curr - app) / |app|     (scale-only; sign comes only from numerator)

    denom_zero:
      - "nan": if app == 0 -> pct_change = NaN and valid flag = 0
      - "eps": if app == 0 -> pct_change = (curr-app)/eps (usually not recommended)

    Returns a new DataFrame unless inplace=True.
    """
    out = df if inplace else df.copy()

    def _clean(s: pd.Series, min_valid, max_valid, inv_vals) -> pd.Series:
        s = s.replace(list(inv_vals), np.nan)
        s = pd.to_numeric(s, errors="coerce")
        if min_valid is not None:
            s = s.mask(s < min_valid, np.nan)
        if max_valid is not None:
            s = s.mask(s > max_valid, np.nan)
        return s

    for spec in specs:
        name = spec["name"]
        curr_col = spec["curr"]
        app_col = spec["app"]
        shift_col = spec.get("shift", None)

        inv_vals = tuple(spec.get("invalid_values", invalid_values))
        min_valid = spec.get("min_valid", default_min_valid)
        max_valid = spec.get("max_valid", default_max_valid)
        denom_mode_i = spec.get("denom_mode", denom_mode)

        curr = _clean(out[curr_col], min_valid, max_valid, inv_vals)
        app = _clean(out[app_col], min_valid, max_valid, inv_vals)

        if shift_col and shift_col in out.columns:
            shift = _clean(out[shift_col], min_valid, max_valid, inv_vals)
        else:
            shift = curr - app

        out[f"{name}{suffix_abs}"] = shift
        out[f"{name}{suffix_dir}"] = np.sign(shift).astype("float")

        # ---- pct change (vs application value) ----
        if denom_mode_i not in {"signed", "abs"}:
            raise ValueError("denom_mode must be 'signed' or 'abs'")

        denom = app if denom_mode_i == "signed" else np.abs(app)

        if denom_zero == "nan":
            valid = curr.notna() & app.notna() & (denom != 0)
        elif denom_zero == "eps":
            valid = curr.notna() & app.notna()
            denom = denom.mask(denom == 0, eps)
        else:
            raise ValueError("denom_zero must be 'nan' or 'eps'")

        pct = np.where(valid, (curr - app) / (denom + eps), np.nan)
        if clip_pct is not None:
            pct = np.clip(pct, -clip_pct, clip_pct)

        out[f"{name}{suffix_pct}"] = pct
        out[f"{name}{suffix_valid}"] = valid.astype(int)

        # ---- symmetric pct change ----
        sym_denom = (np.abs(curr) + np.abs(app)) / 2.0
        sym_valid = curr.notna() & app.notna() & (sym_denom != 0)
        sym = np.where(sym_valid, (curr - app) / (sym_denom + eps), np.nan)
        if clip_sym is not None:
            sym = np.clip(sym, -clip_sym, clip_sym)
        out[f"{name}{suffix_sym}"] = sym

        # ---- log ratio change (only if both > 0) ----
        log_valid = curr.notna() & app.notna() & (curr > 0) & (app > 0)
        logc = np.where(log_valid, np.log((curr + eps) / (app + eps)), np.nan)
        if clip_log is not None:
            logc = np.clip(logc, -clip_log, clip_log)
        out[f"{name}{suffix_log}"] = logc

    return out

