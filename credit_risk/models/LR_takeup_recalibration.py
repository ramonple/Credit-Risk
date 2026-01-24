import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from typing import Optional, Tuple, Dict, Any, Iterable
from sklearn.linear_model import LogisticRegression

# ==========================================================
#   Generate bins for a series
# ==========================================================
def _make_bins(series: pd.Series, n_bins=10, method="quantile", edges=None):
    s = pd.Series(series).astype(float)

    if edges is not None:
        bins = [-np.inf] + list(edges) + [np.inf]
        b = pd.cut(s, bins=bins, include_lowest=True)
        return b, np.array(edges, dtype=float)

    if method == "quantile":
        b = pd.qcut(s, q=n_bins, duplicates="drop")
        # Extract edges so you can reuse them on scoring data if needed
        edges = np.unique(np.quantile(s.dropna().values, np.linspace(0, 1, b.cat.categories.size + 1)[1:-1]))
        return b, edges

    if method == "uniform":
        b = pd.cut(s, bins=n_bins, include_lowest=True)
        edges = np.linspace(s.min(), s.max(), n_bins + 1)[1:-1]
        return b, edges

    raise ValueError("method must be one of: {'quantile','uniform'}")


# ==========================================================
#   Segment AUC/Gini check by APR bins
# ==========================================================
def segment_auc_by_apr(
    df: pd.DataFrame,
    apr_col: str,
    p_col: str,
    y_col: str,
    n_bins: int = 10,
    method: str = "quantile",
    edges=None,
):
    """
    Compute segment-level AUC and Gini by APR bins, plus a GLOBAL row.

    Requires a helper _make_bins(series, n_bins, method, edges) that returns:
      (bin_series, edges_out)
    """
    d = df[[apr_col, p_col, y_col]].copy()

    # Create APR bins
    d["apr_bin"], edges_out = _make_bins(d[apr_col], n_bins=n_bins, method=method, edges=edges)

    rows = []

    # --- Global performance row ---
    if d[y_col].nunique() < 2:
        global_auc = np.nan
    else:
        global_auc = roc_auc_score(d[y_col], d[p_col])

    rows.append({
        "apr_bin": "GLOBAL",
        "count": len(d),
        "event_rate": float(d[y_col].mean()),
        "auc": global_auc,
        "gini": None if np.isnan(global_auc) else 2 * global_auc - 1,
    })

    # --- Segment rows ---
    for b, g in d.groupby("apr_bin", observed=True):
        if g[y_col].nunique() < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(g[y_col], g[p_col])

        rows.append({
            "apr_bin": str(b),
            "count": len(g),
            "event_rate": float(g[y_col].mean()),
            "auc": auc,
            "gini": None if np.isnan(auc) else 2 * auc - 1,
        })

    out = pd.DataFrame(rows)
    return out, edges_out




# ==========================================================
#   Calibration check by APR bins
# ==========================================================
def calibration_by_apr_bins(
    df: pd.DataFrame,
    apr_col: str,
    p_col: str,
    y_col: str,
    n_bins: int = 10,
    method: str = "quantile",
    edges=None,
):
    """
    Returns:
      summary_df with per-APR-bin stats:
        - count, event_rate (mean y), pred_mean (mean p), cal_error (y - p)
    """
    d = df[[apr_col, p_col, y_col]].copy()
    d["apr_bin"], edges_out = _make_bins(d[apr_col], n_bins=n_bins, method=method, edges=edges)

    grp = d.groupby("apr_bin", observed=True)
    summary = grp.agg(
        count=(y_col, "size"),
        apr_mean=(apr_col, "mean"),
        pred_mean=(p_col, "mean"),
        event_rate=(y_col, "mean"),
        pred_median=(p_col, "median"),
        apr_min=(apr_col, "min"),
        apr_max=(apr_col, "max"),
    ).reset_index()

    summary["cal_error"] = summary["event_rate"] - summary["pred_mean"]
    summary["cal_error_abs"] = summary["cal_error"].abs()

    return summary, edges_out


def plot_calibration_by_apr_bins(summary_df: pd.DataFrame, title="Calibration by APR bins"):
    """
    Two quick plots:
      (A) event_rate vs pred_mean by APR bin (with 45-degree reference)
      (B) calibration error (event_rate - pred_mean) vs apr_mean
    """
    # (A) Predicted vs Actual per APR bin
    x = summary_df["pred_mean"].values
    y = summary_df["event_rate"].values

    fig1 = plt.figure()
    plt.scatter(x, y)
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Mean predicted probability (within APR bin)")
    plt.ylabel("Observed take-up rate (within APR bin)")
    plt.title(title + " (Actual vs Predicted)")
    plt.grid(True)
    plt.show()

    # (B) Calibration error vs APR
    fig2 = plt.figure()
    plt.plot(summary_df["apr_mean"].values, summary_df["cal_error"].values, marker="o")
    plt.axhline(0.0)
    plt.xlabel("Mean APR (within bin)")
    plt.ylabel("Calibration error: mean(y) - mean(p_hat)")
    plt.title(title + " (Calibration Error vs APR)")
    plt.grid(True)
    plt.show()


# ==========================================================
# Residual plot vs APR
# ==========================================================
def residual_summary_by_apr(
    df: pd.DataFrame,
    apr_col: str,
    p_col: str,
    y_col: str,
    n_bins: int = 25,
    method: str = "quantile",
    edges=None,
):
    """
    Uses probability residuals: r = y - p_hat
    Returns binned summary for plotting mean residual vs APR.
    """
    d = df[[apr_col, p_col, y_col]].copy()
    d["resid"] = d[y_col].astype(float) - d[p_col].astype(float)
    d["apr_bin"], edges_out = _make_bins(d[apr_col], n_bins=n_bins, method=method, edges=edges)

    grp = d.groupby("apr_bin", observed=True)
    out = grp.agg(
        count=("resid", "size"),
        apr_mean=(apr_col, "mean"),
        resid_mean=("resid", "mean"),
        resid_median=("resid", "median"),
        resid_std=("resid", "std"),
        apr_min=(apr_col, "min"),
        apr_max=(apr_col, "max"),
    ).reset_index()

    return out, edges_out


def plot_residual_vs_apr(resid_summary_df: pd.DataFrame, title="Mean residual vs APR"):
    """
    Plots mean residual (y - p_hat) vs mean APR per bin.
    """
    fig = plt.figure()
    plt.plot(resid_summary_df["apr_mean"].values, resid_summary_df["resid_mean"].values, marker="o")
    plt.axhline(0.0)
    plt.xlabel("Mean APR (within bin)")
    plt.ylabel("Mean residual: y - p_hat")
    plt.title(title)
    plt.grid(True)
    plt.show()


# ======================
# Example usage
# ======================
# summary_cal, cal_edges = calibration_by_apr_bins(df, "APR", "p_hat", "take_up", n_bins=10, method="quantile")
# plot_calibration_by_apr_bins(summary_cal, title="Calibration by APR deciles")
#
# resid_sum, resid_edges = residual_summary_by_apr(df, "APR", "p_hat", "take_up", n_bins=25, method="quantile")
# plot_residual_vs_apr(resid_sum, title="Binned mean residual vs APR")




# ==========================================================
#   calibrate/modify predictions by APR bins
# ==========================================================
def _clip_prob(p, eps=1e-6):
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1 - eps)

def _logit(p, eps=1e-6):
    p = _clip_prob(p, eps)
    return np.log(p / (1 - p))

def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def modify_predictions_by_apr_ranges(
    df: pd.DataFrame,
    apr_col: str,
    p_col: str,
    y_col: str,
    apr_ranges,
    mode: str = "logit_shift",   # "mean_scale" | "logit_shift" | "platt"
    convert_threshold=None,
    out_proba_col: str = "p_adj",
    out_flag_col: str = "y_adj",
    eps: float = 1e-6,
):
    """
    Modify predicted probabilities only within explicitly defined APR ranges.
    
    Mean scaling: 
       multiplies predicted probabilities in the selected APR range 
       by a constant so the average predicted take-up matches the observed take-up.

    Log-odds (logit) shift:
       adds a constant to the log-odds of the predicted probabilities, 
       correcting systematic over- or under-confidence while preserving ranking.

    Platt scaling:
         fits a small logistic regression on the model’s scores within the APR range 
         to re-map scores to calibrated probabilities, adjusting both level and confidence.


    apr_ranges: list of dicts with keys:
      - name (str)
      - min (float or None)
      - max (float or None)
    """
    df_out = df.copy()
    p = _clip(df[p_col].values, eps)
    y = df[y_col].values.astype(int)
    s = _logit(p, eps)

    params = {}

    for r in apr_ranges:
        name = r.get("name", "range")
        lo = r.get("min", None)
        hi = r.get("max", None)

        mask = np.ones(len(df_out), dtype=bool)
        if lo is not None:
            mask &= df_out[apr_col].values >= lo
        if hi is not None:
            mask &= df_out[apr_col].values < hi

        if mask.sum() == 0:
            continue

        pg, yg, sg = p[mask], y[mask], s[mask]

        if mode == "mean_scale":
            k = float(np.mean(yg) / np.mean(pg)) if np.mean(pg) > 0 else 1.0
            p[mask] = _clip(k * pg, eps)
            params[name] = {"k": k}

        elif mode == "logit_shift":
            delta = float(_logit(np.mean(yg), eps) - _logit(np.mean(pg), eps))
            p[mask] = _clip(_sigmoid(sg + delta), eps)
            params[name] = {"delta": delta}

        elif mode == "platt":
            if len(np.unique(yg)) < 2:
                delta = float(_logit(np.mean(yg), eps) - _logit(np.mean(pg), eps))
                a, b = delta, 1.0
            else:
                lr = LogisticRegression(max_iter=2000)
                lr.fit(sg.reshape(-1, 1), yg)
                a, b = float(lr.intercept_[0]), float(lr.coef_[0][0])
            p[mask] = _clip(_sigmoid(a + b * sg), eps)
            params[name] = {"a": a, "b": b}

    df_out[out_proba_col] = p

    if convert_threshold is not None:
        df_out[out_flag_col] = (p >= convert_threshold).astype(int)

    return df_out, params

