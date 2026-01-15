import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

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
def segment_auc_by_apr(df, apr_col, p_col, y_col, n_bins=10):
    """
    Compute segment-level AUC and Gini by APR  bins.
    """
    df = df.copy()

    # APR quantile bins
    df["apr_bin"] = pd.qcut(df[apr_col], q=n_bins, duplicates="drop")

    rows = []
    for b, g in df.groupby("apr_bin"):
        # Need both classes present to compute AUC
        if g[y_col].nunique() < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(g[y_col], g[p_col])

        rows.append({
            "apr_bin": b,
            "count": len(g),
            "event_rate": g[y_col].mean(),
            "auc": auc,
            "gini": None if np.isnan(auc) else 2 * auc - 1
        })

    return pd.DataFrame(rows)



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

