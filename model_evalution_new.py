import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curve_and_topx(
    df: pd.DataFrame,
    y_col: str,
    score_col: str,
    top_fracs=(0.01, 0.05, 0.10, 0.20),
    dropna: bool = True,
    title: str | None = None,
):
    """
    Plot Precision–Recall curve and return a Top-X% summary table.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing y_col (0/1) and score_col (float scores, higher = more risky).
    y_col : str
        Binary label column name (0/1).
    score_col : str
        Prediction score/probability column name (higher = more risky).
    top_fracs : tuple
        Fractions for Top X% metrics (e.g., 0.10 = top 10%).
    dropna : bool
        Drop rows with NaN in y_col or score_col.
    title : str | None
        Plot title.

    Returns
    -------
    topx_df : pd.DataFrame
        One row per Top X% with:
          - top_pct, k, precision_at_top, recall_at_top, threshold
          - base_rate, lift_vs_base_rate
          - pr_auc, n, n_pos
    fig, ax : matplotlib Figure and Axes
    """
    # ---- 1) Clean & extract arrays (single source of truth) ----
    use = df[[y_col, score_col]].copy()
    if dropna:
        use = use.dropna(subset=[y_col, score_col])

    if use.empty:
        raise ValueError("No rows to evaluate after filtering/dropna.")

    y = use[y_col].astype(int).to_numpy()
    s = use[score_col].astype(float).to_numpy()

    extra = set(np.unique(y)) - {0, 1}
    if extra:
        raise ValueError(f"{y_col} must be binary 0/1. Found extra values: {extra}")

    n = len(y)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("No positive (bad) cases in y; PR curve undefined.")

    base_rate = float(y.mean())

    # ---- 2) PR curve + PR-AUC (Average Precision) ----
    precision, recall, _ = precision_recall_curve(y, s)
    pr_auc = float(average_precision_score(y, s))

    # ---- 3) Top-X% metrics ----
    order = np.argsort(s)[::-1]  # high -> low
    rows = []
    pr_points = []  # for markers on plot: (recall, precision, label)

    for frac in top_fracs:
        frac = float(frac)
        if not (0 < frac <= 1):
            raise ValueError(f"Each top_fracs value must be in (0,1]. Got {frac}")

        k = int(np.ceil(frac * n))
        k = max(k, 1)

        top_idx = order[:k]
        y_top = y[top_idx]

        p_at = float(y_top.mean())
        r_at = float(y_top.sum() / n_pos)
        thr = float(s[top_idx].min())  # cutoff score for inclusion in top-k
        lift = float(p_at / base_rate) if base_rate > 0 else np.nan

        rows.append({
            "top_frac": frac,
            "top_pct": frac * 100.0,
            "k": k,
            "precision_at_top": p_at,
            "recall_at_top": r_at,
            "threshold": thr,
            "base_rate": base_rate,
            "lift_vs_base_rate": lift,
            "pr_auc": pr_auc,
            "n": n,
            "n_pos": n_pos,
            "y_col": y_col,
            "score_col": score_col,
        })
        pr_points.append((r_at, p_at, f"Top {int(round(frac*100))}%"))

    topx_df = pd.DataFrame(rows).sort_values("top_frac").reset_index(drop=True)

    # ---- 4) Plot ----
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"PR curve (AP={pr_auc:.4f})")
    ax.axhline(base_rate, linestyle="--", label=f"Baseline (bad rate={base_rate:.4f})")

    for r_at, p_at, lab in pr_points:
        ax.scatter([r_at], [p_at], label=lab)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title or f"Precision–Recall Curve: {score_col}")
    ax.legend(loc="best")

    return topx_df, fig, ax



def explain_first_topx_row(topx_df: pd.DataFrame):
    """
    Print a human-readable explanation for the first row of a Top-X% table.
    Assumes the table is sorted by top_frac (smallest X first).
    """
    row = topx_df.iloc[0]

    msg = (
        f"For the top {row['top_pct']:.0f}% highest-risk accounts "
        f"({int(row['k'])} out of {int(row['n'])} total), "
        f"the observed bad rate is {row['precision_at_top']:.2%}, "
        f"compared with an overall portfolio bad rate of {row['base_rate']:.2%}. "
        f"This represents a lift of {row['lift_vs_base_rate']:.1f}× over random selection. "
        f"These accounts capture {row['recall_at_top']:.1%} of all bad outcomes "
        f"(3@12)."
    )

    print(msg)

# new checks for model evaluation

# 1. Since you said your predicted dataset already contains all columns + bad_flag + pred_proba, make one canonical dataframe for evaluation:
eval_df, y_col="bad_flag", p_col="pred_proba", month_col="month", segment_cols=[...]

# Helper: build a clean eval_df once
def build_eval_df(df, split_name, y_col, p_col, written_month_col):
    out = df.copy()
    out["dataset_split"] = split_name
    out["month"] = pd.to_datetime(out[written_month_col]).dt.to_period("M").dt.to_timestamp()
    # keep y as float so immature can be NaN
    if y_col in out.columns:
        out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    out[p_col] = pd.to_numeric(out[p_col], errors="coerce")
    return out

# ======== Brier + reliability bins ========
from sklearn.metrics import brier_score_loss

def calibration_report(eval_df, y_col="bad_flag", p_col="pred_proba", n_bins=10):
    df = eval_df.dropna(subset=[y_col, p_col]).copy()
    y = df[y_col].astype(int).to_numpy()
    p = df[p_col].astype(float).to_numpy()

    brier = brier_score_loss(y, p)

    # decile bins by predicted probability
    df["bin"] = pd.qcut(df[p_col], q=n_bins, duplicates="drop")
    bins = (df.groupby("bin")
              .agg(
                  n=("bin", "size"),
                  mean_pred=(p_col, "mean"),
                  actual_rate=(y_col, "mean"),
                  min_pred=(p_col, "min"),
                  max_pred=(p_col, "max"),
              )
              .reset_index(drop=True))

    return brier, bins

# ======== monthly forecast error KPIs ========
def monthly_error_kpis(monthly_df, actual_col="actual_bad_rate", pred_col="pred_bad_rate"):
    df = monthly_df.dropna(subset=[actual_col, pred_col]).copy()
    df["error"] = df[pred_col] - df[actual_col]
    mae = float(np.mean(np.abs(df["error"])))
    rmse = float(np.sqrt(np.mean(df["error"]**2)))
    bias = float(np.mean(df["error"]))
    return {"MAE": mae, "RMSE": rmse, "BIAS": bias, "n_months": len(df)}, df


# ======== segment level monthly overview ========
def segment_monthly_overview(eval_df, segment_cols, y_col="bad_flag", p_col="pred_proba", month_col="month"):
    """
    Returns a table with, for each segment and month:
      - actual bad rate (where available)
      - predicted bad rate (avg PD)
      - counts
      - error
    """
    df = eval_df.copy()

    # predicted monthly
    pred = (df.groupby(segment_cols + [month_col], as_index=False)
              .agg(
                  n=("dataset_split", "size"),
                  pred_bad_rate=(p_col, "mean")
              ))

    # actual monthly (only where y available)
    has_y = df[y_col].notna()
    act = (df.loc[has_y]
             .groupby(segment_cols + [month_col], as_index=False)
             .agg(
                 n_labeled=(y_col, "size"),
                 actual_bad_rate=(y_col, "mean")
             ))

    out = pred.merge(act, on=segment_cols + [month_col], how="left")
    out["error"] = out["pred_bad_rate"] - out["actual_bad_rate"]
    return out


# ===== Runner: one function to run everything =====
def run_full_evaluation(
    df_train_pred, df_test_pred, df_immature_pred,
    y_col="bad_flag", p_col="pred_proba", written_month_col="written_month",
    segment_cols=None
):
    if segment_cols is None:
        segment_cols = []

    # 1) Build combined eval_df
    eval_train = build_eval_df(df_train_pred, "train", y_col, p_col, written_month_col)
    eval_test  = build_eval_df(df_test_pred, "test",  y_col, p_col, written_month_col)
    eval_imm   = build_eval_df(df_immature_pred, "immature", y_col, p_col, written_month_col)
    eval_all = pd.concat([eval_train, eval_test, eval_imm], ignore_index=True)

    # 2) Existing outputs you already have:
    # - ROC/PR + TopX (typically on train/test separately)
    # - Monthly overview plot and monthly_df (on eval_all)
    # I’ll assume you already compute monthly_df elsewhere.

    # 3) New: calibration on TEST (recommended)
    brier_test, calib_bins_test = calibration_report(eval_test, y_col=y_col, p_col=p_col, n_bins=10)

    # 4) New: segment monthly view (recommended)
    seg_monthly = None
    if segment_cols:
        seg_monthly = segment_monthly_overview(eval_all, segment_cols, y_col=y_col, p_col=p_col, month_col="month")

    return {
        "eval_all": eval_all,
        "brier_test": brier_test,
        "calibration_bins_test": calib_bins_test,
        "segment_monthly": seg_monthly,
    }

#What to use to decide “best model”

# ======= For your use case, I’d rank models by =======
# a. Monthly MAE / Bias on TEST months (forecasting quality)
# b. Brier score on TEST (probability quality)
# c. Gini / PR-AUC on TEST (ranking quality)
# d. Stability across key segments (no nasty pockets)


Use TEST data for all performance evaluation and model comparison; use TRAIN and IMMATURE data only for diagnostics, context, and monitoring.
                                                                                                   
| Check                 | Train | Test | Immature |
| --------------------- | ----- | ---- | -------- |
| AUC / Gini            | ◯     | ✅    | ❌        |
| PR curve / Top-X%     | ❌     | ✅    | ❌        |
| Brier / Calibration   | ❌     | ✅    | ❌        |
| Confusion metrics     | ❌     | ✅    | ❌        |
| Monthly MAE/RMSE/Bias | ❌     | ✅    | ❌        |
| Monthly trend plot    | ◯     | ◯    | ◯        |
| Segment performance   | ❌     | ✅    | ❌        |
| Drift / PSI           | ◯     | ◯    | ◯        |

Legend:
✅ = must use
◯ = optional / diagnostic
❌ = should not use


