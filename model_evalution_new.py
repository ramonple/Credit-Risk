====================================== summary =================================================
Model Performance Summary

1. Objective & Scope
Target: 3@12 delinquency (binary)
Portfolio: 
Sample: 

2. Discrimination (Ranking Power)
ROC-AUC / Gini (Test)
PR-AUC (Test)
(Assesses ability to rank bads above goods.)

3. Precision–Recall Operating Point Analysis (Top-X%)
Precision@X%, Recall@X%, Lift (Test)
Interpretation of Top-X% capture
(Assesses risk concentration under fixed capacity.)

4. Calibration & Probability Quality
Brier score (Test)
Reliability curve (deciles, Test)
(Assesses whether predicted PDs are numerically meaningful.)

5. Forecasting Performance (Portfolio-Level)
Monthly actual vs predicted 3@12 (plot)
Monthly MAE / RMSE / Bias (Test months)
(Primary metric for DDM forecasting.)

6. Stability & Segmentation
Performance stability by written month (Test)
Key segment checks (e.g. RN band, channel)
(Ensures robustness and absence of weak pockets.)

7. Explainability
Feature importance (Top 10)
SHAP summary plot (Train)
(Provides transparency and directional validation.)

8. Benchmark Comparison
ML vs Logistic Regression
Summary decision


9. Conclusion
Selected model
Key strengths / limitations
Recommended next steps

===============================================================================================
=== reporting function ===
    


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



import numpy as np
import pandas as pd

def monthly_forecast_report(
    df: pd.DataFrame,
    month_col: str,
    y_col: str = "bad_flag",
    p_col: str = "pred_proba",
    weight_col: str | None = None,
    split_col: str | None = None,
    split_for_kpi: tuple[str, ...] | None = ("test",),  # set None to use all labeled rows
    month_as_period: bool = True,
    show_last_n_months: int | None = 12,  # None => show all
):
    """
    End-to-end monthly forecasting report from a scored dataset (auto-displays tables).

    - Displays overall KPI table (MAE/RMSE/Bias) + plain-English explanation
    - Displays monthly table (rates/errors as % strings)
    - Returns: monthly_df (numeric), overall_df (numeric), monthly_display_df (formatted)

    Notes
    -----
    - Immature months are handled naturally: y_col can be NaN => actual rate & errors become NaN.
    - Overall KPIs are computed across labeled months only, optionally restricted to splits
      (e.g., TEST months only) via split_col + split_for_kpi.
    """

    # Lazy import so the function still works outside notebooks
    try:
        from IPython.display import display  # type: ignore
    except Exception:
        display = None

    def wmean(x, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        m = np.isfinite(x) & np.isfinite(w)
        if m.sum() == 0:
            return np.nan
        return float(np.sum(x[m] * w[m]) / np.sum(w[m]))

    def fmt_pct(x, digits=2):
        return f"{x:.{digits}%}" if pd.notna(x) else ""

    # -------------------------
    # 1) Clean and normalize
    # -------------------------
    cols = [month_col, p_col]
    if y_col in df.columns:
        cols.append(y_col)
    if weight_col:
        cols.append(weight_col)
    if split_col:
        cols.append(split_col)

    use = df[cols].copy()
    use[p_col] = pd.to_numeric(use[p_col], errors="coerce")
    if y_col in use.columns:
        use[y_col] = pd.to_numeric(use[y_col], errors="coerce")

    if month_as_period:
        use["_month"] = pd.to_datetime(use[month_col]).dt.to_period("M").dt.to_timestamp()
    else:
        use["_month"] = use[month_col]

    # -------------------------
    # 2) Monthly table (ALL months)
    # -------------------------
    rows = []
    for m, g in use.groupby("_month", sort=True):
        n = len(g)

        # predicted monthly rate
        if weight_col:
            pred_rate = wmean(g[p_col], g[weight_col])
        else:
            pred_rate = float(np.nanmean(g[p_col]))

        # actual monthly rate (only where y exists)
        if y_col in g.columns:
            labeled = g[g[y_col].notna()]
            n_labeled = len(labeled)
            if n_labeled > 0:
                if weight_col:
                    actual_rate = wmean(labeled[y_col], labeled[weight_col])
                else:
                    actual_rate = float(labeled[y_col].mean())
            else:
                actual_rate = np.nan
        else:
            n_labeled = 0
            actual_rate = np.nan

        row = {
            "_month": m,
            "n": n,
            "n_labeled": n_labeled,
            "pred_bad_rate": pred_rate,
            "actual_bad_rate": actual_rate,
        }

        if split_col and split_col in g.columns:
            vc = g[split_col].value_counts(dropna=False)
            row["dominant_split"] = vc.index[0]

        rows.append(row)

    monthly_df = pd.DataFrame(rows).sort_values("_month").reset_index(drop=True)
    monthly_df["error"] = monthly_df["pred_bad_rate"] - monthly_df["actual_bad_rate"]
    monthly_df["abs_error"] = monthly_df["error"].abs()
    monthly_df["sq_error"] = monthly_df["error"] ** 2

    # -------------------------
    # 3) Overall KPIs (labeled months only; optionally restricted to splits)
    # -------------------------
    if split_col and split_for_kpi is not None:
        desired = set(split_for_kpi)
        df_kpi = use.copy()
        df_kpi = df_kpi[df_kpi[split_col].isin(desired)]
        if y_col in df_kpi.columns:
            df_kpi = df_kpi[df_kpi[y_col].notna()]
        else:
            df_kpi = df_kpi.iloc[0:0]

        kpi_rows = []
        for m, g in df_kpi.groupby("_month", sort=True):
            if weight_col:
                pred_rate = wmean(g[p_col], g[weight_col])
                actual_rate = wmean(g[y_col], g[weight_col])
            else:
                pred_rate = float(np.nanmean(g[p_col]))
                actual_rate = float(np.nanmean(g[y_col]))
            kpi_rows.append({"_month": m, "pred": pred_rate, "actual": actual_rate})

        kpi_months = pd.DataFrame(kpi_rows)
        if not kpi_months.empty:
            kpi_months["error"] = kpi_months["pred"] - kpi_months["actual"]
            mae = float(np.mean(np.abs(kpi_months["error"])))
            rmse = float(np.sqrt(np.mean(kpi_months["error"] ** 2)))
            bias = float(np.mean(kpi_months["error"]))
            n_months = int(len(kpi_months))
        else:
            mae = rmse = bias = np.nan
            n_months = 0

        kpi_scope = f"{split_for_kpi}"
    else:
        eligible = monthly_df.dropna(subset=["actual_bad_rate", "pred_bad_rate"]).copy()
        if len(eligible) > 0:
            mae = float(eligible["abs_error"].mean())
            rmse = float(np.sqrt(eligible["sq_error"].mean()))
            bias = float(eligible["error"].mean())
            n_months = int(len(eligible))
        else:
            mae = rmse = bias = np.nan
            n_months = 0

        kpi_scope = "all labeled months"

    overall_df = pd.DataFrame([{
        "kpi_scope": kpi_scope,
        "weighting": "weighted" if weight_col else "unweighted",
        "n_months": n_months,
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
    }])

    # -------------------------
    # 4) Display-friendly tables (% formatting)
    # -------------------------
    # Monthly display
    monthly_display_df = monthly_df.copy()
    if np.issubdtype(monthly_display_df["_month"].dtype, np.datetime64):
        monthly_display_df["_month"] = monthly_display_df["_month"].dt.strftime("%Y-%m")

    for c in ["actual_bad_rate", "pred_bad_rate", "error", "abs_error"]:
        monthly_display_df[c] = monthly_display_df[c].map(lambda x: fmt_pct(x, digits=2))
    monthly_display_df["sq_error"] = monthly_display_df["sq_error"].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")

    display_cols = ["_month", "n", "n_labeled", "actual_bad_rate", "pred_bad_rate", "error", "abs_error"]
    if "dominant_split" in monthly_display_df.columns:
        display_cols.insert(1, "dominant_split")
    monthly_display_df = monthly_display_df[display_cols]

    # Overall display (also show % nicely)
    overall_display_df = overall_df.copy()
    for c in ["MAE", "RMSE", "Bias"]:
        overall_display_df[c] = overall_display_df[c].map(lambda x: fmt_pct(x, digits=4) if pd.notna(x) else "")

    # -------------------------
    # 5) Show results + explanation
    # -------------------------
    if display is not None:
        display(overall_display_df)
    else:
        print(overall_display_df.to_string(index=False))

    # Explanation (plain English)
    if n_months == 0 or not np.isfinite(mae):
        explanation = (
            "Overall monthly forecast KPIs are not available because there are no labeled months "
            f"in the selected KPI scope ({kpi_scope})."
        )
    else:
        mae_pp = mae * 100
        rmse_pp = rmse * 100
        bias_pp = bias * 100

        if abs(bias_pp) < 0.05:
            bias_sentence = "Bias is close to zero, indicating little systematic over/under prediction."
        else:
            direction = "over-predicts" if bias_pp > 0 else "under-predicts"
            bias_sentence = f"On average the model {direction} by about {abs(bias_pp):.2f} percentage points."

        explanation = (
            f"Interpretation (based on {n_months} labeled months in scope {kpi_scope}):\n"
            f"- MAE = {mae:.4%} (~{mae_pp:.2f} percentage points): the typical month-level miss between "
            f"predicted and actual 3@12 rate.\n"
            f"- RMSE = {rmse:.4%} (~{rmse_pp:.2f} percentage points): similar to MAE implies few extreme outlier months.\n"
            f"- Bias = {bias:.4%} (~{bias_pp:.2f} percentage points): {bias_sentence}"
        )

    print(explanation)

    # Monthly table display
    if show_last_n_months is not None and len(monthly_display_df) > show_last_n_months:
        monthly_to_show = monthly_display_df.tail(show_last_n_months)
    else:
        monthly_to_show = monthly_display_df

    if display is not None:
        display(monthly_to_show)
    else:
        print(monthly_to_show.to_string(index=False))

    return monthly_df, overall_df, monthly_display_df



# ======== segment level monthly overview ========
import pandas as pd
import numpy as np

def segment_monthly_overview(
    df: pd.DataFrame,
    segment_cols: list[str],
    month_col: str,
    y_col: str = "bad_flag",
    p_col: str = "pred_proba",
    weight_col: str | None = None,
    month_as_period: bool = True,
):
    """
    Segment-level monthly overview:
      - predicted monthly bad rate = avg(pred_proba) (weighted if weight_col provided)
      - actual monthly bad rate = avg(bad_flag) for rows where bad_flag is not null
      - counts + error

    Works even if future/immature months have bad_flag = NaN.

    Returns
    -------
    pd.DataFrame with one row per (segment(s), month):
      segment cols..., month, n, n_labeled, pred_bad_rate, actual_bad_rate, error
    """

    use_cols = segment_cols + [month_col, p_col]
    if y_col in df.columns:
        use_cols.append(y_col)
    if weight_col:
        use_cols.append(weight_col)

    use = df[use_cols].copy()
    use[p_col] = pd.to_numeric(use[p_col], errors="coerce")
    if y_col in use.columns:
        use[y_col] = pd.to_numeric(use[y_col], errors="coerce")

    # normalize month to month start for clean grouping
    if month_as_period:
        use["_month"] = pd.to_datetime(use[month_col]).dt.to_period("M").dt.to_timestamp()
    else:
        use["_month"] = use[month_col]

    # weighted mean helper
    def wmean(x, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        m = np.isfinite(x) & np.isfinite(w)
        if m.sum() == 0:
            return np.nan
        return float(np.sum(x[m] * w[m]) / np.sum(w[m]))

    group_keys = segment_cols + ["_month"]

    # predicted monthly bad rate (all rows)
    if weight_col:
        pred = (use.groupby(group_keys, as_index=False)
                  .apply(lambda g: pd.Series({
                      "n": len(g),
                      "pred_bad_rate": wmean(g[p_col], g[weight_col])
                  }))
                  .reset_index(drop=True))
    else:
        pred = (use.groupby(group_keys, as_index=False)
                  .agg(n=(p_col, "size"),
                       pred_bad_rate=(p_col, "mean")))

    # actual monthly bad rate (only labeled rows)
    if y_col in use.columns:
        labeled = use[use[y_col].notna()].copy()

        if weight_col:
            act = (labeled.groupby(group_keys, as_index=False)
                        .apply(lambda g: pd.Series({
                            "n_labeled": len(g),
                            "actual_bad_rate": wmean(g[y_col], g[weight_col])
                        }))
                        .reset_index(drop=True))
        else:
            act = (labeled.groupby(group_keys, as_index=False)
                        .agg(n_labeled=(y_col, "size"),
                             actual_bad_rate=(y_col, "mean")))
    else:
        act = pred[group_keys].copy()
        act["n_labeled"] = 0
        act["actual_bad_rate"] = np.nan

    out = pred.merge(act, on=group_keys, how="left")
    out["error"] = out["pred_bad_rate"] - out["actual_bad_rate"]

    # sort for readability
    out = out.sort_values(group_keys).reset_index(drop=True)

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


