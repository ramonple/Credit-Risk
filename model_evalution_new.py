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

