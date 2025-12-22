import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

def train_lr_and_eval_with_topx(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    top_fracs=(0.01, 0.05, 0.10, 0.20),
    lr_kwargs=None,
    dropna=True,
):
    """
    Trains Logistic Regression on train_df and evaluates on train/test:
      - ROC curve (train/test)
      - PR curve (train/test)
      - marks Top X% points on PR curves
      - prints Top X% tables (train/test)

    Inputs:
      train_df, test_df: DataFrames
      feature_cols: list of feature column names
      target_col: binary target column (0/1)
      top_fracs: fractions for top X% points
      lr_kwargs: dict passed to LogisticRegression(...)
      dropna: drop NA rows in features/target before training/eval

    Returns:
      model, topx_train_df, topx_test_df
    """

    if lr_kwargs is None:
        lr_kwargs = {}

    # ---------- helper: prepare X,y ----------
    def _prep(df: pd.DataFrame):
        cols = feature_cols + [target_col]
        use = df[cols].copy()
        if dropna:
            use = use.dropna(subset=cols)

        X = use[feature_cols]
        y = use[target_col].astype(int).to_numpy()

        extra = set(np.unique(y)) - {0, 1}
        if extra:
            raise ValueError(f"{target_col} must be binary 0/1. Found extra values: {extra}")

        return X, y, use

    # ---------- helper: compute Top X% table ----------
    def _topx_table(y_true, y_score, top_fracs, label_name):
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)

        n = len(y_true)
        n_pos = int(y_true.sum())
        if n == 0:
            raise ValueError(f"{label_name}: empty evaluation set.")
        if n_pos == 0:
            raise ValueError(f"{label_name}: no positive cases; TopX undefined.")

        base_rate = float(y_true.mean())
        pr_auc = float(average_precision_score(y_true, y_score))

        order = np.argsort(y_score)[::-1]
        rows = []
        marker_points = []  # (recall, precision, label)

        for frac in top_fracs:
            frac = float(frac)
            k = int(np.ceil(frac * n))
            k = max(k, 1)

            top_idx = order[:k]
            y_top = y_true[top_idx]

            precision_at_top = float(y_top.mean())
            recall_at_top = float(y_top.sum() / n_pos)
            threshold = float(y_score[top_idx].min())
            lift = float(precision_at_top / base_rate) if base_rate > 0 else np.nan

            rows.append({
                "dataset": label_name,
                "top_frac": frac,
                "top_pct": frac * 100.0,
                "k": k,
                "precision_at_top": precision_at_top,
                "recall_at_top": recall_at_top,
                "threshold": threshold,
                "base_rate": base_rate,
                "lift_vs_base_rate": lift,
                "pr_auc": pr_auc,
                "n": n,
                "n_pos": n_pos,
            })
            marker_points.append((recall_at_top, precision_at_top, f"{label_name} Top {int(round(frac*100))}%"))

        topx_df = pd.DataFrame(rows).sort_values("top_frac").reset_index(drop=True)
        return topx_df, marker_points

    # ---------- 1) prepare data ----------
    X_train, y_train, _ = _prep(train_df)
    X_test, y_test, _ = _prep(test_df)

    # ---------- 2) train model ----------
    # sensible defaults for credit data; override via lr_kwargs
    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        **lr_kwargs
    )
    model.fit(X_train, y_train)

    # ---------- 3) predicted probabilities ----------
    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    # ---------- 4) ROC ----------
    auc_train = roc_auc_score(y_train, p_train)
    auc_test = roc_auc_score(y_test, p_test)

    fpr_tr, tpr_tr, _ = roc_curve(y_train, p_train)
    fpr_te, tpr_te, _ = roc_curve(y_test, p_test)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr_tr, tpr_tr, label=f"Train ROC (AUC={auc_train:.4f}, Gini={2*auc_train-1:.4f})")
    ax_roc.plot(fpr_te, tpr_te, label=f"Test ROC (AUC={auc_test:.4f}, Gini={2*auc_test-1:.4f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="best")

    # ---------- 5) PR + TopX ----------
    prec_tr, rec_tr, _ = precision_recall_curve(y_train, p_train)
    prec_te, rec_te, _ = precision_recall_curve(y_test, p_test)

    ap_train = average_precision_score(y_train, p_train)
    ap_test = average_precision_score(y_test, p_test)

    topx_train_df, markers_train = _topx_table(y_train, p_train, top_fracs, "Train")
    topx_test_df, markers_test = _topx_table(y_test, p_test, top_fracs, "Test")

    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(rec_tr, prec_tr, label=f"Train PR (AP={ap_train:.4f})")
    ax_pr.plot(rec_te, prec_te, label=f"Test PR (AP={ap_test:.4f})")

    # baselines
    ax_pr.axhline(y_train.mean(), linestyle="--", label=f"Train baseline (BR={y_train.mean():.4f})")
    ax_pr.axhline(y_test.mean(), linestyle=":", label=f"Test baseline (BR={y_test.mean():.4f})")

    # add TopX markers
    for r, p, lab in markers_train:
        ax_pr.scatter([r], [p], label=lab)
    for r, p, lab in markers_test:
        ax_pr.scatter([r], [p], label=lab)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.set_title("Precision–Recall Curve + Top X% Points")
    ax_pr.legend(loc="best")

    # ---------- 6) print TopX tables ----------
    print("\n=== Top X% results: TRAIN ===")
    print(topx_train_df[[
        "top_pct","k","precision_at_top","recall_at_top","lift_vs_base_rate","threshold","base_rate","pr_auc"
    ]].to_string(index=False))

    print("\n=== Top X% results: TEST ===")
    print(topx_test_df[[
        "top_pct","k","precision_at_top","recall_at_top","lift_vs_base_rate","threshold","base_rate","pr_auc"
    ]].to_string(index=False))

    plt.show()

    return model, topx_train_df, topx_test_df
