import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import roc_auc_score, brier_score_loss

# ============================================================
# 1. LOAD DATA
# ------------------------------------------------------------
# reference_data = training or previous stable period
# current_data   = latest production batch
# prediction_col = model scores (PDs)
# target_col     = actual outcome if available
# ============================================================

reference_data = pd.read_csv("reference_period.csv")
current_data   = pd.read_csv("current_period.csv")

# ============================================================
# 2. DEFINE PSI FUNCTION
# ============================================================

def calculate_psi(ref_series, cur_series, bins=10):
    ref_perc, _ = np.histogram(ref_series, bins=bins)
    cur_perc, _ = np.histogram(cur_series, bins=bins)

    ref_perc = ref_perc / len(ref_series)
    cur_perc = cur_perc / len(cur_series)

    psi = np.sum((ref_perc - cur_perc) * np.log(ref_perc / cur_perc))
    return psi

# ============================================================
# 3. DATA DRIFT: PSI FOR EACH FEATURE
# ============================================================

feature_psi = {}

for col in reference_data.columns:
    if col in ["target", "prediction"]:
        continue
    psi_val = calculate_psi(reference_data[col], current_data[col])
    feature_psi[col] = psi_val

# Flag drift based on common thresholds
drift_flags = {
    col: ("No Issue" if psi < 0.1 else "Moderate Drift" if psi < 0.25 else "Significant Drift")
    for col, psi in feature_psi.items()
}

# ============================================================
# 4. FEATURE-LEVEL DRIFT WITH KS & CHI-SQUARE
# ============================================================

feature_ks = {}
feature_chi = {}

for col in reference_data.columns:
    if col in ["target", "prediction"]:
        continue
    
    # KS test (continuous features)
    try:
        ks_stat, ks_p = ks_2samp(reference_data[col], current_data[col])
        feature_ks[col] = ks_p
    except:
        feature_ks[col] = np.nan

    # Chi-square (categorical features)
    try:
        contingency = pd.crosstab(reference_data[col], current_data[col])
        chi_stat, chi_p, _, _ = chi2_contingency(contingency)
        feature_chi[col] = chi_p
    except:
        feature_chi[col] = np.nan

# ============================================================
# 5. PREDICTION DRIFT
# ============================================================

psi_prediction = calculate_psi(
    reference_data["prediction"],
    current_data["prediction"]
)

# ============================================================
# 6. MODEL PERFORMANCE MONITORING (IF TARGET AVAILABLE)
# ============================================================

performance_metrics = {}

if "target" in current_data.columns:
    performance_metrics["AUC"] = roc_auc_score(current_data["target"], 
                                               current_data["prediction"])

    # Calibration
    performance_metrics["BrierScore"] = brier_score_loss(current_data["target"], 
                                                         current_data["prediction"])

    # Default rate shift
    performance_metrics["BadRate"] = current_data["target"].mean()

# ============================================================
# 7. PRODUCE MONITORING REPORT
# ============================================================

report = {
    "Feature_PSI": feature_psi,
    "DriftFlags": drift_flags,
    "KS_pvalues": feature_ks,
    "ChiSquare_pvalues": feature_chi,
    "Prediction_PSI": psi_prediction,
    "Performance": performance_metrics
}

print("\n====== MODEL MONITORING REPORT ======\n")
for section, values in report.items():
    print(section)
    print(values)
    print("-------------------------------------")


# ============================================================
#         PSI
# ============================================================
import numpy as np
import pandas as pd

def calculate_psi_debug(base, target, bins=10, bucket_type='quantile', eps=1e-6, print_debug=True):
    """
    Calculate PSI with diagnostics.
    - base, target: pandas Series or array-like
    - bins: int number of bins (for numeric variables)
    - bucket_type: 'quantile' or 'bins' (equal-width)
    - eps: tiny value to replace zeros
    - print_debug: if True, prints diagnostics
    Returns: total_psi (float), psi_per_bin (pd.Series), diagnostics (dict)
    """
    # Convert to Series, dropna
    base = pd.Series(base).dropna().reset_index(drop=True)
    target = pd.Series(target).dropna().reset_index(drop=True)

    if len(base) == 0 or len(target) == 0:
        raise ValueError("Base or target series is empty after dropping NA.")

    diagnostics = {}
    # If categorical-like (few unique values), treat separately
    if base.nunique() <= min(10, max(3, bins//2)):
        # treat as categorical
        cats = sorted(set(base.unique()) | set(target.unique()))
        base_counts = base.astype(object).value_counts().reindex(cats, fill_value=0)
        target_counts = target.astype(object).value_counts().reindex(cats, fill_value=0)
        base_perc = base_counts / base_counts.sum()
        target_perc = target_counts / target_counts.sum()

        # replace zeros
        base_perc = base_perc.replace(0, eps)
        target_perc = target_perc.replace(0, eps)

        psi_per_bin = (base_perc - target_perc) * np.log(base_perc / target_perc)
        total_psi = psi_per_bin.sum()

        diagnostics.update({
            "mode": "categorical",
            "categories": cats,
            "base_counts": base_counts,
            "target_counts": target_counts,
            "base_perc": base_perc,
            "target_perc": target_perc,
        })
    else:
        # Numeric handling: compute breakpoints from base
        if bucket_type == 'quantile':
            # quantiles (may collapse if many identical values)
            quantiles = np.linspace(0, 1, bins + 1)
            breakpoints = np.unique(np.quantile(base, quantiles))
            if len(breakpoints) <= 1:
                # fallback to small epsilon-based min/max
                min_v, max_v = base.min(), base.max()
                if min_v == max_v:
                    # extreme fallback: create trivial bins
                    breakpoints = np.array([min_v - 1e-9, max_v + 1e-9])
                else:
                    breakpoints = np.linspace(min_v, max_v, bins + 1)
            # if unique breakpoints less than bins+1, we still proceed (some bins will be merged)
        else:
            # equal width bins across base range
            min_v, max_v = base.min(), base.max()
            if min_v == max_v:
                breakpoints = np.array([min_v - 1e-9, max_v + 1e-9])
            else:
                breakpoints = np.linspace(min_v, max_v, bins + 1)

        # If breakpoints not strictly increasing (rare), expand tiny epsilon
        for i in range(1, len(breakpoints)):
            if breakpoints[i] <= breakpoints[i-1]:
                breakpoints[i] = breakpoints[i-1] + 1e-9

        # Bin both datasets using the base breakpoints
        base_bins = pd.cut(base, breakpoints, include_lowest=True)
        target_bins = pd.cut(target, breakpoints, include_lowest=True)

        base_counts = base_bins.value_counts(sort=False)
        target_counts = target_bins.value_counts(sort=False)

        base_perc = base_counts / len(base)
        target_perc = target_counts / len(target)

        # Save diagnostics
        diagnostics.update({
            "mode": "numeric",
            "breakpoints": breakpoints,
            "base_counts": base_counts,
            "target_counts": target_counts,
            "base_perc_before_replace": base_perc,
            "target_perc_before_replace": target_perc
        })

        # Replace zeros with eps to avoid log(0)
        base_perc = base_perc.replace(0, eps)
        target_perc = target_perc.replace(0, eps)

        psi_per_bin = (base_perc - target_perc) * np.log(base_perc / target_perc)
        total_psi = psi_per_bin.sum()

        diagnostics.update({
            "base_perc": base_perc,
            "target_perc": target_perc
        })

    # Print debugging info if requested
    if print_debug:
        print("=== PSI DIAGNOSTICS ===")
        print("Total PSI:", total_psi)
        for k, v in diagnostics.items():
            if k in ("base_counts", "target_counts", "base_perc", "target_perc", "base_perc_before_replace", "target_perc_before_replace"):
                print(f"\n{k}:")
                print(v)
            elif k == "breakpoints":
                print("\nbreakpoints:")
                print(v)
            elif k == "categories":
                print("\ncategories:")
                print(v)
            else:
                # small summary
                pass
        print("\npsi_per_bin:")
        print(psi_per_bin)
        print("========================\n")

    return float(total_psi), psi_per_bin, diagnostics


# Example helper to produce a PSI report for a DataFrame
def psi_report_debug(df_base, df_target, features=None, exclude=None, bins=10, bucket_type='quantile'):
    """
    Compute PSI for multiple features and return DataFrame with diagnostics.
    - features: list of columns to compute PSI for. If None, use intersection of both dfs minus exclude.
    - exclude: list of columns to skip (e.g., target)
    """
    if exclude is None:
        exclude = []
    if features is None:
        features = [c for c in df_base.columns if c in df_target.columns and c not in exclude]
    results = []
    diagnostics = {}
    for col in features:
        try:
            psi_val, psi_bins, diag = calculate_psi_debug(df_base[col], df_target[col], bins=bins, bucket_type=bucket_type, print_debug=False)
            results.append((col, psi_val))
            diagnostics[col] = diag
        except Exception as e:
            results.append((col, np.nan))
            diagnostics[col] = {"error": str(e)}
    psi_df = pd.DataFrame(results, columns=['feature', 'psi_value']).sort_values('psi_value', ascending=False).reset_index(drop=True)
    return psi_df, diagnostics
