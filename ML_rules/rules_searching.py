import pandas as pd
import numpy as np

import pandas as pd

def final_feature_for_rule_construction() -> pd.DataFrame:
    """
    Create an empty DataFrame template used for defining feature rules.

    Columns:
        - variable: Name of the feature
        - Valid Min: Minimum acceptable value of the feature
        - Valid Max: Maximum acceptable value of the feature
        - Search Min: Minimum value to search in rule optimization
        - Search Max: Maximum value to search in rule optimization
        - Step: Step size for searching
        - Direction: Increase or decrease direction for search
        - Type: Data type or business indicator category

    Returns:
        pd.DataFrame: An empty DataFrame with the predefined columns.
    """
    
    columns = [
        'variable',
        'Valid Min', 'Valid Max',
        'Search Min', 'Search Max',
        'Step',
        'Direction',
        'Type'
    ]

    return pd.DataFrame(columns=columns)


# ==================================
#          Rules Performance
# ==================================

import numpy as np

def rule_checking(
    data,
    rule,
    bad_flag,
    bal_variable,
    bad_bal_variable
):
    """
    Evaluate the performance impact of a single rule.

    Parameters
    ----------
    data : DataFrame
        Entire population.
    rule : DataFrame
        Subset of `data` that meets the rule condition.
    bad_flag : str
        Column indicating bad=1 or bad>0.
    bal_variable : str
        Column name for balance.
    bad_bal_variable : str
        Column name for bad balance.

    GB ratio = (Bad Removed % of total bads) / (Good Removed % of total goods)
    """

    # ===============================================================
    #  1. BASELINE (ORIGINAL) METRICS
    # ===============================================================
    total_volume = len(data)
    total_balance = data[bal_variable].sum()
    total_bad_balance = data[bad_bal_variable].sum()

    original_bad = data.loc[data[bad_flag] > 0]
    total_bad_volume = len(original_bad)

    # Baseline BRs
    original_br_vol = np.round((total_bad_volume / total_volume) * 100, 2)
    original_br_bal = np.round((total_bad_balance / total_balance) * 100, 2)

    # ===============================================================
    #  2. MARGINAL (RULE SELECTED) METRICS
    # ===============================================================
    marginal_volume = len(rule)
    marginal_balance = rule[bal_variable].sum()
    marginal_bad_balance = rule[bad_bal_variable].sum()

    marginal_bad = rule.loc[rule[bad_flag] > 0]
    marginal_bad_volume = len(marginal_bad)

    # Marginal BRs
    marginal_br_vol = np.round((marginal_bad_volume / marginal_volume) * 100, 2) if marginal_volume > 0 else 0
    marginal_br_bal = np.round((marginal_bad_balance / marginal_balance) * 100, 2) if marginal_balance > 0 else 0

    # Marginal percentages
    marginal_volume_pct = np.round((marginal_volume / total_volume) * 100, 2)
    marginal_balance_pct = np.round((marginal_balance / total_balance) * 100, 2)

    marginal_bad_volume_pct = np.round((marginal_bad_volume / total_bad_volume) * 100, 2) if total_bad_volume > 0 else 0
    marginal_bad_balance_pct = np.round((marginal_bad_balance / total_bad_balance) * 100, 2) if total_bad_balance > 0 else 0

    # Good:Bad ratio in marginal population
    total_good_volume = total_volume - total_bad_volume
    marginal_good_volume = marginal_volume - marginal_bad_volume

    # Avoid zero division
    bad_removed_pct = (marginal_bad_volume / total_bad_volume) if total_bad_volume > 0 else 0
    good_removed_pct = (marginal_good_volume / total_good_volume) if total_good_volume > 0 else 0

    gb_ratio = (
        np.round(bad_removed_pct / good_removed_pct, 2)
        if good_removed_pct > 0 else np.nan
    )

    # ===============================================================
    #  3. NEW BASELINE AFTER REMOVING RULE SELECTED POP
    # ===============================================================
    new_volume = total_volume - marginal_volume
    new_balance = total_balance - marginal_balance
    new_bad_volume = total_bad_volume - marginal_bad_volume
    new_bad_balance = total_bad_balance - marginal_bad_balance

    new_br_vol = np.round((new_bad_volume / new_volume) * 100, 2) if new_volume > 0 else 0
    new_br_bal = np.round((new_bad_balance / new_balance) * 100, 2) if new_balance > 0 else 0


    # ===============================================================
    #  PRINT RESULTS
    # ===============================================================

    print("\n==================== BASELINE PERFORMANCE ====================")
    print(f"Total Volume      : {total_volume}")
    print(f"Total Balance     : {total_balance:,.2f}")
    print(f"Total Bad Volume  : {total_bad_volume}")
    print(f"Total Bad Balance : {total_bad_balance:,.2f}")
    print(f"BR (Vol)          : {original_br_vol}%")
    print(f"BR (Bal)          : {original_br_bal}%")

    print("\n==================== RULE IMPACT (MARGINAL) ====================")
    print(f"Rule Volume              : {marginal_volume}   ({marginal_volume_pct}% of total)")
    print(f"Rule Balance             : {marginal_balance:,.2f}   ({marginal_balance_pct}% of total)")

    print(f"Rule Bad Volume          : {marginal_bad_volume}   ({marginal_bad_volume_pct}% of all bads)")
    print(f"Rule Bad Balance         : {marginal_bad_balance:,.2f}   ({marginal_bad_balance_pct}% of all bad balance)")

    print(f"Rule BR (Vol)            : {marginal_br_vol}%   (Baseline = {original_br_vol}%)")
    print(f"Rule BR (Bal)            : {marginal_br_bal}%   (Baseline = {original_br_bal}%)")

    print(f"Bad:Good Ratio    : {gb_ratio}")

    print("\n==================== NEW BASELINE AFTER RULE ====================")
    print(f"New Volume        : {new_volume}")
    print(f"New Balance       : {new_balance:,.2f} ({new_balance/100000:,.2f}M)")
    print(f"New Bad Volume    : {new_bad_volume:,.2f} ({new_bad_volume/1000:,.2f})")
    print(f"New Bad Balance   : {new_bad_balance:,.2f}")
    print(f"New BR (Vol)      : {new_br_vol}%")
    print(f"New BR (Bal)      : {new_br_bal}%")

    print("\n===============================================================\n")





def combine_checking_gb_ratio(
    data,
    min_bads,
    rule,
    bad_flag,
    bal_variable,
    bad_bal_variable
):
    """
    Quickly compute GB ratio for a rule.

    GB ratio = (Bad Removed % of total bads) / (Good Removed % of total goods)

    If the rule removes fewer bads than `min_bads`, returns np.nan.

    """

    # -------------------------
    # Total population stats
    # -------------------------
    total_volume = len(data)
    total_bad_volume = (data[bad_flag] > 0).sum()
    total_good_volume = total_volume - total_bad_volume

    if total_volume == 0 or total_bad_volume == 0:
        return np.nan

    # -------------------------
    # Rule statistics
    # -------------------------
    marginal_volume = len(rule)
    marginal_bad_volume = (rule[bad_flag] > 0).sum()
    marginal_good_volume = marginal_volume - marginal_bad_volume

    # Rule too weak → return NaN
    if marginal_bad_volume < min_bads:
        return np.nan

    # -------------------------
    # Core percentages
    # -------------------------
    bad_removed_pct = marginal_bad_volume / total_bad_volume
    good_removed_pct = marginal_good_volume / total_good_volume if total_good_volume > 0 else np.nan

    # If no good removed → infinite improvement
    if good_removed_pct == 0:
        return np.inf

    # -------------------------
    # GB ratio
    # -------------------------
    gb_ratio = bad_removed_pct / good_removed_pct

    return np.round(gb_ratio, 4)


# Template usage:

# results = []

# for i in range(1, 5):          # loop over threshold 1
#     for j in range(20, 40):    # loop over threshold 2
        
#         rule = data[(data['x'] > i) & (data['y'] > j)]
        
#         gb = combine_checking_gb_ratio(
#             data=data,
#             min_bads=50,
#             rule=rule,
#             bad_flag='bad_flag',
#             bal_variable='balance',
#             bad_bal_variable='bad_balance'
#         )
        
#         results.append([i, j, gb])

# results = pd.DataFrame(results, columns=['Variable1', 'Variable2', 'GB'])

# results = results.sort_values(by='GB', ascending=False)

# print(results)



import numpy as np

def combine_checking_bal_br_times(
    data,
    min_bads,
    rule,
    bad_flag,
    bal_variable,
    bad_bal_variable
):
    """
    Quickly compute the ratio of rule marginal bad balance rate over baseline bad balance rate.

    Ratio = (Rule Bad Balance / Rule Total Balance) / (Baseline Bad Balance / Baseline Total Balance)

    If the rule removes fewer bads than `min_bads`, returns np.nan.

    """

    # -------------------------
    # Baseline metrics
    # -------------------------
    total_balance = data[bal_variable].sum()
    total_bad_balance = data[bad_bal_variable].sum()

    baseline_rate = total_bad_balance / total_balance if total_balance > 0 else np.nan

    # -------------------------
    # Rule metrics
    # -------------------------
    marginal_bad_volume = (rule[bad_flag] > 0).sum()
    if marginal_bad_volume < min_bads:
        return np.nan

    rule_balance = rule[bal_variable].sum()
    rule_bad_balance = rule[bad_bal_variable].sum()
    marginal_rate = rule_bad_balance / rule_balance if rule_balance > 0 else np.nan

    # -------------------------
    # Ratio
    # -------------------------
    if baseline_rate == 0:
        return np.nan  
    ratio = marginal_rate / baseline_rate

    return np.round(ratio, 4)


# Template usage:

# results = []

# for i in range(1, 5):          # loop over threshold 1
#     for j in range(20, 40):    # loop over threshold 2
        
#         rule = data[(data['x'] > i) & (data['y'] > j)]
        
#         bal_br_times = combine_checking_bal_br_times(
#             data=data,
#             min_bads=50,
#             rule=rule,
#             bad_flag='bad_flag',
#             bal_variable='balance',
#             bad_bal_variable='bad_balance'
#         )
        
#         results.append([i, j, bal_br_times])

# results = pd.DataFrame(results, columns=['Variable1', 'Variable2', 'BR Bal Times'])

# results = results.sort_values(by='BR Bal Times', ascending=False)

# print(results)




#=================================
#      Rules Summary Table
#=================================
import numpy as np
import pandas as pd
from typing import Callable, Dict, Iterable, Optional, Union, Any

def build_rule_summary_table(
    data: pd.DataFrame,
    rules: Union[Dict[str, Any], Iterable[Any]],
    rule_performance_simple_records: Callable[..., tuple],
    *,
    baseline: Optional[Any] = None,
    baseline_name: str = "BASELINE",
    bad_flag_col: str = "bad_flag",
    balance_col: Optional[str] = None,
    rule_to_mask: Optional[Callable[[pd.DataFrame, Any], pd.Series]] = None,
    performance_expects_mask: bool = True,
) -> pd.DataFrame:
    """
    Build a policy summary table where each row represents:
      - BASELINE (row 1)
      - BASELINE + rule_i (row i)

    The wrapper calls `rule_performance_simple_records` to get:
      total vol removed, bad vol removed, total balance removed, bad balance removed, G:B ratio

    Columns:
      rule name, volume, bad volume, bad volume reduced, bad vol%, total bad,
      total bad reduced%, bad balance, bad balance reduced%, bad bal%, G:B

    Parameters
    ----------
    data:
        Input dataframe.
    rules:
        Ad-hoc rules to evaluate. Either:
          - dict {rule_name: rule_object}
          - iterable of rule_objects (will auto-name Rule_001, Rule_002, ...)
    rule_performance_simple_records:
        Your function that returns:
          (total_vol_removed, bad_vol_removed, total_bal_removed, bad_bal_removed, gb_ratio)
    baseline:
        Baseline policy. Can be:
          - a mask (pd.Series bool)
          - a rule object (if rule_to_mask provided)
          - None (meaning baseline removes nothing)
    bad_flag_col:
        Column in data indicating bad (1) vs good (0).
    balance_col:
        Optional balance/exposure column. Used only to compute totals; the removed balances
        come from your performance function.
    rule_to_mask:
        Function (data, rule) -> boolean mask indicating records removed/flagged by rule.
        Required if your rules/baseline are not already boolean masks.
    performance_expects_mask:
        If True: wrapper calls rule_performance_simple_records(data, removed_mask)
        If False: wrapper calls rule_performance_simple_records(data, rule_obj)
    """

    if bad_flag_col not in data.columns:
        raise ValueError(f"bad_flag_col '{bad_flag_col}' not found in data columns")

    # ---- normalise rules into (name, obj) pairs ----
    if isinstance(rules, dict):
        rule_items = list(rules.items())
    else:
        rule_items = [(f"Rule_{i:03d}", r) for i, r in enumerate(list(rules), start=1)]

    # ---- totals for % columns ----
    is_bad = data[bad_flag_col].astype(int) == 1
    is_good = ~is_bad
    total_bad = int(is_bad.sum())
    total_good = int(is_good.sum())

    if total_bad == 0:
        raise ValueError("total_bad is 0; cannot compute bad-related percentages.")
    if total_good == 0:
        # GB ratio formula needs goods in denominator
        # We'll still compute other metrics; GB may be NaN/inf depending on your function.
        pass

    total_bad_balance = None
    if balance_col is not None:
        if balance_col not in data.columns:
            raise ValueError(f"balance_col '{balance_col}' not found in data columns")
        total_bad_balance = data.loc[is_bad, balance_col].sum()
        if total_bad_balance == 0:
            total_bad_balance = None  # avoid divide-by-zero

    # ---- helper: get mask for a "policy" ----
    def _to_mask(rule_or_mask: Any) -> pd.Series:
        if rule_or_mask is None:
            return pd.Series(False, index=data.index)

        if isinstance(rule_or_mask, pd.Series):
            return rule_or_mask.reindex(data.index).fillna(False).astype(bool)

        if rule_to_mask is None:
            raise ValueError(
                "rule_to_mask must be provided when baseline/rules are not boolean masks."
            )
        m = rule_to_mask(data, rule_or_mask)
        if not isinstance(m, pd.Series):
            m = pd.Series(m, index=data.index)
        return m.reindex(data.index).fillna(False).astype(bool)

    # ---- helper: call your performance function ----
    def _perf(rule_or_mask: Any, mask: Optional[pd.Series] = None) -> tuple:
        if performance_expects_mask:
            # expects (data, removed_mask)
            if mask is None:
                mask = _to_mask(rule_or_mask)
            return rule_performance_simple_records(data, mask)
        else:
            # expects (data, rule_obj)
            return rule_performance_simple_records(data, rule_or_mask)

    # ---- baseline metrics ----
    baseline_mask = _to_mask(baseline)
    base_total_vol, base_bad_vol, base_total_bal, base_bad_bal, base_gb = _perf(baseline, baseline_mask)

    rows = []
    rows.append({
        "rule name": baseline_name,
        "volume": base_total_vol,
        "bad volume": base_bad_vol,
        "bad volume reduced": base_bad_vol,
        "bad vol%": (base_bad_vol / total_bad) if total_bad else np.nan,
        "total bad": total_bad,
        "total bad reduced%": (base_bad_vol / total_bad) if total_bad else np.nan,
        "bad balance": base_bad_bal,
        "bad balance reduced%": (base_bad_bal / total_bad_balance) if total_bad_balance else np.nan,
        "bad bal%": (base_bad_bal / total_bad_balance) if total_bad_balance else np.nan,
        "G:B": np.nan,  # baseline doesn't have incremental GB
    })

    # ---- evaluate each rule as baseline ∪ rule ----
    for rule_name, rule_obj in rule_items:
        rule_mask = _to_mask(rule_obj)
        policy_mask = baseline_mask | rule_mask

        pol_total_vol, pol_bad_vol, pol_total_bal, pol_bad_bal, _ = _perf(policy_mask, policy_mask)

        # Incremental GB for the rule on top of baseline, per your formula:
        # (Δbad/total_bad) / (Δgood/total_good)
        delta_bad = int((policy_mask & is_bad).sum()) - int((baseline_mask & is_bad).sum())
        delta_good = int((policy_mask & is_good).sum()) - int((baseline_mask & is_good).sum())

        if total_good == 0 or delta_good <= 0:
            gb = np.inf if delta_bad > 0 else np.nan
        else:
            gb = (delta_bad / total_bad) / (delta_good / total_good)

        rows.append({
            "rule name": rule_name,
            "volume": pol_total_vol,
            "bad volume": pol_bad_vol,
            "bad volume reduced": pol_bad_vol,
            "bad vol%": (pol_bad_vol / total_bad) if total_bad else np.nan,
            "total bad": total_bad,
            "total bad reduced%": (pol_bad_vol / total_bad) if total_bad else np.nan,
            "bad balance": pol_bad_bal,
            "bad balance reduced%": (pol_bad_bal / total_bad_balance) if total_bad_balance else np.nan,
            "bad bal%": (pol_bad_bal / total_bad_balance) if total_bad_balance else np.nan,
            "G:B": gb,
        })

    out = pd.DataFrame(rows)

    # Optional: nicer rounding (keep raw numbers)
    # out["bad vol%"] = out["bad vol%"].round(4)
    # out["total bad reduced%"] = out["total bad reduced%"].round(4)
    # out["bad balance reduced%"] = out["bad balance reduced%"].round(4)
    # out["bad bal%"] = out["bad bal%"].round(4)
    # out["G:B"] = out["G:B"].replace([np.inf], np.nan).round(4)

    return out


#rules = {
#    "R1: high util": mask_r1,
#    "R2: dpd>=10": mask_r2,
#}