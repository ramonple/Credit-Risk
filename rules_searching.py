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
