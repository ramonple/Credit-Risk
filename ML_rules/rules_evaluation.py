import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


# =================================
#       Redundancy Analysis
# =================================
from matplotlib_venn import venn2, venn2_circles
import matplotlib.pyplot as plt

def two_rules_redundancy(
        rule1_total, rule1_bad,
        rule2_total, rule2_bad,
        rule12_both_total, rule12_both_bad,
        rule1_name, rule2_name
):

    only_r1_total = rule1_total - rule12_both_total
    only_r2_total = rule2_total - rule12_both_total

    venn2(
        subsets=(only_r1_total, only_r2_total, rule12_both_total),
        set_labels=(rule1_name, rule2_name),
        set_colors=('orange', 'blue'),
        alpha=0.6
    )
    venn2_circles(
        subsets=(only_r1_total, only_r2_total, rule12_both_total),
        linestyle='dashed',
        linewidth=1
    )
    plt.title('All Declined Accounts')
    plt.show()

    only_r1_bad = rule1_bad - rule12_both_bad
    only_r2_bad = rule2_bad - rule12_both_bad

    venn2(
        subsets=(only_r1_bad, only_r2_bad, rule12_both_bad),
        set_labels=(rule1_name, rule2_name),
        set_colors=('orange', 'blue'),
        alpha=0.6
    )
    venn2_circles(
        subsets=(only_r1_bad, only_r2_bad, rule12_both_bad),
        linestyle='dashed',
        linewidth=1
    )
    plt.title('Detected Bad Accounts')
    plt.show()


def three_rules_redundancy(
        r1_total, r1_bad,
        r2_total, r2_bad,
        r3_total, r3_bad,
        r12_total, r12_bad,
        r13_total, r13_bad,
        r23_total, r23_bad,
        r123_total, r123_bad,
        rule1_name, rule2_name, rule3_name
):
    """
    Visualize redundancy across 3 rules using Venn diagrams.
    
    All inputs represent counts:
      - r1_total: customers hit by Rule 1
      - r12_total: customers hit by both Rule 1 and Rule 2
      - r123_total: customers hit by Rules 1,2,3
      etc.
    Same structure for xxx_bad counts.
    """

    only_r1 = r1_total - r12_total - r13_total + r123_total
    only_r2 = r2_total - r12_total - r23_total + r123_total
    only_r3 = r3_total - r13_total - r23_total + r123_total

    only_r12 = r12_total - r123_total
    only_r13 = r13_total - r123_total
    only_r23 = r23_total - r123_total
    only_r123 = r123_total

    venn3(
        subsets = (
            only_r1,
            only_r2,
            only_r12,
            only_r3,
            only_r13,
            only_r23,
            only_r123
        ),
        set_labels=(rule1_name, rule2_name, rule3_name),
        set_colors=('orange','blue','green'),
        alpha=0.6
    )
    venn3_circles(
        subsets = (
            only_r1, only_r2, only_r12,
            only_r3, only_r13, only_r23, only_r123
        ),
        linestyle='dashed',
        linewidth=1
    )

    plt.title("All Declined Accounts")
    plt.show()


    only_r1_bad = r1_bad - r12_bad - r13_bad + r123_bad
    only_r2_bad = r2_bad - r12_bad - r23_bad + r123_bad
    only_r3_bad = r3_bad - r13_bad - r23_bad + r123_bad

    only_r12_bad = r12_bad - r123_bad
    only_r13_bad = r13_bad - r123_bad
    only_r23_bad = r23_bad - r123_bad
    only_r123_bad = r123_bad

    venn3(
        subsets = (
            only_r1_bad,
            only_r2_bad,
            only_r12_bad,
            only_r3_bad,
            only_r13_bad,
            only_r23_bad,
            only_r123_bad
        ),
        set_labels=(rule1_name, rule2_name, rule3_name),
        set_colors=('orange','blue','green'),
        alpha=0.6
    )
    venn3_circles(
        subsets = (
            only_r1_bad, only_r2_bad, only_r12_bad,
            only_r3_bad, only_r13_bad, only_r23_bad, only_r123_bad
        ),
        linestyle='dashed',
        linewidth=1
    )

    plt.title("Detected Bad Accounts")
    plt.show()


# =================================
#       New Baseline Analysis
# =================================
def group_analysis_table_one_rule(
        data,
        rule,               
        group_column,
        bad_flag,
        bal_variable,
        bad_bal_variable
):
    """
    Perform detailed group analysis before and after applying one rule.

    - rule: data[data['colume'] > x]
    """

    original = data.groupby(group_column).agg(
        original_volume     = (bad_flag, 'count'),
        original_bad_volume = (bad_flag, 'sum'),
        original_total_bal  = (bal_variable, 'sum'),
        original_bad_bal    = (bad_bal_variable, 'sum')
    )
    
    original['original_good_volume'] = (original['original_volume'] - original['original_bad_volume'])
    original['original_br_vol'] = (original['original_bad_volume'] / original['original_volume'] * 100)
    original['original_br_bal'] = (original['original_bad_bal'] / original['original_total_bal'] * 100)

    original = original.sort_values(by='original_volume', ascending=False)

    new_base = data[~rule].copy()

    new = new_base.groupby(group_column).agg(
        new_volume     = (bad_flag, 'count'),
        new_bad_volume = (bad_flag, 'sum'),
        new_total_bal  = (bal_variable, 'sum'),
        new_bad_bal    = (bad_bal_variable, 'sum')
    )

    new['new_good_volume'] = new['new_volume'] - new['new_bad_volume']
    new['new_br_vol'] = new['new_bad_volume'] / new['new_volume'] * 100
    new['new_br_bal'] = new['new_bad_bal'] / new['new_total_bal'] * 100

    new = new.sort_values(by='new_volume', ascending=False)


    entire = original.join(new, how='outer').fillna(0)


    entire['Declined Volume'] = entire['original_volume'] - entire['new_volume']
    entire['Declined Bads']   = entire['original_bad_volume'] - entire['new_bad_volume']
    entire['Declined Goods']  = entire['Declined Volume'] - entire['Declined Bads']
    entire['Declined Balance'] = entire['original_total_bal'] - entire['new_total_bal']
    entire['Declined Bad Balance'] = entire['original_bad_bal'] - entire['new_bad_bal']

    entire['Marginal BR Vol'] = np.where(
        entire['Declined Volume'] > 0,
        entire['Declined Bads'] / entire['Declined Volume'] * 100,
        0
    )
    entire['Marginal BR Bal'] = np.where(
        entire['Declined Balance'] > 0,
        entire['Declined Bad Balance'] / entire['Declined Balance'] * 100,
        0
    )

    entire['Volume_Reduc%'] = entire['Declined Volume'] / entire['original_volume'] * 100
    entire['Balance_Reduc%'] = entire['Declined Balance'] / entire['original_total_bal'] * 100
    entire['Bad Volume_Reduc%'] = entire['Declined Bads'] / entire['original_bad_volume'] * 100
    entire['Bad Balance_Reduc%'] = entire['Declined Bad Balance'] / entire['original_bad_bal'] * 100

    entire['GB'] = np.where(
        (entire['Declined Goods'] > 0) & (entire['original_good_volume'] > 0),
        (entire['Declined Bads'] / entire['original_bad_volume']) /
        (entire['Declined Goods'] / entire['original_good_volume']),
        np.nan
    )

    return entire





def plot_one_rule_before_after(result_table,
                               group_column,
                               before_bal="original_total_bal",
                               before_rate="original_br_bal",
                               after_bal="new_total_bal",
                               after_rate="new_br_bal",
                               title="Before vs After Rule"):
    """
    Compare Before/After rule performance:
    Bars = Balance
    Line = Bad Balance Rate
    """

    df = result_table.reset_index()
    x = range(len(df))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # --- Bar: Before & After Balance ---
    ax1.bar([p - width/2 for p in x], df[before_bal],
            width=width, color="grey", label="Before Balance")
    ax1.bar([p + width/2 for p in x], df[after_bal],
            width=width, color="skyblue", label="After Balance")

    ax1.set_ylabel("Balance", color="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df[group_column])

    # --- Line: Before & After Bad Balance Rate ---
    ax2 = ax1.twinx()
    ax2.plot(df[group_column], df[before_rate],
             marker='o', color="black", label="Before Bad Bal %")
    ax2.plot(df[group_column], df[after_rate],
             marker='o', color="red", label="After Bad Bal %")

    ax2.set_ylabel("Bad Bal %")

    plt.title(title)

    # Combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    plt.show()





def group_analysis_table_two_rules(
        data,
        rule1,                   
        rule2,                   
        group_column,
        bad_flag,
        bal_variable,
        bad_bal_variable
):
    """
    Compare group-level business impact of two rules independently 
    applied on the same original base.
    """

    original = data.groupby(group_column).agg(
        original_volume     = (bad_flag, 'count'),
        original_bad_volume = (bad_flag, 'sum'),
        original_total_bal  = (bal_variable, 'sum'),
        original_bad_bal    = (bad_bal_variable, 'sum')
    )

    original['original_good_volume'] = (original['original_volume'] - original['original_bad_volume'])
    original['original_br_vol'] = (original['original_bad_volume'] / original['original_volume'] * 100)
    original['original_br_bal'] = (original['original_bad_bal'] / original['original_total_bal'] * 100)

    new1_base = data[~rule1]

    new1 = new1_base.groupby(group_column).agg(
        r1_volume     = (bad_flag, 'count'),
        r1_bad_volume = (bad_flag, 'sum'),
        r1_total_bal  = (bal_variable, 'sum'),
        r1_bad_bal    = (bad_bal_variable, 'sum')
    )

    new1['r1_good_volume'] = new1['r1_volume'] - new1['r1_bad_volume']
    new1['r1_br_vol'] = new1['r1_bad_volume'] / new1['r1_volume'] * 100
    new1['r1_br_bal'] = new1['r1_bad_bal'] / new1['r1_total_bal'] * 100

    new2_base = data[~rule2]

    new2 = new2_base.groupby(group_column).agg(
        r2_volume     = (bad_flag, 'count'),
        r2_bad_volume = (bad_flag, 'sum'),
        r2_total_bal  = (bal_variable, 'sum'),
        r2_bad_bal    = (bad_bal_variable, 'sum')
    )

    new2['r2_good_volume'] = new2['r2_volume'] - new2['r2_bad_volume']
    new2['r2_br_vol'] = new2['r2_bad_volume'] / new2['r2_volume'] * 100
    new2['r2_br_bal'] = new2['r2_bad_bal'] / new2['r2_total_bal'] * 100

    entire = (
        original
        .join(new1, how='outer')
        .join(new2, how='outer')
        .fillna(0)
    )

    entire['r1_declined_volume'] = (entire['original_volume'] - entire['r1_volume'])
    entire['r1_declined_bads'] = (entire['original_bad_volume'] - entire['r1_bad_volume'])
    entire['r1_declined_goods'] = (entire['r1_declined_volume'] - entire['r1_declined_bads'])
    entire['r1_declined_bal'] = (entire['original_total_bal'] - entire['r1_total_bal'])
    entire['r1_declined_bad_bal'] = (entire['original_bad_bal'] - entire['r1_bad_bal'])

    entire['r1_marginal_br_vol'] = np.where(
        entire['r1_declined_volume'] > 0,
        entire['r1_declined_bads'] / entire['r1_declined_volume'] * 100,
        0
    )
    entire['r1_marginal_br_bal'] = np.where(
        entire['r1_declined_bal'] > 0,
        entire['r1_declined_bad_bal'] / entire['r1_declined_bal'] * 100,
        0
    )

    entire['r1_volume_reduc%'] = (entire['r1_declined_volume'] / entire['original_volume'] * 100)
    entire['r1_balance_reduc%'] = (entire['r1_declined_bal'] / entire['original_total_bal'] * 100)
    entire['r1_bad_vol_reduc%'] = (entire['r1_declined_bads'] / entire['original_bad_volume'] * 100)
    entire['r1_bad_bal_reduc%'] = (entire['r1_declined_bad_bal'] / entire['original_bad_bal'] * 100)

    entire['r1_GB'] = np.where(
        (entire['r1_declined_goods'] > 0) & (entire['original_good_volume'] > 0),
        (entire['r1_declined_bads'] / entire['original_bad_volume']) / 
        (entire['r1_declined_goods'] / entire['original_good_volume']),
        np.nan
    )

    entire['r2_declined_volume'] = (entire['original_volume'] - entire['r2_volume'])
    entire['r2_declined_bads'] = (entire['original_bad_volume'] - entire['r2_bad_volume'])
    entire['r2_declined_goods'] = (entire['r2_declined_volume'] - entire['r2_declined_bads'])
    entire['r2_declined_bal'] = (entire['original_total_bal'] - entire['r2_total_bal'])
    entire['r2_declined_bad_bal'] = (entire['original_bad_bal'] - entire['r2_bad_bal'])

    entire['r2_marginal_br_vol'] = np.where(
        entire['r2_declined_volume'] > 0,
        entire['r2_declined_bads'] / entire['r2_declined_volume'] * 100,
        0
    )
    entire['r2_marginal_br_bal'] = np.where(
        entire['r2_declined_bal'] > 0,
        entire['r2_declined_bad_bal'] / entire['r2_declined_bal'] * 100,
        0
    )

    entire['r2_volume_reduc%'] = (
        entire['r2_declined_volume'] / entire['original_volume'] * 100
    )
    entire['r2_balance_reduc%'] = (
        entire['r2_declined_bal'] / entire['original_total_bal'] * 100
    )
    entire['r2_bad_vol_reduc%'] = (
        entire['r2_declined_bads'] / entire['original_bad_volume'] * 100
    )
    entire['r2_bad_bal_reduc%'] = (
        entire['r2_declined_bad_bal'] / entire['original_bad_bal'] * 100
    )

    entire['r2_GB'] = np.where(
        (entire['r2_declined_goods'] > 0) & (entire['original_good_volume'] > 0),
        (entire['r2_declined_bads'] / entire['original_bad_volume']) / 
        (entire['r2_declined_goods'] / entire['original_good_volume']),
        np.nan
    )

    return entire


def plot_two_rules_before_after(result_table,
                                group_column,
                                r1_before_bal="r1_original_total_bal",
                                r1_before_rate="r1_original_br_bal",
                                r1_after_bal="r1_new_total_bal",
                                r1_after_rate="r1_new_br_bal",
                                r2_before_bal="r2_original_total_bal",
                                r2_before_rate="r2_original_br_bal",
                                r2_after_bal="r2_new_total_bal",
                                r2_after_rate="r2_new_br_bal",
                                title="Rule Comparison: Before vs After"):
    """
    Compare Before/After of two rules:
    Bars = Balance
    Lines = Bad Balance Rate
    """

    df = result_table.reset_index()
    x = range(len(df))
    width = 0.2   # smaller bars because 4 per group

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # ======= RULE 1 BARS =======
    ax1.bar([p - 1.5*width for p in x], df[r1_before_bal],
            width=width, color="grey", label="Rule 1 Before Bal")
    ax1.bar([p - 0.5*width for p in x], df[r1_after_bal],
            width=width, color="skyblue", label="Rule 1 After Bal")

    # ======= RULE 2 BARS =======
    ax1.bar([p + 0.5*width for p in x], df[r2_before_bal],
            width=width, color="lightgreen", label="Rule 2 Before Bal")
    ax1.bar([p + 1.5*width for p in x], df[r2_after_bal],
            width=width, color="orange", label="Rule 2 After Bal")

    ax1.set_ylabel("Balance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df[group_column])

    # ======= RULE 1 & RULE 2 LINES =======
    ax2 = ax1.twinx()

    ax2.plot(df[group_column], df[r1_before_rate],
             marker='o', color="black", label="Rule 1 Before BR")
    ax2.plot(df[group_column], df[r1_after_rate],
             marker='o', color="blue", label="Rule 1 After BR")

    ax2.plot(df[group_column], df[r2_before_rate],
             marker='o', color="green", label="Rule 2 Before BR")
    ax2.plot(df[group_column], df[r2_after_rate],
             marker='o', color="red", label="Rule 2 After BR")

    ax2.set_ylabel("Bad Bal %")

    plt.title(title)

    # --- Combined Legend ---
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    plt.show()



# ==============================================
#     Visualise the Group analysis results 
# ==============================================


from plotly.subplots import make_subplots
import plotly.graph_objects as go

def group_performance_one_rule(
    group_data,
    top_x,
    original_bal,
    new_bal,
    original_bad_bal,
    new_bad_bal,
    original_bal_br,
    new_bal_br
):
    """
    Produce two plots:
    
    Plot 1: Bad Balance + Balance BR (line)
    Plot 2: Total Balance + Balance BR (line)

    Parameters
    ----------
    group_data : pd.DataFrame
        Data indexed by group name.
    top_x : int
        Number of rows (after filtering Missing) to display.
    Column name parameters : str
        Names of relevant columns in the dataframe.
    """

    # Filter out Missing and take top N
    selected = group_data[group_data.index != 'Missing'].copy().head(top_x)

    # ==========================================
    #   PLOT 1: BAD BALANCE + BR
    # ==========================================
    fig1 = make_subplots(specs=[[{'secondary_y': True}]])

    fig1.add_trace(
        go.Bar(x=selected.index, y=selected[original_bad_bal],name='Original Bad Balance'),
        secondary_y=False
    )

    fig1.add_trace(
        go.Bar(x=selected.index,y=selected[new_bad_bal],name='New Bad Balance'),
        secondary_y=False
    )

    fig1.add_trace(
        go.Scatter(x=selected.index,y=selected[original_bal_br],mode='lines+markers',name='Original Bal BR'),
        secondary_y=True
    )

    fig1.add_trace(
        go.Scatter(x=selected.index,y=selected[new_bal_br],mode='lines+markers',name='New Bal BR'),
        secondary_y=True
    )

    fig1.update_layout(
        title='Bad Balance & Bal BR Changes',
        width=900,
        height=600,
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        xaxis_title='Group',
        yaxis_title='Bad Balance',
        template='plotly_white'
    )

    fig1.update_yaxes(
        title_text='Balance BR (%)',
        secondary_y=True
    )

    fig1.show()

    # ==========================================
    #   PLOT 2: TOTAL BALANCE + BR
    # ==========================================
    fig2 = make_subplots(specs=[[{'secondary_y': True}]])

    fig2.add_trace(
        go.Bar(x=selected.index,y=selected[original_bal],name='Original Balance'),
        secondary_y=False
    )

    fig2.add_trace(
        go.Bar(x=selected.index,y=selected[new_bal],name='New Balance'),
        secondary_y=False
    )

    fig2.add_trace(
        go.Scatter(x=selected.index,y=selected[original_bal_br],mode='lines+markers',name='Original Bal BR'),
        secondary_y=True
    )

    fig2.add_trace(
        go.Scatter(x=selected.index,y=selected[new_bal_br],mode='lines+markers',name='New Bal BR'),
        secondary_y=True
    )

    fig2.update_layout(
        title='Balance & Bal BR Changes',
        width=900,
        height=600,
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        xaxis_title='Group',
        yaxis_title='Balance',
        template='plotly_white'
    )

    fig2.update_yaxes(
        title_text='Balance BR (%)',
        secondary_y=True
    )

    fig2.show()


from plotly.subplots import make_subplots
import plotly.graph_objects as go

def group_performance_one_rule(
    group_data,
    top_x,
    original_bal,
    new_bal1,
    new_bal2,
    original_bad_bal,
    new_bad_bal1,
    new_bad_bal2,
    original_bal_br,
    new_bal_br1,
    new_bal_br2
):
    """
    Produce two plots:

    Plot 1: Bad Balance vs BR
    Plot 2: Total Balance vs BR

    Each plot contains:
        - 3 bars  (Original, New1, New2)
        - 3 lines (Original BR, New1 BR, New2 BR)
    """

    # -----------------------------------------
    # Filter top groups excluding 'Missing'
    # -----------------------------------------
    selected = group_data[group_data.index != 'Missing'].copy().head(top_x)

    # =====================================================
    #   PLOT 1 — BAD BALANCE
    # =====================================================
    fig1 = make_subplots(specs=[[{'secondary_y': True}]])

    # ---------- Bars ----------
    fig1.add_trace(go.Bar(x=selected.index, y=selected[original_bad_bal], name='Original Bad Bal'), secondary_y=False)
    fig1.add_trace(go.Bar(x=selected.index, y=selected[new_bad_bal1],   name='New Rule 1 Bad Bal'), secondary_y=False)
    fig1.add_trace(go.Bar(x=selected.index, y=selected[new_bad_bal2],   name='New Rule 2 Bad Bal'), secondary_y=False)

    # ---------- BR lines ----------
    fig1.add_trace(go.Scatter(x=selected.index, y=selected[original_bal_br], mode='lines+markers', name='Original BR'), secondary_y=True)
    fig1.add_trace(go.Scatter(x=selected.index, y=selected[new_bal_br1],   mode='lines+markers', name='New Rule 1 BR'), secondary_y=True)
    fig1.add_trace(go.Scatter(x=selected.index, y=selected[new_bal_br2],   mode='lines+markers', name='New Rule 2 BR'), secondary_y=True)

    fig1.update_layout(
        title='Bad Balance & BR Comparison (Original vs Rule1 vs Rule2)',
        width=900,
        height=600,
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title='Group',
        yaxis_title='Bad Balance',
        template='plotly_white'
    )
    fig1.update_yaxes(title_text='BR (%)', secondary_y=True)
    fig1.show()

    # =====================================================
    #   PLOT 2 — TOTAL BALANCE
    # =====================================================
    fig2 = make_subplots(specs=[[{'secondary_y': True}]])

    # ---------- Bars ----------
    fig2.add_trace(go.Bar(x=selected.index, y=selected[original_bal], name='Original Balance'), secondary_y=False)
    fig2.add_trace(go.Bar(x=selected.index, y=selected[new_bal1],     name='New Rule 1 Balance'), secondary_y=False)
    fig2.add_trace(go.Bar(x=selected.index, y=selected[new_bal2],     name='New Rule 2 Balance'), secondary_y=False)

    # ---------- BR lines ----------
    fig2.add_trace(go.Scatter(x=selected.index, y=selected[original_bal_br], mode='lines+markers', name='Original BR'), secondary_y=True)
    fig2.add_trace(go.Scatter(x=selected.index, y=selected[new_bal_br1],     mode='lines+markers', name='New Rule 1 BR'), secondary_y=True)
    fig2.add_trace(go.Scatter(x=selected.index, y=selected[new_bal_br2],     mode='lines+markers', name='New Rule 2 BR'), secondary_y=True)

    fig2.update_layout(
        title='Balance & BR Comparison (Original vs Rule1 vs Rule2)',
        width=900,
        height=600,
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title='Group',
        yaxis_title='Balance',
        template='plotly_white'
    )
    fig2.update_yaxes(title_text='BR (%)', secondary_y=True)
    fig2.show()




#==============================================
#    Rule Summary Table Builder
#==============================================

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


#    rules = {
#    "R1: high util": mask_r1,
#    "R2: dpd>=10": mask_r2,
#}
#summary = build_rule_summary_table(
#    data=df,
#    rules=rules,
#    rule_performance_simple_records=rule_performance_simple_records,
#    baseline=baseline_mask,
#    bad_flag_col="bad_flag",
#    balance_col="bad_bal",
#    performance_expects_mask=True,
#)