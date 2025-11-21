import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================================= 
#       categorical & numerical features split 
# =============================================
def num_cat_list(data, bad_flag, bal_variable, bad_bal_variable):
    """
    This function is used to split the categorical & numerical features.
    variables represent bad flag, total balance and bad balance will be excluded.
    """
    num_list = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
    cat_list = [col for col in data.columns if pd.api.types.is_object_dtype(data[col])]
    
    items_to_remove = [bad_flag, bal_variable, bad_bal_variable]
    num_list = [col for col in num_list if col not in items_to_remove]

    return num_list, cat_list

# ============================================= 
#       information value calcualtion 
# =============================================
def information_value_calculation(data,bad_flag, num_list, cat_list):
    """
    Calculate IV (Information Value) for numeric and categorical variables.
    """
    names = []
    scores = []

    for col in num_list:
        target_data = data[[bad_flag, col]]
        target_data = target_data.copy()
        target_data['account']=1

        decs = target_data.groupby(pd.qcut(target_data[col],10,duplicates = 'drop')).sum()
        decs['positive_rate'] = (decs[bad_flag]/decs['account'])*100
        decs['negative_rate'] = 100 - decs['positive_rate']
        decs['negative_volume'] = decs['account'] - decs[bad_flag]
        decs['DB'] = decs[bad_flag]/sum(decs[bad_flag])
        decs['DG'] = decs['negative_volume']/sum(decs['negative_volume'])
        decs['WoE'] = np.log(decs['DG']/decs['DB'])
        decs['DG-DB']=decs['DG']-decs['DB']
        decs['multi'] = decs['DG-DB'] * decs['WoE']
        # we might get some infs, just remove them
        decs.replace([np.inf,-np.inf],np.nan, inplace=True)
        decs.dropna(subset=['multi'],how='all',inplace=True)
        
        score = sum(decs['multi'])
        scores.append(score)
        names.append(col)

    for col in cat_list:
        contingency_table = pd.crosstab(data[col], data[bad_flag]).copy()
        contingency_table.columns=['good','bad']
        contingency_table['good_dist'] = contingency_table['good']/contingency_table['good'].sum()
        contingency_table['bad_dist'] = contingency_table['bad']/contingency_table['bad'].sum()
        contingency_table['good_dist'].replace(0, 1e-6, inplace=True)
        coningency_table['bad_dist'].replace(0, 1e-6, inplace=True)
        contingency_table['WoE'] = np.log(contingency_table['bad_dist']/contingency_table['good_dist'])
        contingency_table['WoE'].fillna(0,inplace=True)
        contingency_table['IV'] = (contingency_table['good_dist']-contingency_table['bad_dist']) *contingency_table['WoE']
        score = sum(contingency_table['IV'])
        score = 0 if pd.isna(score) else score
        scores.append(score)
        names.append(col)
    result_df = pd.DataFrame({'Variable':names, 'Information Value': scores})
    result_df = result_df.sort_values(by='Information Value', ascending=False)
    return result_df


def information_value_calculation_Dictionary(data,bad_flag, num_list, cat_list, data_dictionary=None):
    """
    Calculate IV (Information Value) for numeric and categorical variables.
    Data dictionary is used to get the corresponding info.
    """
    names = []
    scores = []
    definitions = []

    for col in num_list:
        target_data = data[[bad_flag, col]]
        target_data = target_data.copy()
        target_data['account']=1

        decs = target_data.groupby(pd.qcut(target_data[col],10,duplicates = 'drop')).sum()
        decs['positive_rate'] = (decs[bad_flag]/decs['account'])*100
        decs['negative_rate'] = 100 - decs['positive_rate']
        decs['negative_volume'] = decs['account'] - decs[bad_flag]
        decs['DB'] = decs[bad_flag]/sum(decs[bad_flag])
        decs['DG'] = decs['negative_volume']/sum(decs['negative_volume'])
        decs['WoE'] = np.log(decs['DG']/decs['DB'])
        decs['DG-DB']=decs['DG']-decs['DB']
        decs['multi'] = decs['DG-DB'] * decs['WoE']
        # we might get some infs, just remove them
        decs.replace([np.inf,-np.inf],np.nan, inplace=True)
        decs.dropna(subset=['multi'],how='all',inplace=True)
        
        score = sum(decs['multi'])
        scores.append(score)
        names.append(col)

    for col in cat_list:
        contingency_table = pd.crosstab(data[col], data[bad_flag]).copy()
        contingency_table.columns=['good','bad']
        contingency_table['good_dist'] = contingency_table['good']/contingency_table['good'].sum()
        contingency_table['bad_dist'] = contingency_table['bad']/contingency_table['bad'].sum()
        contingency_table['good_dist'].replace(0, 1e-6, inplace=True)
        coningency_table['bad_dist'].replace(0, 1e-6, inplace=True)
        contingency_table['WoE'] = np.log(contingency_table['bad_dist']/contingency_table['good_dist'])
        contingency_table['WoE'].fillna(0,inplace=True)
        contingency_table['IV'] = (contingency_table['good_dist']-contingency_table['bad_dist']) *contingency_table['WoE']
        score = sum(contingency_table['IV'])
        score = 0 if pd.isna(score) else score
        scores.append(score)
        names.append(col)
    result_df = pd.DataFrame({'Variable':names, 'Information Value': scores})
    result_df = result_df.sort_values(by='Information Value', ascending=False)

    if data_dictionary is not None:
        result_df['cleaned_var'] = result_df['Variable'].apply(lambda x: x.split('_')[-1] if '_' in x else x)
        result_df = result_df.merge(data_dictionary,left_on ='cleaned_var',right_on='Variable', how='left',suffixes=('','_dict'))
        result_df[['Variable','Information Value','Definition','Valid Min','Valid Max','Direction']]
    else:
        result_df[['Variable','Information Value']]
                    
    return result_df

def plot_information_value(information_value_table, top_x):
    """
     Plot the top_x variables by Information Value.
    """
    iv_table = (information_value_table[['Variable','Information Value']].sort_values('Information Value', ascending=False).head(top_x).copy())

    
    fig = go.Figure(data=[
        go.Bar(
             x=iv_table['Variable']
            ,y=iv_table['Information Value']
            ,text=iv_table['Information Value'].round(2)
            ,textposition='auto'
            ,marker_color='indigo'
        )
    ])
    fig.update_layout(
         height=500
        ,width=max(400,top_x*40)
        ,title='Information Value'
        ,xaxis_title='Variable'
        ,yaxis_title='Information Value'
        ,template='plotly_white'
        ,xaxis=dict(tickangle=90)
    )
    fig.show()
    return fig

def iv_survived(iv_table, iv_threshold, num_list, cat_list):
    """
    Return features that pass IV filter and counts.
    """
    # keep only variables meeting threshold
    filter_iv = iv_table[iv_table['Information Value'] >= iv_threshold]

    survived_iv_list = set(filter_iv['Variable'])

    # count by type
    num_iv_survived = sum(var in survived_iv_list for var in num_list)
    cat_iv_survived = sum(var in survived_iv_list for var in cat_list)
    total_iv_survived = num_iv_survived + cat_iv_survived

    return survived_iv_list, num_iv_survived, cat_iv_survived, total_iv_survived


# ============================================= 
#       Correlation Analysis
# =============================================

def correlation_with_target(data, variable_list, bad_flag):
    """
    Calculate the correlation between variables and target (bad_flag)
    Tips: variable list should only contain numberical features
    """
    numeric_vars = [v for v in variable_list if np.issubdtype(data[v].dtype, np.number)]
    correlation_with_bad = data[numeric_vars].corrwith(data[bad_flag])
    correlation_with_bad = correlation_with_bad.reset_index()
    correlation_with_bad.columns=['Variable','Correlation']
    correlation_with_bad['Abs Corr'] = np.abs(correlation_with_bad['Correlation'])
    correlation_with_bad = correlation_with_bad.sort_values(by='Abs Corr', ascending=False)
    return correlation_with_bad

def correlation_between_features(data,variable_list):
    """
    Caulate the correlation between features
    """
    valid_vars = data.columns.intersection(variable_list)
    corr_df = data[valid_vars].corr()
    return corr_df


def get_highly_corr(corr_matrix, corr_threshold):
    """
    Get pairs of features with absolute correlation above the threshold
    """
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1))
    highly_correlated_pairs = upper.stack().reset_index()
    highly_correlated_pairs.columns =['Variable 1','Variable 2','Correlation']
    highly_correlated_pairs = highly_correlated_pairs[
        highly_correlated_pairs['Correlation'].abs() > corr_threshold
    ]
    highly_correlated_pairs = highly_correlated_pairs.reset_index(drop=True)

    return highly_correlated_pairs

def top_iv_correlation(iv_table, corr_matrix, top_x, figsize=(8,6),print_num=True):
    """
    Plot the heatmap of the correlation matrix for the top x features by IV
    """
    top_features = iv_table.sort_values(by='Information Value',ascending=False).head(top_x)['Variable'].tolist()
    top_features = [f for f in top_features if f in corr_matrix.columns]
    if len(top_features) == 0: raise ValueError('None of top features by IV found in correlation Matrix !')

    filter_corr = corr_matrix.loc[top_features, top_features]
    if print_num == True:
        fig = px.imshow(filter_corr, text_auto='.2f', color_continuous_scale='RdBu_r',zmin=-1,zmax=1,
        title=f'Correlation Heatmap (top {len(top_features)} by IV)'
        )
    else:
        fig = px.imshow(filter_corr,color_continuous_scale='RdBu_r',zmin=-1,zmax=1,
        title=f'Correlation Heatmap (top {len(top_features)} by IV)'
        )
    fig.update_layout(
        width=int(figsize[0]*100),
        height=int(figsize[1]*100),
        xaxis=dict(tickangle=90)
    )
    fig.show()


def filter_iv_corr(iv_table, highly_correlated_pairs, iv_threshold):
    """
    Filter features based on IV and correlation.
    For each highly correlated pair, drop the one with lower IV.
    """

    iv_dict = iv_table.set_index('Variable')['Information Value'].to_dict()
    
    drop_variables = set()
    
    for _, row in highly_correlated_pairs.iterrows():
        var1, var2 = row['Variable 1'], row['Variable 2']
        if var1 not in iv_dict or var2 not in iv_dict:
            continue
        
        iv1, iv2 = iv_dict[var1], iv_dict[var2]      
        if iv1 >= iv2:
            drop_variables.add(var2)
        else:
            drop_variables.add(var1)
    

    final_variables = sorted(
        [var for var in iv_dict.keys() if var not in drop_variables and iv_dict[var] > iv_threshold],
        key=lambda x: iv_dict[x],
        reverse=True
    )
    
    return final_variables


def corr_survived(iv_table, highly_correlated_pairs, num_list, cat_list, iv_threshold):
    """
    Count the number of features survived after IV & Correlation filters
    """
    iv_dict = iv_table.set_index('Variable')['Information Value'].to_dict()
    
    drop_variables = set()
    
    for _, row in highly_correlated_pairs.iterrows():
        var1, var2 = row['Variable 1'], row['Variable 2']
        if var1 not in iv_dict or var2 not in iv_dict:
            continue
        
        iv1, iv2 = iv_dict[var1], iv_dict[var2]      
        if iv1 >= iv2:
            drop_variables.add(var2)
        else:
            drop_variables.add(var1)
    
    final_variables = sorted(
        [var for var in iv_dict.keys() if var not in drop_variables and iv_dict[var] > iv_threshold],
        key=lambda x: iv_dict[x],
        reverse=True
    )
    
    num_set = set(num_list)
    cat_set = set(cat_list)
    
    num_sur_corr = len([var for var in final_variables if var in num_set])
    cat_sur_corr = len([var for var in final_variables if var in cat_set])
    total_sur_corr = num_sur_corr + cat_sur_corr
    
    return num_sur_corr, cat_sur_corr, total_sur_corr

# =================================================
#      Multicollinearity Analysis - VIF
# =================================================


def calculate_vif(data, bad_flag, bal_variable, bad_bal_variable, vif_threshold, drop_high_vif=True):
    """
    Calculate Variance Inflation Factor (VIF) for each numeric feature.
    """
    exclude_columns = [bad_flag, bal_variable, bad_bal_variable]
    numerical_cols = data.select_dtypes(include = [np.number]).columns.tolist()
    predictor_cols = [c for c in numerical_cols if c not in exclude_columns]

    if not predictor_cols:
        raise ValueError('No Numerical predictor columns available !')
    
    X = data[predictor_cols].copy()
    remaining_features = X.columns.tolist()

    while True:
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
        if drop_high_vif and vif_data['VIF'].max() > vif_threshold:
            drop_feature = vif_data.sort_values(by='VIF', ascending=False)['Feature'].iloc[0]
            X.drop(columns=[drop_feature], inplace=True)
            remaining_features.remove(drop_feature)
        else:
            break
    return vif_data, remaining_features

# ============================================= 
#      logic checks for Numerical features
# =============================================
def distribution_by_decile_logic(data, target_variable, bad_flag, bal_variable, bad_bal_variable, data_dictionary=None):
    """
    Perform logic checks for numerical features
    """
    baseline_br_vol = (len(data[data[bad_flag]==1])/len(data))
    baseline_br_bal = data[bad_bal_variable].sum() / data[bal_variable].sum()
    
    if data[target_variable].nunique() < 3: return
    missing_percent = (data[target_variable]==0).mean()
    if missing_percent > 0.99: return

    initial_row = data.shape[0]
    target_variable_band = target_variable +'_decile'

    cleaned_var = ''
    cleaned_var = target_variable.split('_')[-1] if '_' in target_variable else target_variable

    min_value, max_value,direction, definition = -9999,99999,0,'N/A'

    if data_dictionary is not None and cleaned_var in data_dictionary['Variable'].values:
        row = data_dictionary[data_dictionary['Variable']==cleaned_var].iloc[0]
        min_value = row['Valid Min'] if pd.notnull(row['Valid Min']) else -9999
        max_value = row['Valid Max'] if pd.notnull(row['Valid Max']) else 99999
        direction = row['Direction'] if pd.notnull(row['Direction']) else 0
        definition = row['Definition'] if pd.notnull(row['Definition']) else 'N/A'

    min_value, max_value, direction  = float(min_value), float(max_value), float(direction)
    missing_group = data[(data[target_variable] < min_value) | (data[target_variable] > max_value)]
    missing_group_bad = missing_group[missing_group[bad_flag]>0]
    br_missing_group = ((missing_group[bad_flag].mean())*100).round(2)
    missing_percent = ((missing_group.shape[0]/data.shape[0])*100).round(2)
    missing_over_baseline = ((br_missing_group/baseline_br_vol)*100).round(2)
    print(f'Missing Vol% is {missing_percent}%, BR for Missing Group is {br_missing_group}%, which is {missing_over_baseline} over baseline BR.')

    data = data[ (data[target_variable]>= min_value) & (data[target_variable]<= max_value)]
    data[target_variable_band] = pd.qcut(data[target_variable], q=10, duplicates='drop')

    data = data[[target_variable,bad_flag,bal_variable,bad_bal_variable]].copy()
    if len(data) < 5:
        return
    results = data.groupby(target_variable_band).agg(
        {target_variable_band:'count'
        ,bad_flag:['sum','mean']
        ,bal_variable:'sum'
        ,bad_bal_variable:'sum'}
    )
    results.columns =['Total Volume','Bad Volume','Bad Vol%','Total Balance','Bad Balance']
    results['Bad Vol%'] = results['Bad Vol%'] * 100
    results['Bad Bal%'] = (results['Bad Balance']/results['Total Balance'])*100


    smallest_group_br, largest_group_br = results['Bad Vol%'].iloc[0],results['Bad Vol%'].iloc[-1]
    if smallest_group_br > largest_group_br and direction == -1:
        print(f'{target_variable}: Counterintuitive - Lower Value Has Higher Bad Rate')
        return
    elif smallest_group_br < largest_group_br and direction == 1:
        print(f'{target_variable}: Counterintuitive - Larger Value Has Higher Bad Rate')
        return     

    results_vol = results[['Bad Vol%']]
    results_vol.plot.bar()
    plt.axhline(y=baseline_br_vol*100, linestyle='dotted', color='red', label='Baseline BR Vol')
    plt.ylabel('Bad Vol%')
    plt.title(f'Bad Vol% by {target_variable}')
    plt.legend()
    plt.show()

    results_bal = results[['Bad Bal%']]
    results_bal.plot.bar()
    plt.axhline(y=baseline_br_bal*100, linestyle='dotted', color='red', label='Baseline BR Bal')
    plt.ylabel('Bad Bal%')
    plt.title(f'Bad Ball% by {target_variable}')
    plt.legend()
    plt.show()



def distribution_by_group(data, target_variable, bad_flag, bal_variable, bad_bal_variable, data_dictionary=None):
    """
    Plot the distribution of Total Volume / Bad Vol% and Total Balance / Bad Bal% for a categorical variable
    """
    data = data[[target_variable, bad_flag, bal_variable, bad_bal_variable]].copy()
    data['volume'] = 1

    group = data.groupby(target_variable).agg(
        Total_Volume=('volume', 'sum'),
        Bad_Volume=(bad_flag, 'sum'),
        Total_Balance=(bal_variable, 'sum'),
        Bad_Balance=(bad_bal_variable, 'sum')
    )
    group['Bad Vol%'] = (group['Bad_Volume'] / group['Total_Volume'] * 100).round(3)
    group['Bad Bal%'] = (group['Bad_Balance'] / group['Total_Balance'] * 100).round(3)

    # Optional: print definition from data dictionary
    cleaned_var = target_variable.split('_')[-1] if '_' in target_variable else target_variable
    if data_dictionary is not None and cleaned_var in data_dictionary['Variable'].values:
        row = data_dictionary[data_dictionary['Variable'] == cleaned_var].iloc[0]
        definition = row['Definition'] if pd.notnull(row['Definition']) else 'N/A'
        print(target_variable, ":", definition)

    # --- Bad Volume plot ---
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.bar(group.index, group['Total_Volume'], color='skyblue', label='Total Volume')
    ax1.set_xlabel(target_variable)
    ax1.set_ylabel('Total Volume', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(group.index, group['Bad Vol%'], color='red', marker='o', label='Bad Vol%')
    ax2.set_ylabel('Bad Vol%', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    baseline_br_vol = (len(data[data[bad_flag]==1]) / len(data)) * 100
    ax2.axhline(y=baseline_br_vol, linestyle='dotted', color='orange', label='Baseline')
    
    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'{target_variable} - Bad Volume Distribution')
    plt.show()

    # --- Bad Balance plot ---
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.bar(group.index, group['Total_Balance'], color='skyblue', label='Total Balance')
    ax1.set_xlabel(target_variable)
    ax1.set_ylabel('Total Balance', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(group.index, group['Bad Bal%'], color='red', marker='o', label='Bad Bal%')
    ax2.set_ylabel('Bad Bal%', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    baseline_br_bal = (data[bad_bal_variable].sum() / data[bal_variable].sum()) * 100
    ax2.axhline(y=baseline_br_bal, linestyle='dotted', color='orange', label='Baseline')

    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'{target_variable} - Bad Balance Distribution')
    plt.show()

def distribution_plot_num_cat(data, target_variable, bad_flag, bal_variable,bad_bal_variable,data_dictionary):
    """
    Plot the distribution plot for both numerical and categorical features
    """
    if pd.api.types.is_categorical_dtype(data[target_variable]) or pd.api.types.is_object_dtype(data[target_variable]):

        distribution_by_group(data, target_variable, bad_flag, bal_variable, bad_bal_variable, data_dictionary)
    else:
        distribution_by_decile_logic(data, target_variable, bad_flag, bal_variable, bad_bal_variable, data_dictionary)


# ============================================= 
#    Good vs. Bad Distribution
# =============================================

def good_bad_distribution(data, target_variable, bad_flag, data_dictionary=None, narrow=False, bins=10):
    """
    Plot the Good vs. Bad distribution for target column
    """
    plt.figure(figsize=(8,6))

    # Ensure cleaned_var is a string
    cleaned_var = target_variable.split('_')[-1] if '_' in target_variable else target_variable
    definition ='N/A'
    valid_min = 0

    # Optional: data dictionary info
    if data_dictionary is not None and cleaned_var in data_dictionary['Variable'].values:
        row = data_dictionary[data_dictionary['Variable'] == cleaned_var].iloc[0]
        definition = row['Definition'] if pd.notnull(row['Definition']) else 'N/A'
        valid_min = row['Valid Min'] if pd.notnull(row['Valid Min']) else 0
        print(target_variable, ":", definition)  

    # Filter extreme / narrow values
    if narrow:
        data = data[data[target_variable] >= valid_min].copy()
    else:
        threshold = data[target_variable].quantile(0.999)
        data = data[(data[target_variable] >= valid_min) & (data[target_variable] < threshold)].copy()

    # --- Numeric variable ---
    if pd.api.types.is_numeric_dtype(data[target_variable]):
        group_good = data[data[bad_flag]==0]
        group_bad = data[data[bad_flag]>0]

        fig, ax1 = plt.subplots(figsize=(6,3))
        sns.histplot(group_good[target_variable], kde=True, bins=bins, ax=ax1, color='blue', alpha=0.6, label='Good')
        ax1.set_ylabel('Good Count', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        sns.histplot(group_bad[target_variable], kde=True, bins=bins, ax=ax2, color='red', alpha=0.6, label='Bad')
        ax2.set_ylabel('Bad Count', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax1.set_xlabel(target_variable)
        plt.title(f'Distribution of {target_variable} by {bad_flag}')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.tight_layout()

    # --- Categorical variable ---
    elif pd.api.types.is_object_dtype(data[target_variable]) or pd.api.types.is_categorical_dtype(data[target_variable]):
        unique_flags = sorted(data[bad_flag].unique())
        fig, axes = plt.subplots(1, len(unique_flags), figsize=(6*len(unique_flags), 3), sharey=True)

        if len(unique_flags) == 1:
            axes = [axes]

        colors = ['blue', 'red']  # Extend if more classes
        for i, flag in enumerate(unique_flags):
            sns.countplot(data=data[data[bad_flag]==flag], x=target_variable, ax=axes[i],
                          color=colors[i % len(colors)])
            axes[i].set_title(f'{bad_flag}={flag} - {target_variable}')
            axes[i].set_xlabel(target_variable)
            axes[i].set_ylabel('Count' if i==0 else '')

        plt.tight_layout()

    plt.show()



# ============================================= 
#    Funnel plot - feature selection process
# =============================================

def funnel_feature_selection(original_feature, iv_survived, corr_survived, iv_corr_survived, logic_survived, expert_judgement = None):

funnel_dictionary = {
        'Initial': original_feature,
        'Information Value Filter': iv_survived,
        'Correlation Filter': corr_survived,
        'IV + Corr Filter': iv_corr_survived,
        'Logic Checks': logic_survived
    }

    if expert_judgement is not None:
        funnel_dictionary['Expert Judgement'] = expert_judgement

    funnel_data = pd.DataFrame({
        'Stage': list(funnel_dictionary.keys()),
        'Feature Number': list(funnel_dictionary.values())
    })

    fig = px.funnel(
        funnel_data,
        x='Feature Number',
        y='Stage',
        title='Funnel plot for Feature Selection Process'
    )

    fig.update_layout(
        yaxis=dict(
            categoryorder='array',
            categoryarray=list(funnel_dictionary.keys()),
            tickfont=dict(size=14)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600
    )

    fig.show()