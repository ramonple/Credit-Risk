import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

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



import pandas as pd
import itertools

def generate_interactions(df, feature_groups, operations):
    """
    df: pandas DataFrame
    feature_groups: dict of {group_name: [feature_list]}
    operations: list of functions taking two or three arguments
    """
    new_features = pd.DataFrame(index=df.index)
    
    for group_name, features in feature_groups.items():
        # pairwise combinations within group
        for f1, f2 in itertools.combinations(features, 2):
            for op in operations:
                try:
                    new_feat_name = f"{f1}_{op.__name__}_{f2}"
                    new_features[new_feat_name] = op(df[f1], df[f2])
                except Exception:
                    continue
    
    # Optionally, cross-group combinations
    # for group1, group2 in itertools.combinations(feature_groups.keys(), 2):
    #     ...
    
    return new_features

import numpy as np

feature_groups = {
    "demographics": ["Age", "Income", "EmploymentLength"],
    "account_activity": ["NumTransactions", "AvgBalance", "NumProducts"]
}

operations = [np.add, np.subtract, np.multiply, lambda x,y: x/y]

new_features = generate_interactions(df, feature_groups, operations)
df_extended = pd.concat([df, new_features], axis=1)

print(df_extended.shape)
