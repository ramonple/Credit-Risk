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
import numpy as np
import itertools

def generate_interactions(df, feature_groups, operations, max_order=2):
    """
    Generate interaction features based on feature groups with smart naming and conditional division.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataset.
    feature_groups : dict
        Dictionary of {group_name: [feature_list]}.
    operations : list
        List of operations: 'add', 'sub', 'mul', 'div'.
    max_order : int
        Maximum number of features to combine (2 for pairwise, 3 for triplet).
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing newly generated features.
    """
    new_features = pd.DataFrame(index=df.index)

    # Generate combinations within groups
    for group_name, features in feature_groups.items():
        for order in range(2, max_order+1):
            for combo in itertools.combinations(features, order):
                if order == 2:
                    f1, f2 = combo
                    for op in operations:
                        try:
                            if op == 'add':
                                new_feat_name = f"{f1}_add_{f2}"
                                new_features[new_feat_name] = df[f1] + df[f2]
                            elif op == 'sub':
                                new_feat_name = f"{f1}_sub_{f2}"
                                new_features[new_feat_name] = df[f1] - df[f2]
                            elif op == 'mul':
                                new_feat_name = f"{f1}_mul_{f2}"
                                new_features[new_feat_name] = df[f1] * df[f2]
                            elif op == 'div':
                                new_feat_name = f"{f1}_div_{f2}"
                                new_features[new_feat_name] = np.where(
                                    df[f2] != 0,
                                    df[f1] / df[f2],
                                    0
                                )
                        except Exception as e:
                            print(f"Skipping {f1}, {f2} due to error: {e}")
                elif order == 3:
                    f1, f2, f3 = combo
                    # For triplets, only allow multiplication or addition (can customize)
                    for op in ['add', 'mul']:
                        try:
                            if op == 'add':
                                new_feat_name = f"{f1}_add_{f2}_add_{f3}"
                                new_features[new_feat_name] = df[f1] + df[f2] + df[f3]
                            elif op == 'mul':
                                new_feat_name = f"{f1}_mul_{f2}_mul_{f3}"
                                new_features[new_feat_name] = df[f1] * df[f2] * df[f3]
                        except Exception as e:
                            print(f"Skipping {f1}, {f2}, {f3} due to error: {e}")

    return new_features

# Example usage:
feature_groups = {
    "demographics": ["Age", "Income", "EmploymentLength"],
    "account_activity": ["NumTransactions", "AvgBalance", "NumProducts"]
}

operations = ['add', 'sub', 'mul', 'div']

# Generate features
new_feats = generate_interactions(df, feature_groups, operations, max_order=3)

# Append to original dataset
df_extended = pd.concat([df, new_feats], axis=1)
print(df_extended.shape)
