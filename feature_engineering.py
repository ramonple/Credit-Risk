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
                            elif op =='max':
                                new_feat_name = f"max_{f1}_{f2}"
                                new_features[new_feat_name] = (df[[f1,f2]]).max(axis=1)
                                
                            elif op =='min':
                                new_feat_name = f"min_{f1}_{f2}"
                                new_features[new_feat_name] = (df[[f1,f2]]).min(axis=1)
                                
                            elif op =='mean':
                                new_feat_name = f"min_{f1}_{f2}"
                                new_features[new_feat_name] = (df[[f1,f2]]).mean(axis=1)

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
