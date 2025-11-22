import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from scipy.stats import ks_2samp


def train_random_forest(
    data,
    target_flag,
    building_feature_after_encoding,
    param_grid=None,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    save_model_path=None,
    save_model_name='RF_model.pkl',
    search_method='grid',              # 'grid' or 'random'
    validation_size=0.2, # validation set size
    random_state=42,
    max_feature_warning_threshold=300
):
    """
    Train a Random Forest with advanced features:
    - Optional GridSearch or RandomizedSearch
    - Hold-out validation evaluation
    - Metrics: AUC, GINI, KS, Accuracy, Confusion Matrix
    - Feature importance extraction
    - Class weight tuning support
    - Reproducibility controls
    - Optional model saving

    Returns
    -------
    best_model : trained model
    results : dict containing:
        - train/validation metrics
        - best params
        - feature importance
        - confusion matrix
    fi: feature important list 
    """

    if len(building_feature_after_encoding) > max_feature_warning_threshold:
        print(f" WARNING: Model input has {len(building_feature_after_encoding)} features.")
        print("Consider dimension reduction, grouping, or regularization.")

    train_data, valid_data = train_test_split(
        data,
        test_size=validation_size,
        random_state=random_state,
        stratify=data[target_flag]
    )

    X_train = train_data[building_feature_after_encoding]
    y_train = train_data[target_flag]

    X_valid = valid_data[building_feature_after_encoding]
    y_valid = valid_data[target_flag]

    if param_grid is None:
        param_grid = {
            'n_estimators': [200, 400],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }

    # Base model
    model = RandomForestClassifier(
        random_state=random_state
    )

    if search_method.lower() == 'random':
        searcher = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=scoring,
            cv=cv,
            n_iter=10,
            n_jobs=n_jobs,
            verbose=1,
            random_state=random_state
        )
    else:
        searcher = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1
        )

    searcher.fit(X_train, y_train)
    best_model = searcher.best_estimator_

    print("Best Parameters:", searcher.best_params_)
    print(f"Best CV Score ({scoring}): {searcher.best_score_:.4f}")

    train_pred_prob = best_model.predict_proba(X_train)[:, 1]
    train_pred_label = best_model.predict(X_train)

    auc_train = roc_auc_score(y_train, train_pred_prob)
    gini_train = 2 * auc_train - 1
    ks_train = ks_2samp(train_pred_prob[y_train == 1], train_pred_prob[y_train == 0]).statistic
    acc_train = accuracy_score(y_train, train_pred_label)
    cm_train = confusion_matrix(y_train, train_pred_label)


    # ----------- Validation set evaluation ----------------
    valid_pred_prob = best_model.predict_proba(X_valid)[:, 1]
    valid_pred_label = best_model.predict(X_valid)

    auc_valid = roc_auc_score(y_valid, valid_pred_prob)
    gini_valid = 2 * auc_valid - 1
    ks_valid = ks_2samp(valid_pred_prob[y_valid == 1],valid_pred_prob[y_valid == 0]).statistic
    acc_valid = accuracy_score(y_valid, valid_pred_label)
    cm_valid = confusion_matrix(y_valid, valid_pred_label)

    # ----------- Feature Importance ----------------
    fi = pd.DataFrame({
        "feature": building_feature_after_encoding,
        "importance": best_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nTop 10 important features:")
    print(fi.head(10))

    if save_model_path is not None:
        full_path = save_model_path + "/" + save_model_name
        with open(full_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"\nModel saved to: {full_path}")



    metrics_df = pd.DataFrame({
    "Metric": ["AUC", "Gini", "KS", "Accuracy"],
    "Train": [auc_train, gini_train, ks_train, acc_train],
    "Validation": [auc_valid, gini_valid, ks_valid, acc_valid]})

    return best_model, searcher.best_params_, metrics_df , fi
