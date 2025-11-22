import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
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
    validation_size=0.2,               # validation set size
    random_state=42,
    max_feature_warning_threshold=300,
    plot_roc=True,
    plot_feature_importance=True
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

    # ----------- Training set ----------------
    train_pred_prob = best_model.predict_proba(X_train)[:, 1]
    train_pred_label = best_model.predict(X_train)

    # ----------- Validation set  ----------------
    valid_pred_prob = best_model.predict_proba(X_valid)[:, 1]
    valid_pred_label = best_model.predict(X_valid)

    # Metrics function
    def compute_metrics(y_true, y_pred_label, y_pred_prob):
        auc_val = roc_auc_score(y_true, y_pred_prob)
        gini_val = 2 * auc_val - 1
        ks_val = ks_2samp(y_pred_prob[y_true==1], y_pred_prob[y_true==0]).statistic
        acc_val = accuracy_score(y_true, y_pred_label)
        cm_val = confusion_matrix(y_true, y_pred_label)
        return auc_val, gini_val, ks_val, acc_val, cm_val

    auc_train, gini_train, ks_train, acc_train, cm_train = compute_metrics(y_train, train_pred_label, train_pred_prob)
    auc_valid, gini_valid, ks_valid, acc_valid, cm_valid = compute_metrics(y_valid, valid_pred_label, valid_pred_prob)

    metrics_df = pd.DataFrame({
    "Metric": ["AUC", "Gini", "KS", "Accuracy"],
    "Train": [auc_train, gini_train, ks_train, acc_train],
    "Validation": [auc_valid, gini_valid, ks_valid, acc_valid]})

    if plot_roc:
        fpr_train, tpr_train, _ = roc_curve(y_train, train_pred_prob)
        fpr_val, tpr_val, _ = roc_curve(y_valid, valid_pred_prob)

        plt.figure(figsize=(8,6))
        plt.plot(fpr_train, tpr_train, label=f'Train ROC (AUC={auc_train:.3f}, Gini={gini_train:.3f})')
        plt.plot(fpr_val, tpr_val, label=f'Validation ROC (AUC={auc_valid:.3f}, Gini={gini_valid:.3f})')
        plt.plot([0,1],[0,1],'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve: Train vs Validation')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    # ----------- Feature Importance ----------------
    fi = pd.DataFrame({
        "feature": building_feature_after_encoding,
        "importance": best_model.feature_importances_
    }).sort_values(by="importance", ascending=False)


    if plot_feature_importance:
        top_features = fi.head(20).sort_values(by='importance')  # sort ascending for better visualization
        plt.figure(figsize=(8,6))
        plt.barh(top_features['feature'], top_features['importance'], color='skyblue')
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Top 20 Feature Importances')
        plt.gca().invert_yaxis()  # largest importance on top
        plt.show()


    if save_model_path is not None:
        full_path = save_model_path + "/" + save_model_name
        with open(full_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"\nModel saved to: {full_path}")



    return best_model, searcher.best_params_, metrics_df , fi
