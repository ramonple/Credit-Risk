import numpy as np
import pandas as pd
from typing import Callable, Iterable, Optional, Union, Any, Dict

def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))

def _get_predicted_proba(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return P(y=1) for sklearn/xgboost-style models.
    Supports: predict_proba, decision_function->sigmoid, predict (fallback).

    a. Models with predict_proba returning two columns (most common).
       proba[:, 0] → P(y = 0), proba[:, 1] → P(y = 1)
       LR, RandomForest, XGBoost, LightGBM,CatBoost, etc.
    b. Models with predict_proba returning one column
       Some boosted / wrapped models
    c. Models with decision_function (no probabilities)
       LinearSVC, SVC(probability=False)
    """
    # Preferred: predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)

        # Standard binary case: (n, 2)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]

        # Already (n,)
        if proba.ndim == 1:
            return proba

    # Fallback: decision_function -> sigmoid
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return _sigmoid(score)

    # Last resort
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def model_deploy(
    data: pd.DataFrame,
    modelling_feature_list,
    target_flag,
    model,
    convert_threshold=None,
    proba_col="pred_proba",
    flag_col="pred_flag",
):
    """
    Parameters
    ----------
    data : raw DataFrame (will not be modified)
    modelling_feature_list : list of feature names used for modelling
    target_flag : kept for interface consistency (not required for scoring)
    model : fitted classifier
    convert_threshold : float in (0,1), optional. If provided, will create binary flag column
    """

    df_out = data.copy()

    X = df_out[modelling_feature_list]
    p = _get_predicted_proba(model, X)
    p = np.clip(p, 0.0, 1.0)

    df_out[proba_col] = p

    if convert_threshold is not None:
        df_out[flag_col] = (p >= convert_threshold).astype(int)

    return df_out
