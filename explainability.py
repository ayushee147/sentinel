"""
explainability.py
SHAP-based feature attribution for Sentinel.
We wrap the Bayesian model's mean prediction in a callable that SHAP can explain.
"""

import numpy as np
import pandas as pd
import shap
from model import preprocess, FEATURES, FEATURE_LABELS
import pymc as pm


def make_predictor(model, trace):
    """
    Returns a function f(X_array) -> prob_array usable by SHAP.
    Uses the posterior mean of betas and intercept for speed
    (SHAP needs thousands of evaluations; full posterior is too slow).
    """
    # Extract posterior means
    intercept_mean = float(trace.posterior["intercept"].values.mean())
    betas_mean     = trace.posterior["betas"].values.reshape(-1, len(FEATURES)).mean(axis=0)

    def predict_fn(X):
        logits = intercept_mean + X @ betas_mean
        return 1 / (1 + np.exp(-logits))

    return predict_fn, intercept_mean, betas_mean


def compute_shap(model, trace, user_df: pd.DataFrame, background_df: pd.DataFrame):
    """
    Compute SHAP values for user_df using background_df as the reference distribution.

    Returns:
        shap_values : np.ndarray (n_users, n_features)
        feature_names : list[str]
    """
    predict_fn, _, _ = make_predictor(model, trace)

    X_back = preprocess(background_df).values.astype(float)
    X_user = preprocess(user_df).values.astype(float)

    # Use KernelExplainer — model-agnostic, works with any callable
    # We use a small background sample (100 pts) for speed
    idx      = np.random.choice(len(X_back), min(100, len(X_back)), replace=False)
    explainer = shap.KernelExplainer(predict_fn, X_back[idx])
    sv        = explainer.shap_values(X_user, nsamples=200, silent=True)

    return sv, FEATURES


def shap_summary(shap_values, feature_names, top_n=5):
    """
    Returns a ranked DataFrame of feature impacts for display.
    Positive = pushes probability up. Negative = pulls it down.
    """
    sv = np.array(shap_values).flatten()
    rows = []
    for feat, val in zip(feature_names, sv):
        rows.append({
            "Feature":   FEATURE_LABELS.get(feat, feat),
            "Impact":    float(val),
            "Direction": "positive" if val >= 0 else "negative",
        })

    df = pd.DataFrame(rows)
    df["AbsImpact"] = df["Impact"].abs()
    df = df.sort_values("AbsImpact", ascending=False).head(top_n).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df


if __name__ == "__main__":
    from generate_data import generate_dataset
    from model import load_model
    import os

    os.makedirs("data", exist_ok=True)
    df = generate_dataset()

    model, trace = load_model()

    user = pd.DataFrame([{
        "cgpa": 7.75, "leetcode": 75, "projects": 4,
        "has_internship": 1, "dsa_hrs_week": 5,
        "project_hrs_week": 8, "gym_consistency": 60,
        "resume_score": 65, "success": 1
    }])

    sv, feats = compute_shap(model, trace, user, df)
    summary   = shap_summary(sv, feats)
    print("\nSHAP Feature Attribution:")
    print(summary.to_string(index=False))
