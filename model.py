"""
model.py — Sentinel Bayesian model with cloudpickle-based saving
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import cloudpickle
import os

FEATURES = [
    "cgpa", "leetcode_log", "projects", "has_internship",
    "dsa_hrs_week", "project_hrs_week", "gym_consistency", "resume_score",
]

FEATURE_LABELS = {
    "cgpa": "CGPA", "leetcode_log": "LeetCode Problems",
    "projects": "# Projects", "has_internship": "Prior Internship",
    "dsa_hrs_week": "DSA hrs/week", "project_hrs_week": "Project hrs/week",
    "gym_consistency": "Gym Consistency (%)", "resume_score": "Resume Score",
}

MODEL_PATH = "model/sentinel_trace.pkl"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["leetcode_log"] = np.log1p(out["leetcode"])
    bounds = {
        "cgpa": (5.0, 10.0), "leetcode_log": (0.0, np.log1p(600)),
        "projects": (0.0, 8.0), "has_internship": (0.0, 1.0),
        "dsa_hrs_week": (0.0, 20.0), "project_hrs_week": (0.0, 25.0),
        "gym_consistency": (0.0, 100.0), "resume_score": (0.0, 100.0),
    }
    for feat, (lo, hi) in bounds.items():
        out[feat] = (out[feat] - lo) / (hi - lo)
        out[feat] = out[feat].clip(0, 1)
    return out[FEATURES]


def build_and_train(df: pd.DataFrame, draws=1000, tune=1000, chains=2, cores=1):
    X = preprocess(df).values.astype(float)
    y = df["success"].values.astype(int)
    n_feat = X.shape[1]

    with pm.Model() as model:
        X_data = pm.Data("X_data", X, mutable=True)
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=n_feat)
        logit_p = intercept + pt.dot(X_data, betas)
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        pm.Bernoulli("obs", p=p, observed=y)
        trace = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores,
            target_accept=0.9, progressbar=True,
            return_inferencedata=True,
        )

    return model, trace


def predict_proba(model, trace, user_df: pd.DataFrame):
    X_new = preprocess(user_df).values.astype(float)
    with model:
        pm.set_data({"X_data": X_new})
        ppc = pm.sample_posterior_predictive(
            trace, var_names=["p"], progressbar=False
        )
    samples = ppc.posterior_predictive["p"].values
    samples = samples.reshape(-1, X_new.shape[0])
    return samples.mean(axis=0), np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def save_model(model, trace):
    os.makedirs("model", exist_ok=True)
    # cloudpickle handles PyMC/PyTensor internal objects that
    # standard pickle cannot serialize (closures, local functions, etc.)
    with open(MODEL_PATH, "wb") as f:
        cloudpickle.dump({"model": model, "trace": trace}, f)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    with open(MODEL_PATH, "rb") as f:
        obj = cloudpickle.load(f)
    return obj["model"], obj["trace"]


def counterfactual_analysis(model, trace, base_profile: dict, scenarios: dict):
    results = {}
    base_df = pd.DataFrame([base_profile])
    results["Current Profile"] = predict_proba(model, trace, base_df)
    for label, overrides in scenarios.items():
        cf = base_profile.copy()
        cf.update(overrides)
        results[label] = predict_proba(model, trace, pd.DataFrame([cf]))
    return results
