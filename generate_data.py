"""
generate_data.py
Synthetic student profiles with MATHEMATICALLY CALIBRATED intercepts.

Calibration method:
  For each company, intercept is solved so that sigmoid(intercept + 
  weighted_sum_at_average_student) == documented_target_baseline_rate.
  
  This means: an average IIIT-D student profile (CGPA 7.8, LC 100,
  2.5 projects, 35% internship rate, etc.) produces exactly the
  documented selection rate for that company.

  intercept = logit(target_rate) - weighted_sum(average_features)
  
  This is the only principled way to set an intercept — back-calculated
  from the observable outcome you want to reproduce.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200

COMPANY_PROFILES = {
    "Google": {
        "intercept": -6.425,   # → exactly 10% for average student
        "lc_weight":    2.8,
        "cgpa_weight":  1.2,
        "proj_weight":  0.8,
        "intern_weight":1.0,
        "resume_weight":0.8,
        "dsa_weight":   1.0,
        "projh_weight": 0.4,
        "gym_weight":   0.2,
        "target_rate":  0.10,
        "description":  "DSA/LeetCode dominates. 250+ problems near-essential for OA.",
        "lc_threshold": 250,
        "key_signal":   "LeetCode count",
    },
    "Microsoft": {
        "intercept": -5.376,   # → exactly 22% for average student
        "lc_weight":    2.0,
        "cgpa_weight":  1.4,
        "proj_weight":  1.2,
        "intern_weight":1.3,
        "resume_weight":1.0,
        "dsa_weight":   0.8,
        "projh_weight": 0.6,
        "gym_weight":   0.2,
        "target_rate":  0.22,
        "description":  "Balanced: DSA + strong projects + behavioral fit.",
        "lc_threshold": 150,
        "key_signal":   "Balance of DSA and projects",
    },
    "Databricks": {
        "intercept": -5.435,   # → exactly 18% for average student
        "lc_weight":    1.6,
        "cgpa_weight":  1.0,
        "proj_weight":  1.6,
        "intern_weight":1.4,
        "resume_weight":1.2,
        "dsa_weight":   0.8,
        "projh_weight": 1.0,
        "gym_weight":   0.1,
        "target_rate":  0.18,
        "description":  "ML systems depth + real deployed projects matter most.",
        "lc_threshold": 100,
        "key_signal":   "Deployed ML projects",
    },
    "Palantir": {
        "intercept": -6.039,   # → exactly 13% for average student
        "lc_weight":    1.8,
        "cgpa_weight":  1.0,
        "proj_weight":  1.8,
        "intern_weight":1.6,
        "resume_weight":1.0,
        "dsa_weight":   0.8,
        "projh_weight": 0.8,
        "gym_weight":   0.3,
        "target_rate":  0.13,
        "description":  "Problem-solving, initiative, real-world engineering impact.",
        "lc_threshold": 150,
        "key_signal":   "Self-driven projects with real impact",
    },
    "Nvidia": {
        "intercept": -5.753,   # → exactly 16% for average student
        "lc_weight":    1.8,
        "cgpa_weight":  1.6,
        "proj_weight":  1.4,
        "intern_weight":1.2,
        "resume_weight":0.9,
        "dsa_weight":   0.9,
        "projh_weight": 0.7,
        "gym_weight":   0.2,
        "target_rate":  0.16,
        "description":  "Systems knowledge + CGPA + hardware-adjacent projects.",
        "lc_threshold": 150,
        "key_signal":   "CGPA + systems projects",
    },
    "DE Shaw": {
        "intercept": -6.927,   # → exactly 7% for average student
        "lc_weight":    2.4,
        "cgpa_weight":  1.8,
        "proj_weight":  0.8,
        "intern_weight":1.2,
        "resume_weight":0.8,
        "dsa_weight":   1.2,
        "projh_weight": 0.4,
        "gym_weight":   0.1,
        "target_rate":  0.07,
        "description":  "Quant aptitude + high CGPA (8.0+) + strong DSA.",
        "lc_threshold": 300,
        "key_signal":   "CGPA above 8.0 + heavy DSA",
    },
    "Jane Street": {
        "intercept": -7.910,   # → exactly 3% for average student
        "lc_weight":    3.0,
        "cgpa_weight":  1.6,
        "proj_weight":  0.6,
        "intern_weight":1.0,
        "resume_weight":0.6,
        "dsa_weight":   1.4,
        "projh_weight": 0.3,
        "gym_weight":   0.1,
        "target_rate":  0.03,
        "description":  "Competitive programming essential. Hardest technical bar.",
        "lc_threshold": 400,
        "key_signal":   "Competitive programming (Codeforces Div 1 level)",
    },
    "GE Vernova": {
        "intercept": -4.275,   # → exactly 40% for average student
        "lc_weight":    1.0,
        "cgpa_weight":  1.4,
        "proj_weight":  1.4,
        "intern_weight":2.0,
        "resume_weight":1.2,
        "dsa_weight":   0.6,
        "projh_weight": 0.8,
        "gym_weight":   0.3,
        "target_rate":  0.40,
        "description":  "Domain experience + applied projects + CGPA.",
        "lc_threshold": 75,
        "key_signal":   "Domain internship experience",
    },
}

DEFAULT_COMPANY = "Microsoft"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_dataset(n=N, company=DEFAULT_COMPANY):
    profile = COMPANY_PROFILES[company]

    cgpa            = np.clip(np.random.normal(7.8,  0.6,  n), 5.0, 10.0)
    leetcode        = np.clip(np.random.exponential(100,   n), 0,   600).astype(int)
    projects        = np.clip(np.random.poisson(2.5,        n), 0,    8).astype(int)
    has_internship  = np.random.binomial(1, 0.35,           n)
    dsa_hrs         = np.clip(np.random.normal(5,   2.5,    n), 0,   20)
    project_hrs     = np.clip(np.random.normal(6,   3,      n), 0,   25)
    gym_consistency = np.clip(np.random.normal(55,  20,     n), 0,  100)
    resume_score    = np.clip(np.random.normal(60,  15,     n), 0,  100)

    cgpa_n   = (cgpa - 5) / 5
    lc_n     = np.log1p(leetcode) / np.log1p(600)
    proj_n   = projects / 8
    dsa_n    = dsa_hrs / 20
    projh_n  = project_hrs / 25
    gym_n    = gym_consistency / 100
    resume_n = resume_score / 100

    logit = (
        profile["intercept"]
        + profile["lc_weight"]     * lc_n
        + profile["cgpa_weight"]   * cgpa_n
        + profile["proj_weight"]   * proj_n
        + profile["intern_weight"] * has_internship
        + profile["resume_weight"] * resume_n
        + profile["dsa_weight"]    * dsa_n
        + profile["projh_weight"]  * projh_n
        + profile["gym_weight"]    * gym_n
        + np.random.normal(0, 0.35, n)
    )

    prob    = sigmoid(logit)
    outcome = np.random.binomial(1, prob)

    return pd.DataFrame({
        "cgpa":            cgpa,
        "leetcode":        leetcode,
        "projects":        projects,
        "has_internship":  has_internship,
        "dsa_hrs_week":    dsa_hrs,
        "project_hrs_week":project_hrs,
        "gym_consistency": gym_consistency,
        "resume_score":    resume_score,
        "success":         outcome,
    })


def get_company_names():
    return list(COMPANY_PROFILES.keys())


def get_company_description(company):
    return COMPANY_PROFILES[company]["description"]


def get_company_target_rate(company):
    return COMPANY_PROFILES[company]["target_rate"]


def get_all_profiles():
    return COMPANY_PROFILES


if __name__ == "__main__":
    print("Verifying calibrated baseline rates:")
    print("-" * 55)
    for company in COMPANY_PROFILES:
        df = generate_dataset(5000, company=company)
        rate = df["success"].mean()
        target = COMPANY_PROFILES[company]["target_rate"]
        print(f"  {company:<15} actual={rate:.1%}  target={target:.0%}  {'✓' if abs(rate-target) < 0.05 else '!'}")
