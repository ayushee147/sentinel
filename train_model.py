"""
train_model.py
Run this ONCE before launching the app.
Generates synthetic data, trains the Bayesian model, saves to disk.
Takes ~3-8 minutes on a laptop CPU.
"""

import os


def main():
    os.makedirs("data",  exist_ok=True)
    os.makedirs("model", exist_ok=True)

    print("=" * 55)
    print("  SENTINEL — Model Training")
    print("=" * 55)

    print("\n[1/3] Generating synthetic student dataset...")
    from generate_data import generate_dataset
    df = generate_dataset(1200, company="Microsoft")
    df.to_csv("data/student_profiles.csv", index=False)
    print(f"      \u2713 {len(df)} samples | success rate: {df['success'].mean():.1%}")

    print("\n[2/3] Training Bayesian logistic regression (PyMC)...")
    print("      This takes a few minutes. Grab a coffee \u2615")
    from model import build_and_train, save_model
    model, trace = build_and_train(df, draws=1000, tune=1000, chains=2, cores=1)
    save_model(model, trace)
    print("      \u2713 Model trained and saved to model/sentinel_trace.pkl")

    print("\n[3/3] Running sanity check...")
    import pandas as pd
    from model import load_model, predict_proba
    model, trace = load_model()

    test_profiles = [
        {"label": "Strong candidate",
         "cgpa":7.9, "leetcode":300, "projects":5, "has_internship":1,
         "dsa_hrs_week":10, "project_hrs_week":10, "gym_consistency":70,
         "resume_score":80, "success":1},
        {"label": "Average candidate",
         "cgpa":7.75, "leetcode":75, "projects":4, "has_internship":1,
         "dsa_hrs_week":5, "project_hrs_week":8, "gym_consistency":60,
         "resume_score":65, "success":1},
        {"label": "Weak candidate",
         "cgpa":6.5, "leetcode":20, "projects":1, "has_internship":0,
         "dsa_hrs_week":1, "project_hrs_week":2, "gym_consistency":20,
         "resume_score":30, "success":0},
    ]

    for p in test_profiles:
        label = p.pop("label"); p.pop("success")
        df_t = pd.DataFrame([p])
        m, lo, hi = predict_proba(model, trace, df_t)
        print(f"      {label}: {m[0]:.1%} [{lo[0]:.1%}\u2013{hi[0]:.1%}]")

    print("\n\u2705 All done! Launch the app with:")
    print("   streamlit run app.py\n")


if __name__ == "__main__":
    main()
