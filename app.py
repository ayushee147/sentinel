"""
app.py — Sentinel: Personal Growth Optimizer
Streamlit front-end with personalized roadmap and company recommendations.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Sentinel — Growth Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252a3a);
        border: 1px solid #2d3547; border-radius: 12px;
        padding: 20px; text-align: center; margin: 8px 0;
    }
    .metric-value { font-size: 2.4rem; font-weight: 700; color: #4f8ef7; }
    .metric-label { font-size: 0.85rem; color: #8892a4; margin-top: 4px; }
    .action-card {
        background: #1a1f2e; border-left: 3px solid #4f8ef7;
        border-radius: 8px; padding: 12px 16px; margin: 8px 0;
    }
    .action-title { font-weight: 600; color: #e6edf3; }
    .action-impact { color: #3fb950; font-size: 0.9rem; }
    .action-detail { color: #8892a4; font-size: 0.82rem; margin-top: 4px; }
    .warning-card {
        background: #2d1b1b; border-left: 3px solid #f85149;
        border-radius: 8px; padding: 10px 14px; margin: 6px 0;
        color: #f85149; font-size: 0.87rem;
    }
    .company-card {
        background: #1a2530; border: 1px solid #2d3547;
        border-radius: 10px; padding: 14px; margin: 6px 0;
    }
    .roadmap-card {
        background: linear-gradient(135deg, #1a2e1a, #1a1f2e);
        border: 1px solid #3fb950; border-radius: 12px;
        padding: 16px 20px; margin: 10px 0;
    }
    .week-badge {
        background: #4f8ef7; color: white; border-radius: 4px;
        padding: 2px 8px; font-size: 0.78rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Sentinel model… (first launch takes ~10 min)")
def load_sentinel():
    from model import load_model
    import os
    if not os.path.exists("model/sentinel_trace.pkl"):
        st.info("First launch — training model. This takes ~10 minutes. Please wait.")
        from generate_data import generate_dataset
        from model import build_and_train, save_model
        import os
        os.makedirs("model", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        df = generate_dataset(1200, company="Microsoft")
        model, trace = build_and_train(df, draws=500, tune=500, chains=1, cores=1)
        save_model(model, trace)
        return model, trace
    model, trace = load_model()
    return model, trace


def fast_predict(trace, profile: dict) -> tuple:
    from model import FEATURES
    df = _profile_to_df(profile)

    # Import preprocess locally to avoid circular import issues
    from model import preprocess
    X = preprocess(df).values.astype(float)

    intercepts = trace.posterior["intercept"].values.flatten()
    betas = trace.posterior["betas"].values.reshape(-1, len(FEATURES))
    logits = intercepts[:, None] + betas @ X.T
    probs = 1 / (1 + np.exp(-logits))
    samples = probs[:, 0]
    return float(samples.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _profile_to_df(p: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "cgpa": p["cgpa"], "leetcode": p["leetcode"],
        "projects": p["projects"], "has_internship": float(p["has_internship"]),
        "dsa_hrs_week": p["dsa_hrs_week"], "project_hrs_week": p["project_hrs_week"],
        "gym_consistency": p["gym_consistency"], "resume_score": p["resume_score"],
        "success": 0,
    }])


def build_scenarios(trace, base: dict) -> list:
    base_prob, _, _ = fast_predict(trace, base)
    candidates = [
        {"label": "Increase DSA to 10 hrs/week",
         "override": {"dsa_hrs_week": 10, "leetcode": min(base["leetcode"] + 80, 600)},
         "hrs_needed": 5, "weeks": 12,
         "rationale": "Consistent DSA + more problems — highest signal for screening rounds."},
        {"label": "Complete & deploy Sentinel project",
         "override": {"projects": base["projects"] + 1, "resume_score": min(base["resume_score"] + 12, 100)},
         "hrs_needed": 15, "weeks": 2,
         "rationale": "Live deployed ML project with Bayesian/SHAP stack stands out for ML roles."},
        {"label": "Reach 250 LeetCode problems",
         "override": {"leetcode": 250},
         "hrs_needed": 8, "weeks": 8,
         "rationale": "250+ is the threshold where most FAANG screening filters pass comfortably."},
        {"label": "Improve CGPA to 8.2",
         "override": {"cgpa": 8.2},
         "hrs_needed": 10, "weeks": 16,
         "rationale": "Crosses common shortlisting cutoffs (Google India ~7.5, MS India ~8.0+)."},
        {"label": "Improve resume & GitHub portfolio",
         "override": {"resume_score": min(base["resume_score"] + 20, 100)},
         "hrs_needed": 3, "weeks": 1,
         "rationale": "Clean READMEs, live project links, quantified bullets raise resume signal."},
        {"label": "DSA drops below 3 hrs/week",
         "override": {"dsa_hrs_week": 2},
         "hrs_needed": 0, "weeks": 0,
         "rationale": "Warning: reducing DSA practice significantly weakens screening performance."},
    ]
    results = []
    for c in candidates:
        cf = {**base, **c["override"]}
        cf_prob, _, _ = fast_predict(trace, cf)
        results.append({**c, "delta": cf_prob - base_prob, "cf_prob": cf_prob})
    results.sort(key=lambda x: x["delta"], reverse=True)
    return results


def build_roadmap(trace, base: dict, total_hrs: int, weeks: int, scenarios: list) -> list:
    """
    Given available hours/week and time horizon,
    build a prioritized weekly allocation plan.
    """
    total_budget = total_hrs * weeks
    positive = [s for s in scenarios if s["delta"] > 0]
    roadmap = []
    remaining_hrs = total_budget
    current_week = 1

    for action in positive[:4]:
        action_total_hrs = action["hrs_needed"] * action["weeks"]
        if action_total_hrs > remaining_hrs:
            continue
        end_week = min(current_week + action["weeks"] - 1, weeks)
        roadmap.append({
            "action": action["label"],
            "start_week": current_week,
            "end_week": end_week,
            "hrs_per_week": action["hrs_needed"],
            "impact": action["delta"],
            "new_prob": action["cf_prob"],
            "rationale": action["rationale"],
        })
        remaining_hrs -= action_total_hrs
        current_week = end_week + 1
        if current_week > weeks:
            break

    return roadmap


def get_alternative_companies(trace, profile: dict, current_company: str) -> list:
    """Compare user's probability across all companies."""
    from generate_data import get_all_profiles
    all_profiles = get_all_profiles()
    results = []
    for company, cp in all_profiles.items():
        prob, lo, hi = fast_predict(trace, profile)
        # Adjust probability by company baseline ratio
        adjustment = cp["target_rate"] / all_profiles[current_company]["target_rate"]
        adjusted_prob = min(prob * adjustment, 0.95)
        results.append({
            "company": company,
            "prob": adjusted_prob,
            "target_rate": cp["target_rate"],
            "description": cp["description"],
            "key_signal": cp["key_signal"],
        })
    results.sort(key=lambda x: x["prob"], reverse=True)
    return results


def gauge_chart(prob, lo, hi):
    pct = prob * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 48, "color": "#4f8ef7"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8892a4"},
            "bar": {"color": "#4f8ef7", "thickness": 0.25},
            "steps": [
                {"range": [0, 30], "color": "#2d1b1b"},
                {"range": [30, 60], "color": "#1f2a1f"},
                {"range": [60, 80], "color": "#1a2530"},
                {"range": [80, 100], "color": "#1a2e1a"},
            ],
            "threshold": {"line": {"color": "#3fb950", "width": 3}, "thickness": 0.8, "value": 60},
            "bgcolor": "#0f1117",
        },
        title={"text": f"95% CI: {lo*100:.0f}% – {hi*100:.0f}%", "font": {"color": "#8892a4", "size": 13}},
    ))
    fig.update_layout(paper_bgcolor="#0f1117", font_color="#c9d1d9",
                      height=280, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def waterfall_chart(scenarios, base_prob):
    labels = [s["label"] for s in scenarios]
    deltas = [s["delta"] * 100 for s in scenarios]
    colors = ["#3fb950" if d >= 0 else "#f85149" for d in deltas]
    fig = go.Figure(go.Bar(
        x=deltas, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{d:+.1f}%" for d in deltas], textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#8892a4", line_width=1)
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font_color="#c9d1d9",
        height=340,
        xaxis=dict(title="Change in Success Probability (%)", color="#8892a4", gridcolor="#1e2433", zeroline=False),
        yaxis=dict(color="#c9d1d9", tickfont=dict(size=11)),
        margin=dict(t=10, b=40, l=10, r=60),
    )
    return fig


def main():
    # Header
    st.markdown("""
    <div style='text-align:center; padding: 24px 0 8px 0;'>
        <span style='font-size:2.4rem; font-weight:800; color:#4f8ef7;'>🎯 SENTINEL</span><br>
        <span style='color:#8892a4; font-size:1.05rem;'>
            Uncertainty-Aware Personal Growth Optimizer
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    try:
        model, trace = load_sentinel()
    except FileNotFoundError:
        st.error("⚠️ Model not trained yet. Run `python train_model.py` first.")
        st.stop()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📋 Your Profile")

        from generate_data import get_company_names, get_company_description, get_company_target_rate
        company = st.selectbox("🎯 Target Company", get_company_names(), index=1)
        st.caption(f"{get_company_description(company)} | Baseline selection rate: ~{get_company_target_rate(company):.0%}")
        st.markdown("---")

        st.markdown("**Academic**")
        cgpa = st.slider("CGPA", 5.0, 10.0, 7.75, 0.05)
        leetcode = st.slider("LeetCode Problems Solved", 0, 600, 75, 5)
        projects = st.slider("Completed Projects", 0, 10, 4, 1)
        has_intern = st.checkbox("Prior internship experience?", value=True)

        st.markdown("**Weekly Time Allocation**")
        total_hrs = st.number_input("Total available hours/week", 5, 80, 20)
        prep_weeks = st.slider("Preparation time horizon (weeks)", 4, 52, 16, 4)
        dsa_hrs = st.slider("DSA / LeetCode hrs/week", 0.0, 20.0, 5.0, 0.5)
        project_hrs = st.slider("Project work hrs/week", 0.0, 25.0, 8.0, 0.5)

        st.markdown("**Lifestyle & Resume**")
        gym_pct = st.slider("Gym / Exercise consistency (%)", 0, 100, 60, 5)
        resume_score = st.slider("Resume quality (self-assessed)", 0, 100, 65, 5)
        st.caption("💡 GEV internship + quantified projects ≈ 70–80")

        analyze = st.button("🔍 Analyze & Build Roadmap", use_container_width=True, type="primary")

    if not analyze:
        st.markdown("""
        <div style='text-align:center; padding:80px 20px; color:#8892a4;'>
            <div style='font-size:3rem;'>🎯</div>
            <div style='font-size:1.2rem; margin-top:12px; color:#c9d1d9;'>
                Fill in your profile and click <b>Analyze & Build Roadmap</b>
            </div>
            <div style='margin-top:8px; font-size:0.9rem;'>
                Sentinel computes your success probability, explains what drives it,<br>
                and builds a personalized weekly prep plan within your time budget.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    profile = {
        "cgpa": cgpa, "leetcode": leetcode, "projects": projects,
        "has_internship": int(has_intern), "dsa_hrs_week": dsa_hrs,
        "project_hrs_week": project_hrs, "gym_consistency": gym_pct,
        "resume_score": resume_score,
    }

    with st.spinner("Running Bayesian inference…"):
        prob, lo, hi = fast_predict(trace, profile)

    # ── ROW 1: Gauge + Snapshot ───────────────────────────────────────────────
    col1, col2 = st.columns([1.2, 1], gap="large")
    with col1:
        st.markdown(f"#### 🎯 {company} — Success Probability")
        st.plotly_chart(gauge_chart(prob, lo, hi), use_container_width=True)

    with col2:
        st.markdown("#### 📊 Profile Snapshot")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{cgpa:.2f}</div><div class='metric-label'>CGPA</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{projects}</div><div class='metric-label'>Projects</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{leetcode}</div><div class='metric-label'>LeetCode</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_hrs}h</div><div class='metric-label'>Weekly Hours</div></div>", unsafe_allow_html=True)

        other_hrs = max(0, total_hrs - dsa_hrs - project_hrs - gym_pct / 100 * 7)
        fig_donut = px.pie(
            values=[dsa_hrs, project_hrs, gym_pct / 100 * 7, other_hrs],
            names=["DSA", "Projects", "Gym (~)", "Other"],
            hole=0.55, color_discrete_sequence=["#4f8ef7", "#3fb950", "#f0883e", "#8892a4"],
        )
        fig_donut.update_layout(paper_bgcolor="#0f1117", font_color="#c9d1d9",
                                 height=180, showlegend=True,
                                 legend=dict(font=dict(size=10)),
                                 margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")

    # ── ROW 2: Counterfactuals ────────────────────────────────────────────────
    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown("#### 🔄 What Changes Your Probability")
        st.caption("Model-computed shifts under predefined scenarios. Probability changes are from the Bayesian posterior.")
        with st.spinner("Computing counterfactuals…"):
            scenarios = build_scenarios(trace, profile)
        st.plotly_chart(waterfall_chart(scenarios, prob), use_container_width=True)

    with col4:
        st.markdown("#### 🏢 How You Compare Across Companies")
        st.caption("Same profile evaluated against each company's documented selection criteria and baseline rate.")
        alt_companies = get_alternative_companies(trace, profile, company)
        for ac in alt_companies[:5]:
            bar_width = int(ac["prob"] * 100)
            is_current = "★ " if ac["company"] == company else ""
            color = "#4f8ef7" if ac["company"] == company else "#3fb950" if ac["prob"] > prob else "#6c757d"
            st.markdown(f"""
            <div class='company-card'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#e6edf3; font-weight:600;'>{is_current}{ac["company"]}</span>
                    <span style='color:{color}; font-weight:700; font-size:1.1rem;'>{ac["prob"]:.0%}</span>
                </div>
                <div style='background:#2d3547; border-radius:4px; height:6px; margin:8px 0;'>
                    <div style='background:{color}; width:{bar_width}%; height:6px; border-radius:4px;'></div>
                </div>
                <div style='color:#8892a4; font-size:0.78rem;'>{ac["key_signal"]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── ROW 3: PERSONALIZED ROADMAP ──────────────────────────────────────────
    st.markdown("#### 🗺️ Your Personalized {}-Week Prep Roadmap".format(prep_weeks))
    st.caption(f"Prioritized by probability impact, fitted within your {total_hrs} hrs/week × {prep_weeks} week budget.")

    with st.spinner("Building roadmap…"):
        roadmap = build_roadmap(trace, profile, total_hrs, prep_weeks, scenarios)

    if roadmap:
        cumulative_prob = prob
        for i, step in enumerate(roadmap, 1):
            cumulative_prob = step["new_prob"]
            st.markdown(f"""
            <div class='roadmap-card'>
                <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                    <div>
                        <span class='week-badge'>Week {step["start_week"]}–{step["end_week"]}</span>
                        <span style='color:#e6edf3; font-weight:600; font-size:1rem; margin-left:10px;'>
                            {step["action"]}
                        </span>
                    </div>
                    <div style='text-align:right;'>
                        <span style='color:#3fb950; font-weight:700; font-size:1.1rem;'>
                            {step["impact"]*100:+.1f}% → {step["new_prob"]*100:.0f}%
                        </span>
                        <div style='color:#8892a4; font-size:0.78rem;'>{step["hrs_per_week"]} hrs/week</div>
                    </div>
                </div>
                <div style='color:#8892a4; font-size:0.85rem; margin-top:8px;'>
                    {step["rationale"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#1a2e1a; border:1px solid #3fb950; border-radius:10px;
                    padding:16px; margin-top:12px; text-align:center;'>
            <div style='color:#8892a4; font-size:0.85rem;'>Projected probability after roadmap completion</div>
            <div style='color:#3fb950; font-size:2rem; font-weight:800;'>
                {prob*100:.0f}% → {cumulative_prob*100:.0f}%
            </div>
            <div style='color:#8892a4; font-size:0.8rem; margin-top:4px;'>
                {(cumulative_prob-prob)*100:+.1f} percentage points improvement over {prep_weeks} weeks
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No high-impact actions fit within your current time budget. Try increasing your weekly hours or time horizon.")

    st.markdown("---")

    # ── ROW 4: SHAP ──────────────────────────────────────────────────────────
    st.markdown("#### 🔍 What's Driving Your Probability")
    st.caption("SHAP feature attribution — which inputs matter most. Uses posterior mean for speed; KernelExplainer treats model as black box.")

    try:
        from explainability import compute_shap, shap_summary
        from generate_data import generate_dataset

        @st.cache_data(show_spinner="Computing SHAP attribution…")
        def get_shap_cached(profile_tuple):
            profile_dict = dict(zip(
                ["cgpa","leetcode","projects","has_internship",
                 "dsa_hrs_week","project_hrs_week","gym_consistency","resume_score"],
                profile_tuple
            ))
            bg = generate_dataset(500)
            user_df = _profile_to_df(profile_dict)
            sv, feats = compute_shap(model, trace, user_df, bg)
            return shap_summary(sv, feats, top_n=8)

        profile_tuple = tuple(profile.values())
        shap_df = get_shap_cached(profile_tuple)

        colors = ["#3fb950" if v >= 0 else "#f85149" for v in shap_df["Impact"]]
        fig_shap = go.Figure(go.Bar(
            x=shap_df["Impact"] * 100, y=shap_df["Feature"], orientation="h",
            marker_color=colors,
            text=[f"{v*100:+.1f}%" for v in shap_df["Impact"]], textposition="outside",
        ))
        fig_shap.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", font_color="#c9d1d9",
            height=320,
            xaxis=dict(title="SHAP Impact on Probability (%)", color="#8892a4", gridcolor="#1e2433", zeroline=False),
            yaxis=dict(color="#c9d1d9"),
            margin=dict(t=10, b=40, l=10, r=60),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    except Exception as e:
        st.info(f"SHAP analysis skipped: {e}")

    st.markdown("""
    <div style='text-align:center; color:#3d4451; font-size:0.78rem; margin-top:32px;'>
        Sentinel v1 — Bayesian logistic regression (PyMC NUTS) + SHAP attribution.<br>
        Intercepts calibrated to documented selection rates. Probabilities are model estimates, not guarantees.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
