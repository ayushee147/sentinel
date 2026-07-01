# 🎯 Sentinel — Uncertainty-Aware Personal Growth Optimizer

> *"I have limited time and many goals. What should I prioritize next?"*

Sentinel is a Bayesian inference system that helps students and early-career professionals decide **how to allocate their time** across competing goals — DSA practice, projects, fitness, and career-building activities — by estimating the **probabilistic impact** of each choice on long-term outcomes like internship success.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentinel-ayushee.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyMC](https://img.shields.io/badge/PyMC-5.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What It Does

Given a user profile (CGPA, LeetCode count, projects, weekly time allocation, etc.), Sentinel:

1. **Estimates internship success probability** with calibrated 95% credible intervals via Bayesian logistic regression (PyMC + NUTS sampler)
2. **Explains what's driving the probability** using SHAP feature attribution (KernelExplainer)
3. **Runs counterfactual analysis** — "if I did X instead, how would my probability change?"
4. **Recommends the highest-impact next action** with a personalised rationale

## Example Output

```
Current Profile: Ayushee K.
  CGPA: 7.75 | LeetCode: 75 | Projects: 4 | Prior Internship: Yes

Internship Success Probability: 48% [39% – 57%]   (95% credible interval)

Top Counterfactuals:
  Increase DSA to 10 hrs/week     → +18pp  (66%)
  Complete & deploy Sentinel      → +11pp  (59%)
  Reach 250 LeetCode problems     → +17pp  (65%)
  CGPA drops below 7.0            →  -9pp  (39%)  ⚠️

SHAP Top Features:
  LeetCode Problems   +12.3%
  CGPA                 +8.1%
  Prior Internship     +6.4%
  Resume Score         +4.2%
```

---

## Architecture

```
generate_data.py   →  Synthetic dataset (1,200 student profiles, ground-truth logistic model)
model.py           →  Bayesian logistic regression (PyMC, NUTS, weakly informative priors)
explainability.py  →  SHAP KernelExplainer wrapping posterior mean predictor
app.py             →  Streamlit UI (gauge, waterfall chart, SHAP bar, donut)
train_model.py     →  One-time training script
```

### Model Details

- **Likelihood:** Bernoulli with logistic link
- **Priors:** `intercept ~ Normal(0,1)`, `betas ~ Normal(0,1)` (weakly informative)
- **Sampler:** NUTS, 2 chains × 1000 draws (+ 1000 tune), target_accept=0.9
- **Features:** CGPA, log(LeetCode+1), project count, prior internship, DSA hrs/week, project hrs/week, gym consistency, resume score
- **Uncertainty:** Full posterior predictive for credible intervals; posterior mean betas for SHAP (speed)

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/ayushee147/sentinel.git
cd sentinel

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Train the model (run once, ~3–5 min)
python train_model.py

# 4. Launch the app
streamlit run app.py
```

---

## Tech Stack

| Layer | Library |
|---|---|
| Bayesian inference | PyMC 5, ArviZ |
| Tensor backend | PyTensor |
| Explainability | SHAP (KernelExplainer) |
| Frontend | Streamlit |
| Visualisation | Plotly |
| Data | NumPy, Pandas |

---

## Limitations & Honest Notes

- The synthetic dataset is generated from a logistic model with hand-tuned weights — it reflects plausible patterns but is **not trained on real placement outcomes**
- Probabilities are **model estimates, not guarantees**
- SHAP uses the posterior mean for speed; full posterior SHAP would be more correct but ~100× slower

---

## Future Work

- [ ] Multi-goal optimization (internship + fitness + learning simultaneously)
- [ ] User-uploaded historical data for personalized model fine-tuning
- [ ] Time-series tracking (see how probability evolves week-over-week)
- [ ] Pipeline simulation and hazard detection

---

*Built by Ayushee Kaul | B.Tech ECE, IIIT-D | [LinkedIn](https://www.linkedin.com/in/ayushee-kaul-58146b284/)*
