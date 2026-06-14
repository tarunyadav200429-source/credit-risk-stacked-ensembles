# Credit Risk Prediction via Stacked Ensembles and Bayesian Optimization

### A Statistically Validated Cross-Dataset Comparison

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-2.0-189632">
  <img alt="Datasets" src="https://img.shields.io/badge/UCI_datasets-4-blue">
  <img alt="Classifiers" src="https://img.shields.io/badge/classifiers-8-orange">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

Integrated M.Sc. (Statistics) thesis, **Central University of Rajasthan** (2026). This repository
contains the thesis, the reproducible notebook bundle, and the four public datasets behind it.

> **Research question.** *Among a representative set of credit-risk classifiers, which model performs
> most consistently across multiple UCI datasets, and is its advantage statistically significant
> relative to the others?*
>
> **Answer.** A **Stacked Ensemble** — whose meta-learner is chosen by F1-macro on the out-of-fold
> predictions of five diverse base learners — is the most consistently strong classifier across the
> four datasets. Its advantage is backed by a **Friedman test (p < 0.001 on every metric)** and a
> **Wilcoxon–Holm post-hoc that establishes Stacker > XGBoost at α = 0.05 on AUC (p = 0.0008)**.

📄 **[Read the full thesis (PDF)](thesis/Credit_Risk_Prediction_Thesis.pdf)**

---

## 🎯 Objective & outcome

**Objective —** find which credit-risk classifier is the **most consistent across multiple
datasets**, and test whether its advantage is **statistically significant** (not just a lucky
average on one dataset).

**What this project does —** runs **one uniform pipeline** over **8 classifiers × 4 UCI credit
datasets × 4 feature-selection variants (128 model rows)** with Bayesian hyperparameter tuning,
then submits the whole benchmark to **non-parametric significance testing** (Friedman +
Wilcoxon–Holm post-hoc).

**Did we achieve it? ✅ Yes.** A **Stacked Ensemble** is the most consistent performer (highest
mean F1-macro **and** AUC, and the per-dataset best AUC on **all four** datasets), and its edge
is **statistically validated** — Friedman **p < 0.001** on every metric, with Wilcoxon–Holm
establishing **Stacker > XGBoost (p = 0.0008)** on AUC.

---

## Headline Result

Averaged over the four feature-selection variants on each dataset and then over the four datasets,
the Stacked Ensemble has the **highest mean F1-macro and the highest mean AUC** of all eight models
— and it is the per-dataset best on **AUC for all four datasets**, a sweep no competitor achieves.

| Model | Mean F1-macro | Mean AUC |
|-------|:-------------:|:--------:|
| **Stacked Ensemble** | **0.7790** | **0.8549** |
| Gradient Boosting | 0.7728 | 0.8412 |
| XGBoost | 0.7723 | 0.8452 |
| Random Forest | 0.7622 | 0.8512 |
| Logistic Regression | 0.7592 | 0.8269 |
| SVM-RBF | 0.7559 | 0.8224 |
| Decision Tree | 0.7434 | 0.8027 |
| k-Nearest Neighbours | 0.7336 | 0.8042 |

*Table 5.6 — cross-dataset mean per classifier.*

---

## What Was Built

A **uniform pipeline** that applies the *same* preprocessing, feature-selection sweep, hyperparameter
tuning, and evaluation suite to **8 classifiers × 4 datasets × 4 feature-selection variants = 128
model rows**, then submits the whole benchmark to non-parametric significance testing.

**Eight classifiers.** Logistic Regression · k-Nearest Neighbours · Decision Tree · SVM (RBF) ·
Random Forest · Gradient Boosting · XGBoost · **Stacked Ensemble**.

**Four UCI credit datasets.**

| Dataset | Records | Features | Class balance (good / default) | UCI ID |
|---------|--------:|---------:|--------------------------------|:------:|
| Australian Credit Approval | 690 | 14 | 55.5 / 44.5 (balanced) | 143 |
| Japanese Credit Screening | 690 | 15 | 55.5 / 44.5 (balanced) | 27 |
| German Credit Data | 1,000 | 20 | 70.0 / 30.0 (imbalanced) | 144 |
| Taiwan Default of Credit Card Clients | 30,000 | 23 | 78.0 / 22.0 (imbalanced) | 350 |

**Four feature-selection strategies** swept per dataset: Full · Recursive Feature Elimination (RFE) ·
L1-penalised Logistic Regression (Lasso-LR) · Forward Selection.

---

## Methodology

```
            Preprocess (clean · one-hot · 75/25 stratified split · standardise · SMOTE on imbalanced)
                                           │
        ┌──────────────────────────────────┴──────────────────────────────────┐
        │   BASE LAYER (5 learners, RF & XGBoost tuned by Bayesian optimisation) │
        │     Random Forest · Gradient Boosting · XGBoost · KNN · SVM-RBF        │
        └──────────────────────────────────┬──────────────────────────────────┘
                          5-fold out-of-fold predictions  →  meta-feature matrix Z ∈ ℝ^(N×5)
                                           │
        ┌──────────────────────────────────┴──────────────────────────────────┐
        │   META LAYER (5 candidates) — best chosen by F1-macro                 │
        │     LR · RF · XGBoost · MLP · GA-MaxAccLogit                           │
        └──────────────────────────────────┬──────────────────────────────────┘
                                  Stacker prediction (named after winning meta, e.g. "Stacker (LR)")
```

- **Class imbalance (German, Taiwan only).** SMOTE oversampling on the training partition, followed by
  **GHOST** threshold tuning that picks the decision threshold by maximising Cohen's κ. Balanced
  datasets keep the fixed threshold t = 0.5.
- **Bayesian optimisation.** Random Forest and XGBoost are tuned with a Gaussian-process surrogate and
  an Expected-Improvement acquisition function (12–25 trials per dataset/FS pair), optimising
  cross-validated F1-macro.
- **GA-MaxAccLogit — the methodological contribution.** A logistic meta-learner whose coefficients are
  evolved by a real-coded genetic algorithm. It adapts the **ProfLogit** method of Stripling et al.
  (2018), keeping the genetic machinery intact but **replacing the expected-profit fitness with
  F1-macro − L1 penalty**. Implementation in [`thesis/latex/Appendices/AppendixA.tex`](thesis/latex/Appendices/AppendixA.tex).
- **Evaluation.** Two headline metrics (F1-macro, AUC) plus a **ten-measure diagnostic suite**
  (Accuracy, Type-I/II error, EMCC at the 5:1 credit cost ratio, G-Mean, Discriminant Power, F-Score,
  Cohen's κ, Youden's J) so that no single number can hide a weakness.

---

## Statistical Validation

Each `(dataset × feature-selection)` pair is treated as one evaluation **block**, giving **n = 16
blocks** for **k = 8** classifiers — more powerful than a four-dataset test.

**Average ranks (lower is better), Table 5.7:**

| Model | F1-macro | AUC | Accuracy | G-Mean |
|-------|:--------:|:---:|:--------:|:------:|
| **Stacked Ensemble** | **2.00** | **1.38** | 3.38 | **2.19** |
| Gradient Boosting | 2.75 | 3.75 | 3.09 | 3.84 |
| XGBoost | 2.75 | 3.41 | 3.41 | 3.66 |
| Random Forest | 4.50 | 2.59 | 4.97 | 4.00 |
| SVM-RBF | 4.56 | 5.69 | 4.38 | 5.09 |
| Logistic Regression | 4.94 | 5.44 | 4.50 | 5.00 |
| Decision Tree | 6.00 | 6.81 | 5.75 | 5.78 |
| k-Nearest Neighbours | 7.03 | 6.94 | 6.53 | 6.44 |

**Friedman test (Table 5.8)** — rejects the global null "all eight classifiers perform equally" on
every metric (5% critical values: χ²(7) = 14.067, F(7,105) = 2.098):

| Metric | χ²_F | Iman–Davenport F_F | p-value | Verdict |
|--------|:----:|:------------------:|:-------:|---------|
| F1-macro | 46.20 | 10.53 | < 0.001 | reject H₀ |
| AUC | 76.69 | 32.58 | < 0.001 | reject H₀ |
| Accuracy | 28.62 | 5.15 | < 0.001 | reject H₀ |
| G-Mean | 34.07 | 6.56 | < 0.001 | reject H₀ |

**Wilcoxon signed-rank + Holm post-hoc** (28 pairs). On AUC the Stacker is significantly better than
**6 of 7** competitors at α = 0.05 — including the decisive **Stacker > XGBoost (p = 0.0008)**, the
comparison a four-dataset analysis could not resolve. On F1-macro the leading trio (Stacker, XGBoost,
Gradient Boosting) is statistically tight.

---

## Per-Dataset Winners

Best `(classifier, feature-selection)` row by F1-macro on each dataset (Table 5.1):

| Dataset | Best model | FS | Accuracy | F1-macro | AUC |
|---------|------------|----|:--------:|:--------:|:---:|
| Australian | XGBoost | Lasso-LR | 0.877 | 0.876 | 0.937 |
| Japanese | **Stacker (LR)** | RFE | 0.928 | 0.926 | 0.970 |
| German | XGBoost | RFE | 0.750 | 0.708 | 0.748 |
| Taiwan | **Stacker (MLP)** | Lasso-LR | 0.798 | 0.704 | 0.773 |

The Stacker **wins** Japanese and Taiwan and is the **runner-up** on Australian and German — it is
never the weak link on any dataset. No single feature-selection method dominates across datasets.

---

## Repository Structure

```
credit-risk-stacked-ensembles/
├── README.md
├── LICENSE
├── thesis/
│   ├── Credit_Risk_Prediction_Thesis.pdf   # the full thesis
│   └── latex/                              # LaTeX source (chapters, appendix code, references.bib)
├── notebooks/                             # reproducible Colab bundle
│   ├── 01_australian_credit.ipynb
│   ├── 02_japanese_credit.ipynb
│   ├── 03_german_credit.ipynb
│   ├── 04_taiwan_credit.ipynb
│   ├── 05_friedman_test.ipynb             # Friedman + Iman–Davenport on the 16 blocks
│   └── 06_wilcoxon_holm_posthoc.ipynb     # pairwise Wilcoxon signed-rank with Holm correction
└── data/                                 # the four UCI datasets (CSV)
    ├── australian.csv
    ├── japanese_crx.csv
    ├── german.csv
    └── taiwan_default.csv
```

## Reproducing the Results

The notebooks are written for **Google Colab** and run end-to-end on the open UCI data.

1. Open a notebook in `notebooks/` (per-dataset runs `01`–`04`, statistical tests `05`–`06`).
2. Each per-dataset notebook runs the full pipeline — feature-selection sweep, SMOTE + GHOST on the
   imbalanced sets, Bayesian-optimised bases, the five-candidate stacker — and reports the eleven-metric
   scoreboard.
3. `05` and `06` consume the per-dataset scores and produce the Friedman statistics and the
   Wilcoxon–Holm pairwise verdicts.

Key libraries (thesis §4.4): `scikit-learn 1.3`, `xgboost 2.0`, `imbalanced-learn 0.11` (SMOTE),
`scikit-optimize 0.9` (Bayesian optimisation), `ghostml 0.4` (GHOST), `scipy.stats` (Friedman /
Wilcoxon), `ucimlrepo 0.0.7` (data fetch). A fixed seed (42) is used throughout for reproducibility.

---

## Limitations & Future Work

*(thesis §6.4–6.5)*

- **Four datasets** is at the low end of what the Friedman test recommends; the 16-block design
  compensates but the constraint is real.
- **GHOST threshold tuning** is mildly stochastic, so threshold-dependent metrics wobble by ≈ ±0.02 on
  German and Taiwan — which is why every *formal* claim rests on the reproducible AUC.
- **Single train/test seed** (42); a multi-seed study would add confidence intervals.
- **Future direction:** swap the GA-MaxAccLogit fitness back to **expected profit** (restore ProfLogit's
  EMP objective) so the stacker optimises the lender's profit-and-loss rather than a symmetric metric,
  and validate on more credit datasets.

---

## Author

**Tarun Yadav** — Integrated M.Sc. (Statistics), Central University of Rajasthan
Supervisors: Mr. Rahul Singh Chauhan (Dusker AI) · Dr. K. Satish Kumar (Dept. of Statistics, CURAJ)

🔗 [LinkedIn](https://www.linkedin.com/in/tarun-yadav-8a9a78415/)

## Citation

```bibtex
@mastersthesis{yadav2026creditrisk,
  author = {Tarun Yadav},
  title  = {Credit Risk Prediction via Stacked Ensembles and Bayesian Optimization:
            A Statistically Validated Cross-Dataset Comparison},
  school = {Central University of Rajasthan},
  year   = {2026}
}
```

## License

Code and thesis text released under the [MIT License](LICENSE). The four datasets are the property of
the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) under their respective terms.
Methods build on Wolpert (stacked generalisation, 1992), Chawla et al. (SMOTE, 2002), Chen & Guestrin
(XGBoost, 2016), Snoek et al. (Bayesian optimisation, 2012), Stripling et al. (ProfLogit, 2018),
Esposito et al. (GHOST, 2021), and Demšar (statistical comparisons, 2006).
