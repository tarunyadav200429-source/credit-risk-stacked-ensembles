# Tarun Yadav — Integrated M.Sc. (Statistics) Thesis

**Title:** *Credit Risk Prediction via Stacked Ensembles and Bayesian
Optimization: A Statistically Validated Cross-Dataset Comparison*

**Candidate:** Tarun Yadav
**Supervisor:** Dr. K. Satish, Department of Statistics, Central University of Rajasthan
**Internship:** Mr. Rahul S. Chauhan, Dusker AI

---

## Project layout

```
Tarun_Credit_Risk_Thesis/
├── main.tex                    -- master document
├── title.tex                   -- title page
├── README.md                   -- this file
├── references.bib              -- BibTeX bibliography
├── Frontmatter/
│   ├── Certificate.tex
│   ├── SatishCertificate.tex
│   ├── DuskerCertificate.tex
│   ├── Declaration.tex
│   └── Acknowledgements.tex
├── Chapters/
│   ├── Chapter_1/Chapter_1.tex  -- Introduction
│   ├── Chapter_2/Chapter_2.tex  -- Classification Algorithms
│   ├── Chapter_3/Chapter_3.tex  -- Methodology
│   ├── Chapter_4/Chapter_4.tex  -- Experimental Setup
│   ├── Chapter_5/Chapter_5.tex  -- Results & Statistical Analysis
│   └── Chapter_6/Chapter_6.tex  -- Conclusion
├── Appendices/
│   └── AppendixA.tex            -- Python code listings
└── Images/                      -- figures (curaj_logo.png, etc.)
```

## Compilation

The project is designed to compile against the **Thesis.cls** class
file used by previous CUR Department of Statistics theses (the same
class used by the EM-framework thesis whose `thesis.tex` was provided
as the reference format).  Copy `Thesis.cls` and the .sty files
(`booktabs.sty`, `cleveref.sty`, `fancyhdr.sty`, `listings.sty`,
`mcode.sty`, `natbib.sty`, `rotating.sty`, `setspace.sty`,
`vmargin.sty`) from the reference Overleaf project into this folder
before compiling.  Place the CUR seal at `Images/cur_logo.png`.

Build with `pdflatex` + `bibtex`:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Chapter overview

| Ch. | Title | Pages (target) | Main contribution |
|---|---|---|---|
| 1 | Introduction | ~5 | Motivation, objectives, contributions |
| 2 | Classification Algorithms | ~10 | Math of 8 classifiers, stacking |
| 3 | Methodology | ~10 | FS sweep, SMOTE+GHOST, Bayesian opt, stacker config, GA-MaxAccLogit |
| 4 | Experimental Setup | ~7 | 4 datasets, preprocessing, metrics, implementation |
| 5 | Results & Statistical Analysis | ~10 | Per-dataset + cross-dataset + Friedman + Wilcoxon-Holm |
| 6 | Conclusion | ~4 | Summary, limitations, future work |
| A | Python Code Listings | ~6 | Friedman, Wilcoxon-Holm, GA-MaxAccLogit, Stacker |

Total: ~52 pages plus front matter and bibliography.

## Notes

- A compiled copy of this thesis is available at
  `../Credit_Risk_Prediction_Thesis.pdf` (no LaTeX toolchain required to read it).
- All numerical results in Chapter 5 come from the Colab notebooks in
  [`../../notebooks/`](../../notebooks) — `05_friedman_test.ipynb` and
  `06_wilcoxon_holm_posthoc.ipynb` for the statistical tests, and `01`–`04` for the
  per-dataset runs.
- The Python code reproduced in Appendix A (Friedman, Wilcoxon–Holm, GA-MaxAccLogit,
  Stacker) mirrors the implementations in those notebooks.
