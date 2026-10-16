# MLE-AI-201 Machine Learning — Final Project

Build **two models from scratch** (a decision tree and an SVM) on the
**bina.az** real-estate dataset, then benchmark honestly against
scikit-learn.

- **Team:** Eldar Samedov, Narmina Rzayeva, Mirkenan Qedirli
- **Repository:** https://github.com/Eldar2107/binaaz-price-ml
- **Tag:** `v1.0-final`
- **Due:** **Sun 18 Oct 2026, 23:59 (Baku)**

## What you submit (on Moodle)
Exactly **three** things; **one member submits** for the whole team:
1. **GitHub link** to your repo (tagged `v1.0-final`).
2. **Report PDF** (IEEE two-column — compiled from `report/report.tex`).
3. **Slides PDF** (`presentation/slides.pdf`).

Keep `contribution_report.md` in the repo root (checked at the demo).

## The two tasks
- **Task A — Regression:** predict `price` (modeled in `log(price)` domain).
- **Task B — Classification:** predict a derived **price tier**
  (premium vs standard, median split on the training set).

## Golden rule
Your **decision tree** and your **SVM** are implemented in **pure NumPy code**.
scikit-learn is used solely for comparison baselines (`DecisionTreeClassifier/Regressor`, `LinearSVC`, `SGDClassifier`, `SVC(rbf)`). For SVM, optimization is performed via the **Pegasos** stochastic sub-gradient descent algorithm (with primal objective tracking and optional Random Fourier Features).

## Repo layout
```
.
├── README.md               # this file
├── requirements.txt        # pinned dependencies
├── .gitignore
├── contribution_report.md  # who did what (keep in repo root)
├── data/                   # dataset folder (not committed — see data/README.md)
│   └── README.md
├── src/                    # from-scratch models and pipeline
│   ├── data_prep.py        #   loading, cleaning, split, tier label, standardization
│   ├── decision_tree.py    #   DecisionTree (Gini/entropy/MSE, vectorized split)
│   ├── svm.py              #   PegasosSVM (hinge-loss sub-gradient, RFF, sweep)
│   ├── evaluate.py         #   NumPy metrics + comparison helpers
│   ├── cv.py               #   5-fold leak-free cross-validation
│   ├── error_analysis.py   #   model-agnostic error segmentation
│   ├── plots.py            #   confusion matrix, ROC, error plots
│   └── run_all.py          #   ONE command reproduces every number/figure
├── tests/                  # automated unit tests (pytest)
├── scripts/                # hyperparameter and ablation experiments
├── report/                 # IEEE report & outputs
│   ├── report.tex          #   conference paper source
│   ├── figures/            #   EDA & evaluation figures
│   └── tables/             #   test-set & CV benchmark tables
└── presentation/           # presentation slides and documentation
```

## Quick start

### 1. Environment Setup
**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Dataset Setup
Download `bina_az_sale.csv` from Kaggle and place it in the `data/` directory (see `data/README.md` for checksum details):
```
data/bina_az_sale.csv
```

### 3. Run Pipeline & Tests
Run automated test suite:
```bash
pytest tests/ -v
```

Execute the full end-to-end pipeline (EDA, training, baselines, 5-fold CV, error analysis, figures, and tables):
```bash
python -m src.run_all
```
