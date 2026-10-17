# bina.az Real-Estate Price Prediction & Tier Classification

[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![From Scratch](https://img.shields.io/badge/From--Scratch-Pure%20NumPy-success.svg)](https://numpy.org/)
[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B.svg)](https://binaaz-price-ml-nuhy5v73gecqfuyeh6kced.streamlit.app/)
[![Tests Passing](https://img.shields.io/badge/Pytest-51%20Passed-brightgreen.svg)](tests/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black%20%2B%20Flake8-black.svg)](https://github.com/psf/black)
[![Release Tag](https://img.shields.io/badge/Tag-v1.0--final-orange.svg)](https://github.com/Eldar2107/binaaz-price-ml/releases/tag/v1.0-final)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicting real-estate valuations in Baku using a **Decision Tree (CART)** and a **Soft-Margin Linear SVM (Pegasos)**, both built entirely from scratch in pure NumPy and benchmarked rigorously against scikit-learn.

Final project for **MLE-AI-201 Machine Learning**, engineered by **Eldar Samedov**, **Narmina Rzayeva**, and **Mirkenan Qedirli**. Real-world data sourced from [bina.az](https://bina.az/), Azerbaijan's premier real-estate portal (63,908 verified listings across Baku).

---

### 🌐 Interactive Web Demo
Experience real-time price estimation, interactive feature attribution, and tier classification:  
👉 **[binaaz-price-ml.streamlit.app](https://binaaz-price-ml-nuhy5v73gecqfuyeh6kced.streamlit.app/)**

---

## 📌 The Two Core Tasks

We formulate two complementary learning problems on the identical leak-free partition:

| Task | Objective | Formulation & Target | Primary Metric |
| :--- | :--- | :--- | :--- |
| **Task A: Regression** | How much is this listing worth? | Target $y = \ln(1 + \text{price})$; predictions exponentiated via $\exp(\hat{y}) - 1$ back to AZN | **RMSE (log)** & MAE (AZN) |
| **Task B: Classification** | Is it a Premium or Standard listing? | Target $y \in \{0, 1\}$ where $\text{Premium} = \mathbb{I}(\text{price} \ge 210,000 \text{ AZN})$ (train median) | **F1-Score** & ROC-AUC |

---

## 📊 Results at a Glance

All models are evaluated on the identical held-out test split (**9,588 listings**) and verified with **5-fold leak-free cross-validation** on the train+validation pool (**54,320 listings**). Every model — ours or scikit-learn's — is scored using identical NumPy evaluation logic in `src/evaluate.py`.

### Task A: Price Regression ($\ln(\text{price})$ Domain & Inverted AZN)

| Model | RMSE (log) ↓ | $R^2$ (log) ↑ | RMSE (AZN) ↓ | MAE (AZN) ↓ | 5-Fold CV $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Mean Predictor)** | 0.6384 | 0.0000 | 338,412 | 129,540 | 0.0000 ± 0.0000 |
| **Our Decision Tree (from scratch)** | **0.4647** | **0.6096** | **263,908** | **98,086** | 0.6249 ± 0.0043 |
| scikit-learn `DecisionTreeRegressor` | 0.3633 | 0.7614 | 233,671 | 74,607 | 0.7891 ± 0.0107 |

> *Note:* In regression, scikit-learn applies recursive floating-point mean MSE optimizations. Our tree uses discrete quantile pre-binning for speed, achieving solid $R^2 = 0.61$ without numerical drift.

### Task B: Premium vs. Standard Classification (Threshold: 210,000 AZN)

| Model | Accuracy ↑ | Precision ↑ | Recall ↑ | F1-Score ↑ | ROC-AUC ↑ | 5-Fold CV F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Floor** | 0.5060 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 ± 0.0000 |
| **Our Decision Tree (from scratch)** | **0.9217** | **0.9169** | **0.9252** | **0.9210** | **0.9304** | **0.9160 ± 0.0020** |
| scikit-learn `DecisionTreeClassifier` | 0.9161 | 0.9126 | 0.9180 | 0.9153 | 0.9072 | 0.9155 ± 0.0024 |
| **Our Pegasos SVM (from scratch)** | **0.8496** | **0.8473** | **0.8483** | **0.8478** | **0.9248** | **0.8497 ± 0.0034** |
| scikit-learn `LinearSVC` | 0.8431 | 0.8261 | 0.8642 | 0.8447 | 0.9278 | 0.8477 ± 0.0029 |

> 🏆 **Key Milestone:**
> - Our from-scratch **Decision Tree beats scikit-learn** on test F1 (**0.9210 vs 0.9153**) and ROC-AUC (**0.9304 vs 0.9072**).
> - Our from-scratch **Pegasos SVM beats scikit-learn LinearSVC** on test F1 (**0.8478 vs 0.8447**) and 5-fold CV F1 (**0.8497 vs 0.8477**).

---

## 💡 What We Learned & Engineering Highlights

1. **NumPy Vectorization Speedup ($\times 400$):**
   A naive split search over 164 features across 44,732 samples takes $>60\text{s}$ per node. By sorting feature arrays once and utilizing `np.cumsum` for dynamic left/right impurity updates (`_best_split`), we reduced node search to **0.15s**, enabling full-tree training within 11 seconds.
2. **Pegasos Sub-Gradient Convergence:**
   Implementing the primal Pegasos iteration $w_{t+1} = (1 - \eta_t \lambda) w_t + \eta_t y_i x_i$ with step size $\eta_t = \frac{1}{\lambda t}$ guarantees rapid convergence without storing large kernel matrices. Our solver matches the dual optimizer while tracking margin violations in $O(d)$ time.
3. **Random Fourier Features (RFF):**
   By approximating the Gaussian RBF kernel via randomized cosine/sine projections $z(x) = \sqrt{\frac{2}{D}} \cos(W x + b)$, Pegasos learned non-linear decision boundaries with linear-time training.
4. **Dominant Signals in Baku Real-Estate:**
   Feature importance analysis reveals that `area_m2` accounts for **~48.2%** of split decisions, followed by distance to city center (`dist_center_km` at **~21.7%**), building category (Yeni tikili vs. Həyət evi), and out-of-fold district target encodings.
5. **Anatomy of Prediction Errors:**
   - **Boundary Ambiguity:** Classification error peaks sharply in the 200k–350k AZN bracket (13.54% error rate), directly adjacent to the 210k threshold.
   - **Luxury Ceiling:** Properties $>1\text{M}$ AZN suffer from negative bias (underprediction by ~1.04M AZN) because tabular features lack qualitative factors like sea view, luxury renovation, and private compound amenities.

---

## ⚡ Quick Start

### 1. Requirements & Virtual Environment
Requires **Python 3.10, 3.11, or 3.12**.

**PowerShell (Windows):**
```powershell
# Clone the repository
git clone https://github.com/Eldar2107/binaaz-price-ml.git
cd binaaz-price-ml

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

**Bash (Linux / macOS):**
```bash
git clone https://github.com/Eldar2107/binaaz-price-ml.git
cd binaaz-price-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the dataset from Kaggle and place it in the `data/` directory (see [data/README.md](data/README.md) for SHA-256 integrity verification):
```bash
# Expected path:
data/bina_az_sale.csv
```

### 3. Automated Tests
Run the complete test suite (51 unit and integration tests covering data preparation, split purity, tree invariants, and Pegasos updates):
```bash
pytest tests/ -v
```

### 4. Reproduce Complete Pipeline
One command executes data cleaning, leak-free feature encoding, model training, baselines, 5-fold CV, error analysis, and figure generation:
```bash
# Full reproduction run
python -m src.run_all

# Quick check (fast mode with subset folds)
python -m src.run_all --fast

# Data prep and EDA only
python -m src.run_all --only-data
```

### 5. Launch Web UI Locally
```bash
streamlit run streamlit_app.py
```

---

## 🏗️ Pipeline Architecture

The entire project operates as a unified, modular machine learning pipeline:

```
                    ┌─────────────────────────┐
                    │  data/bina_az_sale.csv  │ (100,775 raw listings)
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    src/data_prep.py     │ • URL deduplication (64,454)
                    │  (Data Cleaning & EDA)  │ • Coordinate & outlier filters
                    └────────────┬────────────┘ • 63,908 clean listings
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Leak-Free Split Engine │ • 70% Train (44,732)
                    │     (Stratified Decile) │ • 15% Val   (9,588)
                    └────────────┬────────────┘ • 15% Test  (9,588)
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
┌─────────────────────────┐                     ┌─────────────────────────┐
│   Task A: Regression    │                     │ Task B: Classification  │
│  Target: log1p(price)   │                     │  Threshold: 210,000 AZN │
└───────────┬─────────────┘                     └───────────┬─────────────┘
            │                                               │
            ├───────────────────────┬───────────────────────┤
            ▼                       ▼                       ▼
┌───────────────────────┐ ┌───────────────────┐ ┌─────────────────────────┐
│  src/decision_tree.py │ │    src/svm.py     │ │   scikit-learn Baseline │
│  • Vectorized split   │ │ • Pegasos SGD     │ │ • DecisionTree Reg/Clf  │
│  • Gini/Entropy/MSE   │ │ • Sub-gradient    │ │ • LinearSVC             │
│  • Pre-pruning bounds │ │ • RFF Kernel      │ │ • SGDClassifier         │
└───────────┬───────────┘ └─────────┬─────────┘ └───────────┬─────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    ▼
                    ┌─────────────────────────┐
                    │     src/evaluate.py     │ • RMSE, R², MAE (Task A)
                    │       src/cv.py         │ • Acc, Prec, Rec, F1, ROC
                    │   src/error_analysis.py │ • 5-Fold Leak-Free CV
                    └────────────┬────────────┘ • Subgroup segmentation
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
┌─────────────────────────┐                     ┌─────────────────────────┐
│     report/ & tables/   │                     │    streamlit_app.py     │
│ • LaTeX IEEE Paper      │                     │ • Interactive Web UI    │
│ • Generated Figures/CSV │                     │ • Real-Time Inference   │
└─────────────────────────┘                     └─────────────────────────┘
```

---

## 🛡️ Leakage Prevention Protocol

Data leakage is the most prevalent flaw in real-estate valuation pipelines. We implement strict, mathematically verified isolation protocols:

1. **Deterministic Feature Whitelist:** Raw listing identifiers (`id`, `url`, `title`, `description`, `seller_name`, `phone`) are excluded prior to vectorization.
2. **Train-Only Standardization:** Means $\mu_{\text{train}}$ and variances $\sigma^2_{\text{train}}$ are computed solely on the training fold; validation and test sets are transformed using frozen parameters:
   $$\hat{x} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$
3. **Threshold Containment:** The 210,000 AZN tier threshold is computed as the strict median of the training split ($y_{\text{train}}$) and applied as a constant to validation and test sets.
4. **Cross-Validation Isolation:** Each fold in `src/cv.py` re-fits all feature scalers and imputers independently from scratch.

---

## 📂 Project Structure

```
.
├── .github/workflows/          # CI pipeline automation
├── data/
│   ├── README.md               # Dataset source, schema & checksum verification
│   └── bina_az_sale.csv        # Local dataset (git-ignored)
├── models/
│   └── bundle.pkl              # Serialized trained models & metadata for UI
├── presentation/
│   ├── slides.pdf              # Defense presentation deck
│   └── make_slides.py          # Automated PPTX slide generator
├── report/
│   ├── figures/                # EDA distributions, ROC curves, loss trajectories
│   ├── tables/                 # Benchmark markdown and CSV results
│   └── report.tex              # IEEE conference-style final paper
├── scripts/
│   ├── tree_experiments.py     # Depth, criterion, and pruning sweeps
│   └── svm_experiments.py      # Lambda regularization & RFF sweeps
├── src/
│   ├── data_prep.py            # Cleaning, parsing, outlier handling, leak-free split
│   ├── decision_tree.py        # From-scratch Decision Tree (Vectorized CART)
│   ├── svm.py                  # From-scratch Pegasos Linear SVM & RFF Mapper
│   ├── evaluate.py             # Pure NumPy evaluation metrics
│   ├── cv.py                   # 5-fold cross-validation engine
│   ├── error_analysis.py       # Segmented residual & boundary error analysis
│   ├── plots.py                # Publication-quality visualizer
│   ├── run_all.py              # Single entry-point reproduction driver
│   └── serve.py                # Standalone HTTP / programmatic inference API
├── tests/
│   ├── test_data_prep.py       # Data split, leakage, and parser unit tests
│   ├── test_tree.py            # Decision tree split and boundary tests
│   └── test_svm.py             # Pegasos subgradient and margin tests
├── streamlit_app.py            # Web application UI (Dark-Navy / Teal theme)
├── contribution_report.md      # Individual task distribution and verification
├── requirements.txt            # Locked Python package versions
└── README.md                   # Project documentation
```

---

## 👥 Team & Work Breakdown

| Team Member | Role | Core Contributions | Effort |
| :--- | :--- | :--- | :---: |
| **Eldar Samedov** | Member 1: Data, Preprocessing & Evaluation | • Data parsing, outlier clipping, and leak-free split engine<br>• NumPy evaluation suite (`src/evaluate.py`) & 5-fold CV (`src/cv.py`)<br>• Model-agnostic error analysis & residual segmentation<br>• Streamlit Web UI implementation & cloud deployment | **34%** |
| **Narmina Rzayeva** | Member 2: Decision Tree from Scratch | • Pure NumPy CART implementation (`src/decision_tree.py`)<br>• Vectorized split search using prefix cumulative sums (`_best_split`)<br>• Gini impurity, Entropy, and Mean Squared Error criteria<br>• Depth & min-leaf hyperparameter ablation study | **33%** |
| **Mirkenan Qedirli** | Member 3: Pegasos SVM & Pipelines | • Pegasos stochastic sub-gradient descent solver (`src/svm.py`)<br>• Primal objective tracking, margin metrics & support vector detection<br>• Random Fourier Features (`RFFMapper`) non-linear approximation<br>• Regularization $\lambda$ parameter sweep & scikit-learn benchmarking | **33%** |

---

## 📜 Academic Integrity & Citation

This codebase was developed as part of the **MLE-AI-201: Machine Learning** course curriculum. All model logic in `src/decision_tree.py` and `src/svm.py` was authored from fundamental mathematical definitions in NumPy without relying on high-level machine learning frameworks.

```bibtex
@misc{samedov2026binaaz,
  author = {Samedov, Eldar and Rzayeva, Narmina and Qedirli, Mirkenan},
  title = {Bina.az Real-Estate Valuation and Tier Classification via From-Scratch CART and Pegasos SVM},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Eldar2107/binaaz-price-ml}}
}
```


# OGRUUUUUUU IDEYA OGRUSU
