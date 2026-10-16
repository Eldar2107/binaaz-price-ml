# Contribution Report

Keep this file in the repo root. It is checked at the demo. Be honest and
specific — "helped with everything" is not acceptable. One row per member,
plus a short per-member paragraph.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Samedov, Eldar (Member 1) | @Eldar2107 | Data cleaning, EDA, metrics, leak-free pipeline, error analysis |
| Rzayeva, Narmina (Member 2) | @NarminaRzayeva | Decision tree from scratch, vectorized splitting, tree tuning |
| Qedirli, Mirkenan (Member 3) | @MirkenanQedirli | Pegasos SVM from scratch, RFF non-linear kernel, lambda sweeps |

## Per-member summary
**Samedov, Eldar —** Designed and implemented the end-to-end data pipeline in `src/data_prep.py`, including robust parsing, domain-based outlier clipping, and strict leakage elimination. Built the NumPy-based evaluation suite (`src/evaluate.py`), cross-validation framework (`src/cv.py`), and automated model-agnostic error analysis (`src/error_analysis.py`). Coordinated `run_all.py` pipeline and generated comprehensive EDA and error breakdown figures.

**Rzayeva, Narmina —** Implemented the binary/multi-class classification and regression decision tree from scratch in `src/decision_tree.py`. Vectorized the candidate threshold search using prefix cumulative sums (`_best_split`), achieving sub-second training on the cleaned dataset. Configured tree hyperparameters (depth, split, leaf, impurity decrease) and created hyperparameter ablation and overfitting curves in `scripts/tree_experiments.py` along with unit test validation (`tests/test_tree.py`).

**Qedirli, Mirkenan —** Developed the stochastic subgradient Pegasos solver for soft-margin SVM in `src/svm.py`, supporting primal objective tracking, margin evaluation, and support vector identification. Implemented Random Fourier Features (`RFFMapper`) to project non-linear boundaries into linear space. Designed the regularization parameter sweep (`lambda_sweep`), matched comparisons with scikit-learn baselines (`LinearSVC`, `SGDClassifier`, `SVC(rbf)`), and constructed the unit test suite (`tests/test_svm.py`).

## Rough effort split
Give an approximate percentage per member (should sum to ~100%) and note any
imbalance and why.

| Member | Approx. % |
|--------|-----------|
| Samedov, Eldar | 34% |
| Rzayeva, Narmina | 33% |
| Qedirli, Mirkenan | 33% |
