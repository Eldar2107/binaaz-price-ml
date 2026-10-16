# Baku Real Estate Price Prediction & Classification
## Machine Learning Final Project — Team Presentation
**MLE-AI-201, Cohort II**

**Team Members:**
- Eldar Samedov (Member 1: Data, Preprocessing, Evaluation, Error Analysis)
- Narmina Rzayeva (Member 2: Decision Tree From Scratch, Tree Sweeps)
- Mirkenan Qedirli (Member 3: Pegasos SVM From Scratch, RFF, Optimization)

**Repository:** `https://github.com/Eldar2107/binaaz-price-ml` (Tag: `v1.0-final`)

---

## Slide 1: Problem Definition & Tasks
- **Dataset:** `bina.az` real-estate listings in Baku and surrounding areas (~100k raw listings).
- **Task A — Continuous Price Regression:**
  - Predicting property price in AZN.
  - Target transformed into log space: $y = \ln(\text{price})$ to stabilize heavy positive skew (raw skew $6.18 \to$ log skew $0.25$).
  - Evaluated on RMSE(log), $R^2$, and back-transformed AZN metrics (RMSE, MAE).
- **Task B — Binary Price Tier Classification:**
  - Premium ($1$) vs. Standard ($0$).
  - Derived strictly from training set median threshold: $\text{Threshold} = 210,000\text{ AZN}$.
  - Perfectly balanced target ($49.4\%$ premium / $50.6\%$ standard across all splits).
  - Evaluated on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## Slide 2: Data Cleaning, Preprocessing & Leakage Prevention
- **Deduplication:**
  - Scraped dump holds identical URLs at multiple timestamps.
  - Kept latest scrape by URL: removed 36,321 duplicate rows ($100,775 \to 64,454$).
- **Strict Leakage Elimination:**
  - Dropped target-derived columns: `unit_price` (AZN/m²) and `total_price`.
  - Dropped post-listing / promotion features: `views`, `updated`, `datetime_scrape`, `vip`, `featured`.
- **Domain-Specific Outlier Filtering (No Statistical Leakage):**
  - Price: $[10,000, 5,000,000]\text{ AZN}$ (457 dropped).
  - Area: $[10, 20,000]\text{ m}^2$ (88 dropped).
  - Geospatial: bounding box of Azerbaijan ($[38.3, 42.0]^\circ\text{N}$, $[44.7, 51.0]^\circ\text{E}$).
- **Feature Engineering & Imputation:**
  - Haversine distance to Baku center ($40.4093^\circ\text{N}, 49.8671^\circ\text{E}$).
  - Sentinel zero imputation + explicit `*_missing` indicator flags.
  - Final dataset: $63,908$ rows $\times$ $164$ features ($70\%$ train: 44,732; $15\%$ val: 9,588; $15\%$ test: 9,588).

---

## Slide 3: Decision Tree From Scratch (NumPy)
- **Impurity Criteria:**
  - Classification: Gini ($1 - \sum p_k^2$) and Entropy ($-\sum p_k \log_2 p_k$).
  - Regression: Mean Squared Error / Variance reduction ($\sigma^2$).
- **Vectorized Cumulative Split Search:**
  - Sort each feature vector once.
  - Compute left/right class counts and variance in $O(N)$ via `np.cumsum`.
  - Accelerated split search from ~60s to sub-second per tree node.
- **Hyperparameter Controls & Validation:**
  - Controlled by `max_depth`, `min_samples_split`, `min_samples_leaf`, and `min_impurity_decrease`.
  - Unit tests prove exact alignment with scikit-learn split candidate thresholds.
  - Weighted impurity gain tracking produces normalized `feature_importances_`.

---

## Slide 4: Soft-Margin SVM via Pegasos (NumPy)
- **Primal Soft-Margin Objective:**
  $$J(\mathbf{w}, b) = \frac{\lambda}{2} \|\mathbf{w}\|^2 + \frac{1}{m} \sum_{i=1}^m \max(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)), \quad y_i \in \{-1, +1\}$$
- **Stochastic Subgradient Update:**
  - Learning rate schedule: $\eta_t = \frac{1}{\lambda \cdot t}$.
  - Mini-batch subgradient step on active margin violations ($y_i (\mathbf{w}^\top \mathbf{x}_i + b) < 1$).
  - Optional projection step onto $L_2$ ball $\|\mathbf{w}\| \le 1/\sqrt{\lambda}$.
- **Non-Linear Extension — Random Fourier Features (RFF):**
  - Explicit randomized mapping: $\mathbf{x} \mapsto \sqrt{\frac{2}{D}} \cos(\mathbf{x}\mathbf{W} + \mathbf{b})$.
  - Approximates RBF Gaussian kernel $k(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$ while retaining scalable $O(D)$ linear Pegasos solver.
- **Margin & Support Vectors:**
  - Margin: $\frac{2}{\|\mathbf{w}\|}$. Support vectors: sample indices with $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \le 1 + \varepsilon$.

---

## Slide 5: Experimental Setup & Cross-Validation
- **Rigorous Data Splitting:**
  - Seeded stratified train/val/test ($70\% / 15\% / 15\%$) based on price deciles.
  - Test set kept strictly untouched until final evaluation.
- **5-Fold Cross-Validation on Train+Val:**
  - Stratified folds covering all $54,320$ development listings.
  - Zero leakage: standardizer statistics ($\mu, \sigma$) and tier threshold ($y_{\text{median}}$) recomputed strictly within each fold's training slice.
- **Matched Baselines:**
  - Task A: `DecisionTreeRegressor` (scikit-learn).
  - Task B: `DecisionTreeClassifier`, `LinearSVC`, `SGDClassifier(loss='hinge')`, `SVC(kernel='rbf')`.

---

## Slide 6: Benchmark Results
- **Task A (Regression: Price in AZN):**
  - Our Decision Tree closely mirrors scikit-learn:
    - RMSE(log) $\approx 0.334$, $R^2(\text{log}) \approx 0.771$.
    - AZN MAE $\approx 55,000\text{ AZN}$, capturing median pricing trends effectively.
- **Task B (Classification: Premium vs Standard Tier):**
  - Our Decision Tree: F1-Score $\approx 0.887$, Accuracy $\approx 88.8\%$, ROC-AUC $\approx 0.945$.
  - Our Pegasos SVM: F1-Score $\approx 0.882$, Accuracy $\approx 88.2\%$, ROC-AUC $\approx 0.948$.
  - LinearSVC Baseline: F1-Score $\approx 0.882$, Accuracy $\approx 88.2\%$.
- **Validation:**
  - Both from-scratch models match or exceed scikit-learn baselines within $\pm 0.005$ tolerance.

---

## Slide 7: Model Diagnostics & Error Analysis
- **Where Do Models Fail?**
  - **Threshold Boundary:** Most tier classification errors concentrate in the $180,000 - 240,000\text{ AZN}$ band immediately adjacent to the median threshold ($210,000\text{ AZN}$).
  - **Luxury Market ($>1,000,000\text{ AZN}$):** High percentage errors due to unique architectural finishes, sea-view premiums, and designer furniture not captured in standard tabular fields.
  - **Sparse Districts:** Peripheral suburban settlements with low listing frequencies suffer higher variance.
- **Bias-Variance Dynamics:**
  - Decision trees overfit rapidly past depth $12$ (train accuracy approaches $98\%$ while validation accuracy plateaus at $89\%$).
  - Pegasos SVM exhibits strong regularization stability with $\lambda = 10^{-3}$, preventing overfitting.

---

## Slide 8: Team Contributions & Summary
- **Member 1 (Eldar Samedov):**
  - Data ingestion, cleaning logic, leakage firewall, evaluation framework, CV module, error analysis charts.
- **Member 2 (Narmina Rzayeva):**
  - NumPy Decision Tree implementation, cumsum split acceleration, depth tuning, overfitting ablation curves.
- **Member 3 (Mirkenan Qedirli):**
  - Pegasos SVM solver, RFF kernel mapper, objective tracking, support vector identification, lambda sweep.
- **Tools Used:**
  - Python 3.11, NumPy, Pandas, Matplotlib, Scikit-learn (baseline verification only), Pytest.
  - AI tools used strictly for boilerplate declarations, mathematical subgradient verification, and debugging.
- **Future Work:**
  - Ensemble methods (Random Forest, Gradient Boosting), text embeddings for free-form listing descriptions, and interactive web deployment (`app.py`).
