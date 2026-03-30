# AML Homework 2: GBDT vs MLP on Bank Marketing Dataset

## 1. Introduction

This report investigates two powerful modeling paradigms — Gradient Boosted Decision Trees (GBDT) and Multi-Layer Perceptrons (MLP) — applied to the Bank Marketing dataset from a Portuguese banking institution. The task is binary classification: predicting whether a client will subscribe to a term deposit based on demographic and campaign-related features.

The dataset contains 45,211 records with 16 input features spanning client demographics (age, job, marital status, education), financial indicators (balance, credit default, loans), and campaign attributes (contact type, month, number of contacts, outcome of previous campaigns). The target variable is heavily imbalanced, with only 11.7% positive cases (subscription = "yes").

This controlled experiment enables direct comparison of how ensemble tree methods and neural networks handle the same prediction problem, revealing fundamental differences in their learning mechanisms, preprocessing requirements, and bias-variance characteristics.

## 2. Methods

### 2.1 Data Preprocessing

**Missing Values:** The dataset contains no null values, but several categorical columns use "unknown" as a placeholder (job, education, contact, poutcome). We retained "unknown" as its own category, as it may carry predictive signal — clients who did not disclose information may behave differently from those who did.

**Feature Engineering:** Three derived features were created:
- `was_contacted_before`: Binary flag derived from `pdays` (1 if previously contacted, 0 otherwise). 82% of clients had no prior contact.
- `age_group`: Binned age into 5 groups (young, mid, senior, elder, retired).
- `balance_positive`: Binary flag for positive account balance.

**Duration Removal:** The `duration` feature (last contact duration in seconds) was dropped. While highly predictive, it is only known after a call ends and is unrealistic for deployment-time prediction.

**Encoding:** All categorical variables were one-hot encoded with `drop_first=True` to avoid multicollinearity, resulting in 47 input features.

**Data Split:** Stratified 70/15/15 split into train (31,647), validation (6,782), and test (6,782) sets. Target rates were preserved at ~11.7% across all splits.

**Feature Scaling:** StandardScaler was fit on the training set only and applied to validation and test sets for MLP. This prevents data leakage. Scaling is necessary for MLP because gradient-based optimization is sensitive to feature magnitudes — features on different scales create elongated loss surfaces that slow convergence. Tree-based models are invariant to monotonic transformations since they only use threshold-based splits.

### 2.2 GBDT (XGBoost)

XGBClassifier was trained with `scale_pos_weight` to handle class imbalance (ratio of negative to positive samples ≈ 7.5). Key configurations:

- **Baseline:** learning_rate=0.1, n_estimators=500, max_depth=5, subsample=0.8
- **Learning Rate Comparison:** Tested lr ∈ {0.01, 0.1, 0.3} with 500 trees each
- **Hyperparameter Tuning:** RandomizedSearchCV with 30 iterations, 3-fold CV, optimizing F1-score
- **Best Parameters:** learning_rate=0.05, n_estimators=200, max_depth=7, subsample=0.7, reg_lambda=2.0

Training used `eval_set` with train and validation splits to monitor convergence. The baseline model showed mild overfitting (best validation loss at round 490 of 500, train-val gap of ~0.09).

### 2.3 MLP (MLPClassifier)

sklearn's MLPClassifier was used with standardized features. Since MLPClassifier lacks a `class_weight` parameter, the minority class was oversampled (random oversampling with replacement) to create balanced training data (50/50 split, 55,890 samples).

- **Baseline:** (128,) hidden layer, relu activation, lr=0.001 — converged in 34 iterations on imbalanced data with poor minority recall (0.24)
- **Architecture Comparison:** Tested (64,), (128,), (128, 64), (256, 128, 64)
- **Learning Rate & Activation:** Tested relu vs tanh with lr ∈ {0.001, 0.01, 0.1}
- **Hyperparameter Tuning:** RandomizedSearchCV with 30 iterations, 3-fold CV
- **Best Parameters:** hidden_layer_sizes=(128, 64), activation=tanh, learning_rate_init=0.001, alpha=0.001

## 3. Results

### 3.1 GBDT Results

The baseline XGBoost achieved validation F1=0.45 for the positive class. The learning rate comparison revealed:

| Learning Rate | Val Accuracy | Val F1 | Best Val Loss | Best Round |
|:---:|:---:|:---:|:---:|:---:|
| 0.01 | 0.806 | 0.437 | 0.518 | 499 |
| 0.10 | 0.830 | 0.450 | 0.434 | 490 |
| 0.30 | 0.842 | 0.418 | 0.408 | 490 |

Lower learning rates (0.01) converge slowly and need more trees. Higher rates (0.3) achieve lower loss but sacrifice F1 — the model becomes more conservative on positive predictions, favoring accuracy over recall.

After tuning, the final GBDT achieved validation precision=0.37, recall=0.63, F1=0.47.

Feature importance analysis revealed the top predictors: previous campaign outcome (poutcome), contact method, month of contact (especially May), account balance, and number of previous contacts. These are interpretable and actionable for the bank's marketing team.

*(See figures: fig_xgb_loss_curve.png, fig_xgb_learning_rate.png, fig_xgb_feature_importance.png)*

### 3.2 MLP Results

The baseline MLP on imbalanced data converged in just 34 iterations with validation score plateauing at ~0.9 (accuracy) from the first iteration — the model learned to predict "No" for everything. After oversampling, convergence extended to 235 iterations with meaningful loss descent.

Architecture comparison showed an inverse relationship between depth and F1:

| Architecture | Val Accuracy | Val F1 | Iterations |
|:---:|:---:|:---:|:---:|
| (64,) | 0.774 | 0.360 | 142 |
| (128,) | 0.797 | 0.357 | 235 |
| (128, 64) | 0.833 | 0.359 | 176 |
| (256, 128, 64) | 0.851 | 0.339 | 147 |

Deeper networks increased accuracy but decreased F1 — extra capacity was used to predict "No" more confidently rather than learning the minority boundary.

The best activation/learning rate combination was relu with lr=0.1 (F1=0.41), though the tuned model selected tanh with lr=0.001 via CV.

*(See figures: fig_mlp_loss_balanced_vs_imbalanced.png, fig_mlp_architecture_comparison.png, fig_mlp_lr_activation.png)*

## 4. GBDT vs MLP Comparison

### 4.1 Side-by-Side Test Set Results

| Metric | GBDT (XGBoost) | MLP | Winner |
|:---|:---:|:---:|:---:|
| Accuracy | 0.823 | 0.833 | MLP |
| Precision | 0.350 | 0.301 | GBDT |
| Recall | 0.600 | 0.325 | GBDT |
| F1-score | 0.442 | 0.313 | GBDT |
| AUC-PR | 0.461 | 0.264 | GBDT |
| Training Time | 0.32s | 32.28s | GBDT |

GBDT outperforms MLP on every metric relevant to the minority class. MLP's higher accuracy is misleading — it simply predicts "No" more often, which is rewarded by raw accuracy in an imbalanced setting. The AUC-PR gap (0.46 vs 0.26) shows GBDT maintains better precision-recall trade-offs across all thresholds.

*(See figures: fig_comparison.png)*

### 4.2 Discussion

**When to prefer GBDT vs MLP:** GBDT excels on structured/tabular data with mixed feature types, class imbalance, and interpretability requirements. It requires minimal preprocessing and trains extremely fast. MLP may be preferred for very large datasets with complex non-linear interactions, or for high-dimensional continuous inputs (embeddings, images, sensor data). For this bank marketing dataset, GBDT is the clear winner.

**Interpretability:** GBDT provides built-in feature importance — we identified poutcome, contact method, and month as top predictors, which is directly actionable for campaign strategy. MLP is a black box; post-hoc methods like SHAP can approximate feature importance but add complexity.

**Categorical features and missing values:** XGBoost handles missing values natively and works well with encoded categoricals. MLP requires explicit one-hot encoding and imputation, making the pipeline more complex and error-prone.

**Hyperparameter sensitivity:** MLP showed large performance swings across learning rates (F1: 0.31–0.41), architectures, and activation functions, and critically required class balancing to learn the minority class at all. GBDT was robust — the baseline already achieved F1=0.45, with tuning providing marginal improvements.

## 5. Discussion

### Bias-Variance Reflection

GBDT reduces bias through sequential boosting (each tree corrects the previous one's errors) and controls variance through regularization parameters (max_depth, subsample, reg_lambda). This makes it naturally robust on smaller tabular datasets.

MLP reduces bias through flexible function approximation but is prone to high variance — especially with limited data and class imbalance. The oversampling strategy helped address the class imbalance but introduced variance through duplicated minority samples.

### Limitations

- **Duration feature removed:** Including it would dramatically improve both models but at the cost of realism.
- **Oversampling for MLP:** Random oversampling can lead to overfitting on duplicated minority samples. SMOTE or class-weighted loss functions could be explored.
- **Feature engineering:** Only basic derived features were created. More sophisticated features (interaction terms, campaign timing patterns) could improve performance.
- **MLP architecture:** scikit-learn's MLPClassifier is limited compared to PyTorch/TensorFlow. More advanced architectures (batch normalization, dropout, learning rate scheduling) could improve MLP performance.

## 6. AI Tool Disclosure

**AI tools used:** Claude (Anthropic) was used as a coding assistant throughout this assignment. Specifically:

- **Code generation:** Claude helped write boilerplate code for data loading, preprocessing, model training, hyperparameter tuning, and visualization. Each code cell was reviewed, understood, and executed by the student before proceeding.
- **Debugging:** Claude helped identify and fix issues (e.g., target encoding bug on cell re-runs, line-break error in predict_proba call).
- **Explanation:** Claude provided explanations of modeling decisions (e.g., why scaling matters for MLP, how class imbalance affects training dynamics, interpreting feature importance).

**Student contributions:**
- All modeling decisions (encoding strategy, feature engineering choices, whether to keep duration, oversampling approach) were made by the student through interactive discussion.
- Interpretation of results and analysis of model behavior was done collaboratively.
- The experimental design (which hyperparameters to explore, which visualizations to create) followed the assignment specification with student input on specifics.
