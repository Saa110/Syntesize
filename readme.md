# 🏦 Home Credit Loan Default Risk Prediction
## 24-Hour Machine Learning Hackathon
## 📌 Problem Overview

Financial institutions must accurately assess the risk of loan default before approving credit applications. Your task is to build a machine learning model that predicts whether a loan applicant will default.

You are given structured financial and demographic data for applicants. Using this data, you must predict the probability of default.

---

## 🎯 Objective

Build a model that predicts:

```text
TARGET = 1 → Client will default
TARGET = 0 → Client will repay
```

Your final submission must contain **probability predictions** (not just 0/1 labels).

---

## 📂 Dataset Files

You are provided with the following files:

### 1️⃣ `train.csv`

Contains:

* 120+ feature columns
* Target column: `TARGET`

Use this file to train your model.

---

### 2️⃣ `test.csv`

Contains:

* Same feature columns
* ❌ No `TARGET` column

Use this file to generate predictions.

---

### 3️⃣ `sample_submission.csv`

Format example:

```
SK_ID_CURR,TARGET
100001,0.52
100002,0.13
100003,0.87
```

* `SK_ID_CURR` → Unique applicant ID
* `TARGET` → Predicted probability of default (between 0 and 1)

---

## 📊 Evaluation Metric

### 🏆 Primary Metric: ROC-AUC Score

Submissions will be evaluated using:

```
ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
```

⚠️ Important:

* Submit probabilities (e.g., 0.72), not binary labels.
* Higher ROC-AUC = better model.

---

## 📈 What Is a Good Score?

| ROC-AUC | Performance Level |
| ------- | ----------------- |
| 0.60    | Basic             |
| 0.70    | Good              |
| 0.75    | Strong            |
| 0.80+   | Excellent         |

---

## 🧠 Dataset Characteristics

* ~300,000 records
* 120+ features
* Highly imbalanced (~8% defaults)
* Contains:

  * Numerical features
  * Categorical features
  * Missing values
  * High-cardinality columns

This is a real-world financial dataset.

---

## 🛠 Recommended Workflow

Participants are encouraged to:

1. Perform exploratory data analysis (EDA)
2. Handle missing values carefully
3. Encode categorical variables
4. Address class imbalance
5. Try multiple models
6. Tune hyperparameters
7. Compare performance using cross-validation

---

## 🤖 Suggested Models

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* CatBoost
* Gradient Boosting Models

Neural networks are optional but not required.

---

## 📤 Submission Guidelines

Your submission file must:

* Be named: `submission.csv`
* Contain exactly two columns:

  * `SK_ID_CURR`
  * `TARGET`
* Include predictions for all rows in `test.csv`
* Contain no missing values
* Have probabilities between 0 and 1

Invalid format submissions may be rejected.

---

## 🚫 Rules

* External datasets are not allowed
* Pre-trained models trained on this dataset externally are not allowed
* Maximum submissions per team: (Organizer to define)
* Internet usage: (Organizer to define)

---

## 🏁 Competition Structure

* Public leaderboard score is provisional
* Final ranking may be based on a hidden test set
* In case of tie, presentation score may be considered

---

## 💡 Bonus Points For

* Feature engineering innovation
* Model interpretability (SHAP, feature importance)
* Business insights
* Clear documentation

---

## ⚠️ Important Notes

* Dataset is imbalanced → Accuracy is misleading.
* Focus on ROC-AUC.
* Do not overfit to leaderboard.
* Use proper cross-validation.

---

## 🏆 Final Deliverables

Each team must submit:

1. `submission.csv`
2. Source code (Jupyter Notebook or .py file)
3. Brief explanation of approach (optional if required)

---

Best of luck.
Build responsibly.
Think like a data scientist. 🚀

---

## 🧮 Implemented Modeling Pipeline

- **Data loading**: `src/data/load_data.py` loads `train.csv` and `test.csv` with consistent schemas.
- **Preprocessing**: `src/data/preprocess.py` fixes `DAYS_EMPLOYED` anomalies and prepares categorical encodings.
- **Feature engineering**: `src/data/features.py` builds domain-informed ratios (e.g., `CREDIT_TO_INCOME`, `PAYMENT_TO_INCOME`, `EXT_SOURCE_MEAN`).
- **Cross-validation**: `src/validation/cv.py` sets up 5-fold stratified CV; `src/validation/metrics.py` computes ROC-AUC and summaries.
- **Base models**: `src/models/lgbm.py`, `src/models/xgb.py`, `src/models/catboost.py`, and `src/models/nn.py` train LightGBM, XGBoost, CatBoost, and an MLP with OOF and test predictions, configured via YAML files in `config/`.
- **Stacking**: `src/models/stacking.py` trains a Bayesian Ridge meta-model on OOF predictions and applies it to test predictions.
- **Training entrypoint**: `src/train.py` orchestrates feature building, base model training, and stacking, saving artifacts under `artifacts/`.
- **Inference & submission**: `src/inference.py` reads per-model test predictions plus the stacking model and writes a final `submission.csv` with `SK_ID_CURR` and probability `TARGET`.

To reproduce the full pipeline:

1. Install dependencies from `requirements.txt`.
2. Run `python -m src.train -all models` to train base models and the stacking meta-model.
3. Run `python -m src.inference` to generate `submission.csv` for upload.
