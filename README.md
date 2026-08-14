# Fraud Risk Detection

An end-to-end fraud-ranking case study on highly imbalanced credit-card transactions. The project compares supervised and anomaly-detection models, selects an operating policy using a chronological validation period, and evaluates the chosen model once on a later holdout period.

## Business objective

A fraud team cannot investigate every transaction. The useful question is therefore not simply whether a transaction can be classified, but how much fraud can be found within a limited review queue.

This project treats the model as an **alert-prioritization system for human review**. Average precision measures overall ranking quality, while precision and fraud recall at a fixed review capacity connect the model to investigator workload.

## Headline results

The final class-weighted logistic-regression model was trained on the first 80% of transactions and evaluated on the chronologically latest 20%.

| Metric | Validation | Final test |
|---|---:|---:|
| Average precision | 0.7727 | 0.7622 |
| Precision at 100 reviews | 45.00% | 59.00% |
| Fraud recall at 100 reviews | 78.95% | 78.67% |
| Test ROC-AUC | — | 0.9863 |

On the 56,962-transaction test period, reviewing the 100 highest-risk transactions found **59 of 75 fraud cases**, with 41 legitimate transactions sent for review.

Although class-weighted XGBoost achieved the highest validation average precision (0.7810), paired bootstrap intervals did not establish a decisive advantage over logistic regression. Logistic regression was selected for its comparable fixed-capacity recall, interpretability, and lower implementation complexity.

## Evaluation design

- Transactions are sorted by `Time` and split chronologically: 60% training, 20% validation, and 20% test.
- Model and review-capacity decisions use only training and validation data.
- The final model is refitted on the combined development period and evaluated once on the test period.
- Scaling is fitted inside scikit-learn pipelines to avoid preprocessing leakage.
- Exact duplicate rows are retained because anonymization prevents distinguishing repeated records from distinct transactions; no exact rows overlap the chronological evaluation boundaries.
- Accuracy is not a primary metric because only approximately 0.17% of transactions are fraudulent.
- A naive all-legitimate classifier and an unsupervised Isolation Forest provide reference points.

## Repository structure

```text
notebooks/
├── 01_data_validation_eda.ipynb
├── 02_baselines.ipynb
├── 03_model_comparison.ipynb
├── 04_threshold_and_error_analysis.ipynb
├── 05_uncertainty_and_model_selection.ipynb
└── 06_final_test_evaluation.ipynb
requirements.txt
```

The notebooks are intended to be read in numerical order. Executed outputs are committed so the complete analysis can be reviewed directly on GitHub.

## Data

The analysis uses the public [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) released by the Machine Learning Group at Université Libre de Bruxelles. It contains 284,807 European card transactions collected over approximately two days, including 492 confirmed fraud cases. Features `V1`–`V28` are anonymized PCA components; `Time`, `Amount`, and the binary target `Class` are also provided.

Download `creditcard.csv` from Kaggle and place it at:

```text
data/raw/creditcard.csv
```

The dataset is intentionally excluded from Git. Users are responsible for complying with the dataset's source terms.

## Reproduce the analysis

Python 3.11–3.13 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

After downloading the data, open the notebooks from the repository root and run them in order. Their path discovery also supports launching Jupyter from the `notebooks/` directory.

## Limitations

- The two-day observation period cannot demonstrate long-term concept drift or changing fraud strategies.
- Anonymous PCA features limit business interpretation and domain-driven feature engineering.
- Customer, merchant, device, location, and transaction-network identifiers are unavailable.
- Confirmed labels may omit undetected fraud.
- A capacity of 100 reviews is illustrative rather than derived from staffing or fraud-loss estimates.
- Bootstrap intervals condition on fitted models and do not capture every source of training uncertainty.
- Production use would require monitoring, retraining, probability calibration, fairness review, and controls governing investigator actions.

## Intended use

This is an educational portfolio project, not a production fraud-decision system. Model scores prioritize transactions for human review and should not be used as the sole basis for declining transactions or taking action against customers.
