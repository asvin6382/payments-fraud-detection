# Payments fraud detection (IEEE-CIS)

Scores card payments for fraud risk, so a fraud team only checks the riskiest ones. Trained on the Kaggle
IEEE-CIS Fraud Detection data (590k card payments, 3.5% fraud) and served with FastAPI.

## Notebooks
| Notebook | What it does |
|---|---|
| `notebooks/01_fraud_detection_model.ipynb` | missing values, outliers, statistical tests, feature engineering, 5 models, thresholds, final test, saves the model |
| `notebooks/02_fraud_fastapi_serving.ipynb` | starts the API and scores 2,000 real test payments |

## Results (test month: 92,427 payments, 3,213 fraud)
| | Flagged | Real fraud | Precision | Share of all fraud found |
|---|---|---|---|---|
| Hand-written velocity rules | 566 | 59 | 10% | 1.8% |
| XGBoost, riskiest 1% | 924 | 800 | 87% | 25% |

- Models compared on the validation month: XGBoost (PR-AUC 0.535), HistGradientBoosting (0.514), random forest (0.476), decision tree (0.288), logistic regression (0.264).
- Blocking the riskiest 0.5% was 91% correct.
- Only 14% of the caught fraud came from accounts already labelled fraud earlier. On new customers precision was 85%.
- My time-window features helped on basic columns, but added nothing on top of Vesta's own columns.
- The customer key (card1 + addr1 + day - D1) is the idea shared by the winning Kaggle team.

## API
The API returns APPROVE, STEP_UP (OTP) or BLOCK with the top reasons. On 1,999 test payments it gave the same score as the notebook, at about 32 ms per request.

## Run it
```bash
pip install -r requirements.txt
uvicorn api.fraud_api:app --port 8030
```
Docs page: http://127.0.0.1:8030/docs

The data is not included (Kaggle's rules). Download it from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data), change `DATA_DIR` in notebook 01 and run it first: it creates the payment history file the API needs (set its path with `FRM_HISTORY`).

## Limits
- The payment history is kept in memory, so this is a demo service.
- The rule thresholds were set by hand, so the rules are a simple baseline.
