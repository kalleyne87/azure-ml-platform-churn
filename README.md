# Azure ML Platform - Churn Prediction

## Overview
End-to-end machine learning pipeline that transforms raw customer data into churn predictions and risk segments.

## Tech Stack
Python, Pandas, scikit-learn, Parquet, Azure-ready architecture, GitHub

## Pipeline Stages
Bronze -> Silver -> Gold -> Train -> Score -> Reports

## How to Run

```bash
git clone https://github.com/kalleyne87/azure-ml-platform-churn.git
cd azure-ml-platform-churn

pip install -r requirements.txt

python train/03_train_churn_model.py
python score/04_score_churn.py
python score/05_churn_report.py
python score/06_feature_importance_report.py

## Key Outputs
- Churn probability by customer
- Risk bands
- Revenue at risk estimates
- Feature importance report

## Business Value
Helps teams identify at-risk customers before revenue is lost.

## Screenshots
<img width="1866" height="985" alt="Screenshot 2026-04-26 at 12 04 01 PM" src="https://github.com/user-attachments/assets/89a25e83-10b8-40a0-afbb-dd3079ae3e58" />
<img width="1569" height="822" alt="Screenshot 2026-04-26 at 12 00 17 PM" src="https://github.com/user-attachments/assets/c54eb5dd-9243-4847-860b-bdb764f7a3b3" />

## Future Improvements
- Deploy API
- Automated retraining
- Monitoring
