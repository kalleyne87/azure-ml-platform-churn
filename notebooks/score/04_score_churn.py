from pathlib import Path
import pandas as pd

import joblib
def assign_risk_band(probability: float) -> str:
    if probability >= 0.70:
        return 'high'
    elif probability >= 0.40:
        return 'medium'
    else:
        return 'low'
    
def main():
    project_root = Path(__file__).resolve().parents[2]

    gold_path = project_root / 'data' / 'gold' / 'gold_churn.parquet'
    model_path = project_root / 'models' / 'churn_logistic_regression.pkl'
    output_path = project_root / 'data' / 'outputs' / 'scored_churn.parquet'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the gold data
    gold_df = pd.read_parquet(gold_path)

    # Keep customer_id for output
    customer_ids = gold_df['customer_id'] if 'customer_id' in gold_df.columns else pd.Series(range(len(gold_df)))

    #Reomve columsn not used in scoring
    X_score = gold_df.drop(columns=['customer_id', 'churn_flag'], errors='ignore')

    # Load the trained model
    model = joblib.load(model_path)

    # Score churn probabilities
    churn_probabilities = model.predict_proba(X_score)[:, 1]

    # Convert to predictions
    threshold = 0.40
    predictions = (churn_probabilities >= threshold).astype(int)

    #Build scored dataframe
    scored_df = pd.DataFrame({
        "customer_id": customer_ids,
        "churn_probability": churn_probabilities,
        "churn_prediction": predictions,
    })

    scored_df['risk_band'] = scored_df['churn_probability'].apply(assign_risk_band)

    scored_df.to_parquet(output_path, index=False)

    print("Scoring completed.")
    print(f"Rows scored: {len(scored_df)}")
    print(f"Saved to {output_path}")
    print(scored_df.head(10))

if __name__ == "__main__":
    main()