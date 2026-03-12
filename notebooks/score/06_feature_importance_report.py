from pathlib import Path
import json
import joblib
import pandas as pd 

def main():
    project_root = Path(__file__).resolve().parents[2]

    gold_path = project_root / "data" / "gold" / "gold_churn.parquet"
    registry_path = project_root / "models" / "registry.json"
    report_path = project_root / "data" / "outputs" / "feature_importance_report.csv"

    # Load registry
    with open(registry_path, "r") as f:
        registry = json.load(f)

    latest_model_path = Path(registry["models"][-1]["model_path"])

    if latest_model_path.is_absolute():
        model_path = latest_model_path
    else:
        model_path = (project_root / latest_model_path).resolve()

    # Load model
    model = joblib.load(model_path)

    # Load gold data so we can recover feature names
    gold_df = pd.read_parquet(gold_path)

    # Match the same columns used during training
    X = gold_df.drop(columns=["customer_id", "churn_flag"], errors="ignore")

    # Logistic regression coefficients
    coefficients = model.coef_[0]

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": coefficients
    })

    importance_df["impact"] = importance_df["coefficient"].apply(
        lambda x: "increases_churn" if x > 0 else "decreases_churn"
    )

    importance_df["abs_coefficient"] = importance_df["coefficient"].abs()

    # Sort by strongest overall effect
    importance_df = importance_df.sort_values(
        by="abs_coefficient",
        ascending=False
    )

    # Save full report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(report_path, index=False)

    print(f"Feature importance report saved to: {report_path}")
    print("\nTop 10 churn drivers:")
    print(
        importance_df[importance_df["coefficient"] > 0][
            ["feature", "coefficient"]
        ].sort_values(by="coefficient", ascending=False).head(10)
    )

    print("\nTop 10 retention drivers:")
    print(
        importance_df[importance_df["coefficient"] < 0][
            ["feature", "coefficient"]
        ].sort_values(by="coefficient", ascending=True).head(10)
    )

if __name__ == "__main__":
    main()