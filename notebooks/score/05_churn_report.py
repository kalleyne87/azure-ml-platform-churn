from pathlib import Path
import pandas as pd

def main():
    project_root = Path(__file__).resolve().parents[2]

    scored_path = project_root / 'data' / 'outputs' / 'scored_churn.parquet'
    report_path = project_root / 'data' /'outputs' / 'churn_report.txt'

    scored_df = pd.read_parquet(scored_path)
    
    total_customers = len(scored_df)
    avg_churn_probability = scored_df['churn_probability'].mean()

    high_risk_count = (scored_df['risk_band'] == "high").sum()
    medium_risk_count = (scored_df['risk_band'] == "medium").sum()
    low_risk_count = (scored_df['risk_band'] == "low").sum()

    predicted_churn_count = (scored_df['churn_prediction'] == 1).sum()

    high_risk_revenue = 0.0
    if "monthly_charges" in scored_df.columns:
        high_risk_revenue = scored_df.loc[
            scored_df["risk_band"] == "high", "monthly_charges"
        ].sum()

    top_10 = scored_df.sort_values(by='churn_probability', ascending=False).head(10)

    report_lines = [
        "CHURN RISH REPORT",
        "=" * 50,
        f"Total customers scored: {total_customers}",
        f"Average churn probability: {avg_churn_probability:.2%}",
        f"Predicted churn customers: {predicted_churn_count}",
        f"Estimated monthly revenue at high risk: ${high_risk_revenue:,.2f}",
        "",
        "Risk Band Breakdown:",
        "-" * 50,
        f"High risk customers: {high_risk_count}",
        f"Medium risk customers: {medium_risk_count}",
        f"Low risk customers: {low_risk_count}",
        "",
        "Top 10 Highest-Risk Customers",
        "-" * 50
    ]

    for _, row in top_10.iterrows():
        report_lines.append(
            f"{row['customer_id']} | "
            f"prob={row['churn_probability']:.2%} | "
            f"prediction={row['churn_probability']} | "
            f"risk={row['risk_band']}"
        )

    report_text = "\n".join(report_lines)

    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()  