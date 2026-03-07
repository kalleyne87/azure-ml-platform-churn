import io
from datetime import datetime
import pandas as pd

from azure.identity import AzureCliCredential
from azure.storage.filedatalake import DataLakeServiceClient

ACCOUNT_NAME = "mlplatformtestdev33"
FILE_SYSTEM_NAME = "datalake"

SILVER_DIR = "silver/telco_churn"
GOLD_DIR = "gold/telco_churn"

SILVER_FILE = f"{SILVER_DIR}/churn_clean_20260307004520.parquet"
GOLD_FILE = f"{GOLD_DIR}/churn_features_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.parquet"

def main():
    # Connect to Data Lake
    credentail = AzureCliCredential()
    account_url = f"https://{ACCOUNT_NAME}.dfs.core.windows.net"
    service_client = DataLakeServiceClient(account_url=account_url, credential=credentail)
    file_system_client = service_client.get_file_system_client(FILE_SYSTEM_NAME)

    # Read silver parquet file to bytes
    silver_file_client = file_system_client.get_file_client(SILVER_FILE)
    silver_data = silver_file_client.download_file().readall()

    df = pd.read_parquet(io.BytesIO(silver_data))

    print("Loaded silver data")
    print(f"Rows: {len(df):,} Columns: {len(df.columns)}")

    # Identify ID columns
    id_cols = ["customer_id"] if "customer_id" in df.columns else []
    
    # Define target column
    target_col = "churn_flag"

    # Identify any additional columns to drop (e.g. original churn column)
    columns_to_drop = []
    if "churn" in df.columns:
        columns_to_drop.append("churn")

    # Count of services subscribed
    service_columns = [
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies"
    ]
    df["num_services"] = (df[service_columns] == "Yes").sum(axis=1)
    
    # Select feature columns by excluding ID, target, and any columns to drop
    feature_cols = df.drop(columns=id_cols + [target_col] + columns_to_drop, errors="ignore")
    
    # Identify numeric and categorical columns
    numeric_cols = feature_cols.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_cols.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    # One-hot encode categorical features
    feature_encoded = pd.get_dummies(
        feature_cols,
        columns=categorical_cols,
        drop_first=False,
    )


    gold_df = pd.concat(
        [
            df[id_cols].reset_index(drop=True) if id_cols else pd.DataFrame(),
            df[[target_col]].reset_index(drop=True),
            feature_encoded.reset_index(drop=True),
        ],
        axis=1
    )

    print("Successfully engineered features for gold layer.")
    print(f"Rows: {len(gold_df):,} Columns: {len(gold_df.columns)}")

    # Create gold directory if it doesn't exist
    out = io.BytesIO()
    gold_df.to_parquet(out, index=False)
    out.seek(0)

    try:
        file_system_client.get_directory_client(GOLD_DIR).create_directory()
        print(f"Created gold directory: {GOLD_DIR}")
    except Exception:
        pass # Directory likely already exists

    gold_df.to_parquet("gold_churn.parquet", index=False)
    gold_file_client = file_system_client.get_file_client(GOLD_FILE)
    gold_file_client.upload_data(out, overwrite=True)

    print(f"Wrote gold parquet: {GOLD_FILE}")

if __name__ == "__main__":
    main()


