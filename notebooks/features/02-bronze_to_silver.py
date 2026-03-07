import io
import re
import pandas as pd
from datetime import datetime

from azure.identity import AzureCliCredential
from azure.storage.filedatalake import DataLakeServiceClient

ACCOUNT_NAME = "mlplatformtestdev33"
FILE_SYSTEM_NAME = "datalake"
BRONZE_DIR = "bronze/telco_churn.csv"
SILVER_DIR = "silver/telco_churn"
SILVER_FILE = f"{SILVER_DIR}/churn_clean_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.parquet"

def main():
    # Connect to Data Lake
    account_url = f"https://{ACCOUNT_NAME}.dfs.core.windows.net"
    credential = AzureCliCredential()
    service_client = DataLakeServiceClient(account_url=account_url, credential=credential)

    # Get file system client
    file_system_client = service_client.get_file_system_client(FILE_SYSTEM_NAME)

    # Read the bronze CSV file into a DataFrame
    bronze_file = file_system_client.get_file_client(BRONZE_DIR)
    download_file = bronze_file.download_file()
    csv_bytes = download_file.readall()

    df = pd.read_csv(io.BytesIO(csv_bytes))

    # Normalize column names - convert camelCase to snake_case
    def normalize_column(col: str) -> str:
        col = col.strip()
        col = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', col)
        col = col.replace(" ", "_").replace("-", "_")
        return col.lower()

    df.columns = [normalize_column(c) for c in df.columns]

    # Normalize churn column
    if "churn" in df.columns:
        df["churn"] = df["churn"].astype(str).str.strip().str.lower()
        df["churn_flag"] = (df["churn"] == "yes").astype(int)

    # Clean total_charges column
    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        df["total_charges"] = df["total_charges"].fillna(0)

    # Clean senior_citizen column
    if "senior_citizen" in df.columns:
        df["senior_citizen"] = pd.to_numeric(df["senior_citizen"], errors="coerce").fillna(0).astype(int)
        # Drop any duplicates
        df = df.drop_duplicates()

    print("Successfully cleaned and transformed data.")

    # Write to parquet and upload to silver layer
    out = io.BytesIO()
    df.to_parquet(out, index=False)
    out.seek(0)

    file_system_client.get_directory_client(SILVER_DIR).create_directory()

    silver_file = file_system_client.get_file_client(SILVER_FILE)
    silver_file.upload_data(out, overwrite=True)

    print(f"Wrote silver parquet: {SILVER_FILE}")
    print(f"Rows: {len(df):,} Columns: {len(df.columns)}")

if __name__ == "__main__":
    main()