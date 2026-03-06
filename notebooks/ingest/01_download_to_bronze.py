import pandas as pd
import requests

from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

# Download the dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Save locally
local_file = "telco_churn.csv"
df.to_csv(local_file, index=False)

print("Dataset downloaded and saved locally.")

# Connect to Data Lake
account_name = "mlplatformtestdev33"

account_url = f"https://{account_name}.dfs.core.windows.net"

credential = DefaultAzureCredential()

service_client = DataLakeServiceClient(account_url=account_url, credential=credential)

# Get file system client
file_system_client = service_client.get_file_system_client(file_system="datalake")

# Upload the file to the bronze layer
file_cleint = file_system_client.get_file_client("bronze/telco_churn.csv")

with open(local_file, "rb") as data:
    file_cleint.upload_data(data, overwrite=True)

print("Dataset uploaded to the bronze layer in Data Lake.")