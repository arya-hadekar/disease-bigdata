from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import re

# Start Spark
spark = SparkSession.builder.appName("DataCleaning").getOrCreate()

# Load dataset
df = spark.read.csv("data/dataset.csv", header=True, inferSchema=True)

print("Original Columns:", df.columns)

# 1. Clean column names

def clean_column(name):
    name = name.strip().lower()
    name = re.sub(r'[^a-z0-9]', '_', name)  # replace special chars
    name = re.sub(r'_+', '_', name)         # remove multiple _
    return name

new_columns = [clean_column(c) for c in df.columns]
df = df.toDF(*new_columns)

print("Cleaned Columns:", df.columns)


# 2. Remove duplicates

df = df.dropDuplicates()


# 3. Handle missing values

df = df.fillna(0)


# 4. Convert all symptom columns to numeric

target = "diseases"

for column in df.columns:
    if column != target:
        df = df.withColumn(column, col(column).cast("double"))


# 5. Save cleaned dataset

df.toPandas().to_csv("data/cleaned_dataset.csv", index=False)

print(" Data cleaning completed and saved")