from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType

import os
import sys

# -----------------------------
# Fix Python environment
# -----------------------------
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# -----------------------------
# Project imports
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.append(project_root)

from utils.scoring import score_sequence

# -----------------------------
# Start Spark
# -----------------------------
spark = (
    SparkSession.builder
    .appName("protein-scoring")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# -----------------------------
# Create scoring UDF
# -----------------------------
score_udf = udf(score_sequence, DoubleType())

# -----------------------------
# Read bronze layer
# -----------------------------
input_path = "data/bronze/protein_sequences"

df = spark.read.parquet(input_path)

# -----------------------------
# Score sequences
# -----------------------------
df_scored = df.withColumn(
    "score",
    score_udf(col("optimized_sequence"))
)

# -----------------------------
# Show sample
# -----------------------------
print("\nScored Proteins:\n")

df_scored.show(10, truncate=False)

# -----------------------------
# Write silver layer
# -----------------------------
output_path = "data/silver/protein_sequences"

df_scored.write.mode("overwrite").parquet(output_path)

print(f"\nSUCCESS: Silver layer written to -> {output_path}")

# -----------------------------
# Stop Spark
# -----------------------------
spark.stop()