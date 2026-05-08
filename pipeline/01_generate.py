# ============================================
# 01_generator.py
# ============================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import length
from datetime import datetime
import uuid

# ============================================
# IMPORT YOUR MODELS
# ============================================

from models.rl_generator import generate_rl_sequence
from models.diffusion_generator import generate_diffusion_sequence
from models.flow_matching_generator import generate_flow_sequence

# ============================================
# CONFIG
# ============================================

NUM_SAMPLES_PER_MODEL = 1000

OUTPUT_PATH = "./data/bronze/sequences"

# ============================================
# START SPARK
# ============================================

spark = (
    SparkSession.builder
    .appName("protein-gen")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ============================================
# GENERATION WRAPPERS
# ============================================

def generate_rl():
    sequence = generate_rl_sequence()

    return {
        "protein_id": str(uuid.uuid4()),
        "sequence": sequence,
        "generator_type": "reinforcement_learning",
        "created_at": datetime.utcnow().isoformat()
    }


def generate_diffusion():
    sequence = generate_diffusion_sequence()

    return {
        "protein_id": str(uuid.uuid4()),
        "sequence": sequence,
        "generator_type": "diffusion",
        "created_at": datetime.utcnow().isoformat()
    }


def generate_flow_matching():
    sequence = generate_flow_sequence()

    return {
        "protein_id": str(uuid.uuid4()),
        "sequence": sequence,
        "generator_type": "flow_matching",
        "created_at": datetime.utcnow().isoformat()
    }

# ============================================
# RUN GENERATION
# ============================================

generated_data = []

print("\nGenerating RL proteins...\n")

for _ in range(NUM_SAMPLES_PER_MODEL):
    generated_data.append(generate_rl())

print("\nGenerating Diffusion proteins...\n")

for _ in range(NUM_SAMPLES_PER_MODEL):
    generated_data.append(generate_diffusion())

print("\nGenerating Flow Matching proteins...\n")

for _ in range(NUM_SAMPLES_PER_MODEL):
    generated_data.append(generate_flow_matching())

# ============================================
# CREATE DATAFRAME
# ============================================

df = spark.createDataFrame(generated_data)

# Add sequence length
df = df.withColumn("sequence_length", length("sequence"))

# ============================================
# SHOW SAMPLE
# ============================================

print("\nSample Generated Proteins:\n")

df.show(10, truncate=False)

# ============================================
# WRITE TO BRONZE LAYER
# ============================================

(
    df.write
    .mode("overwrite")
    .partitionBy("generator_type")
    .parquet(OUTPUT_PATH)
)

print(f"\nSUCCESS: Saved generated proteins to:\n{OUTPUT_PATH}")

# ============================================
# OPTIONAL VALIDATION
# ============================================

loaded_df = spark.read.parquet(OUTPUT_PATH)

print("\nValidation Read:\n")

loaded_df.groupBy("generator_type").count().show()

# ============================================
# STOP SPARK
# ============================================

spark.stop()