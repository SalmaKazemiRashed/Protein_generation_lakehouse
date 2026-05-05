from pyspark.sql.functions import udf
from utils.scoring import score_sequence

score_udf = udf(score_sequence)

df = spark.read.format("delta").load("data/bronze/sequences")
df = df.withColumn("score", score_udf("sequence"))

df.write.format("delta").mode("overwrite").save("data/silver/sequences")