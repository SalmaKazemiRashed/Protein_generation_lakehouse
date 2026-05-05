df = spark.read.format("delta").load("data/silver/sequences")

top_df = df.orderBy("score", ascending=False).limit(100)

top_df.write.format("delta").mode("overwrite").save("data/gold/top_sequences")