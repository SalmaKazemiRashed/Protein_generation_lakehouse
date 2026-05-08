# What MLflow Tracks
# Parameters:
# iterations
#  population size
#  Metrics:
#  average score per iteration
#  max score per iteration

# 👉 In Databricks, this shows up as:

# experiment dashboard
# performance curves



import mlflow
from models.rl_generator import generate_sequence, mutate_sequence
from utils.scoring import score_sequence
from tracking.mlflow_utils import start_experiment, log_metrics, log_params

run = start_experiment()

log_params({
    "iterations": 5,
    "population_size": 1000
})

population = [generate_sequence() for _ in range(1000)]

for i in range(5):
    scores = [(seq, score_sequence(seq)) for seq in population]

    avg_score = sum(s for _, s in scores) / len(scores)
    max_score = max(s for _, s in scores)

    log_metrics(i, avg_score, max_score)

    # Select top 20%
    scores.sort(key=lambda x: x[1], reverse=True)
    top = [seq for seq, _ in scores[:200]]

    # Mutate for next generation
    population = [mutate_sequence(seq) for seq in top]

mlflow.end_run()