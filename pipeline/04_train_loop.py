import os
import sys
import random
import mlflow

# -----------------------------
# Project imports
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.append(project_root)

from models.rl_generator import generate_sequence, mutate_sequence
from utils.scoring import score_sequence

# -----------------------------
# MLflow setup
# -----------------------------
db_path = os.path.abspath("mlflow.db")

mlflow.set_tracking_uri(
    f"sqlite:///{db_path}"
)

mlflow.set_experiment("protein_generation_rl")

# -----------------------------
# Configuration
# -----------------------------
ITERATIONS = 20
POPULATION_SIZE = 2000
TOP_K = 500
SEQUENCE_LENGTH = 100

# -----------------------------
# Start MLflow run
# -----------------------------
with mlflow.start_run():

    print("MLflow run started...")

    # -----------------------------
    # Log parameters
    # -----------------------------
    mlflow.log_param("iterations", ITERATIONS)
    mlflow.log_param("population_size", POPULATION_SIZE)
    mlflow.log_param("top_k", TOP_K)
    mlflow.log_param("sequence_length", SEQUENCE_LENGTH)

    # -----------------------------
    # Generate initial population
    # -----------------------------
    population = [
        generate_sequence(length=SEQUENCE_LENGTH)
        for _ in range(POPULATION_SIZE)
    ]

    # -----------------------------
    # Evolution / RL loop
    # -----------------------------
    for iteration in range(ITERATIONS):

        # -----------------------------
        # Score sequences
        # -----------------------------
        scores = [
            (seq, score_sequence(seq))
            for seq in population
        ]

        avg_score = sum(
            score for _, score in scores
        ) / len(scores)

        max_score = max(
            score for _, score in scores
        )

        # -----------------------------
        # Log metrics
        # -----------------------------
        mlflow.log_metric(
            "avg_score",
            avg_score,
            step=iteration
        )

        mlflow.log_metric(
            "max_score",
            max_score,
            step=iteration
        )

        print(
            f"Iteration {iteration + 1} | "
            f"Avg Score: {avg_score:.3f} | "
            f"Max Score: {max_score:.3f}"
        )

        # -----------------------------
        # Select top proteins
        # -----------------------------
        scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top_sequences = [
            seq for seq, _ in scores[:TOP_K]
        ]

        # -----------------------------
        # Create next generation
        # -----------------------------
        new_population = []

        while len(new_population) < POPULATION_SIZE:

            parent = random.choice(top_sequences)

            child = mutate_sequence(parent)

            new_population.append(child)

        population = new_population

    # -----------------------------
    # Final evaluation
    # -----------------------------
    final_scores = [
        score_sequence(seq)
        for seq in population
    ]

    final_best_score = max(final_scores)
    final_avg_score = sum(final_scores) / len(final_scores)

    # -----------------------------
    # Log final metrics
    # -----------------------------
    mlflow.log_metric(
        "final_best_score",
        final_best_score
    )

    mlflow.log_metric(
        "final_avg_score",
        final_avg_score
    )

    print("\nTraining completed successfully.")

print("\nSUCCESS: RL training loop completed.")