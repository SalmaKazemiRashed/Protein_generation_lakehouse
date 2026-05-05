# Protein_generation_lakehouse
RL + Diffusion + Flow Matching + MLflow on a Lakehouse Architecture


# Overview
This project implements an end-to-end AI-driven protein sequence generation platform combining:

* Reinforcement Learning (sequence optimization)
* Diffusion-inspired generation (iterative denoising)
* Flow matching (continuous sequence interpolation)
* Lakehouse data architecture
* Experiment tracking with MLflow

# Tech stack
- Databricks
- Apache Spark / PySpark
- Delta Lake
- MLflow

# Project Goal

```plaintext
Generate → Score → Track → Select → Improve → Repeat
```

# workflow
```plaintext
AI Models → Bronze (raw) → Silver (scored) → Gold (top sequences)
                     ↓
                  MLflow (experiments, metrics, params)
```

The structure of repository would be like:
```plaintext

protein-generation-lakehouse/
│
├── data/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── models/
│   ├── rl_generator.py
│   ├── diffusion_generator.py
│   ├── flow_matching.py
│
├── pipeline/
│   ├── 01_generate.py
│   ├── 02_score.py
│   ├── 03_select.py
│   ├── 04_train_loop.py
│
├── tracking/
│   ├── mlflow_utils.py
│
├── utils/
│   ├── scoring.py
│
├── notebooks/
│   demo_notebook.ipynb
│
├── requirements.txt
└── README.md

```

# setup
```plaintext
pyspark
mlflow
numpy
pandas
```


# 📈 MLflow Tracks
* Parameters:
 - iterations
 - population size
* Metrics:
 - average score per iteration
 - max score per iteration

# 👉 Databricks:
* experiment dashboard
* performance curves

# Lakehouse layers
| Layer  | Purpose                 |
| ------ | ----------------------- |
| Bronze | Raw generated sequences |
| Silver | Scored sequences        |
| Gold   | Top candidates          |
