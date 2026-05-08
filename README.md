# Protein_generation_lakehouse

RL + Diffusion + Flow Matching + MLflow on a Lakehouse Architecture


Here,  I built an end-to-end ML pipeline where protein sequences are generated using RL/diffusion-inspired models, stored in a lakehouse architecture (bronze/silver/gold), scored and iteratively improved through a training loop, with experiment tracking using MLflow.


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
Tested on:
- Python 3.10
- Java 17
- PySpark 3.x

pyspark
delta-spark
mlflow
numpy
pandas
```


# 📈 MLflow Tracks
 Parameters:
 - iterations
 - population size
 Metrics:
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



# Big picture
This is a data+ ML system 
```bash
Generate data → Store → Process → Score → Select → Train → Track
```


# data

1. We generate protein sequence data using 
```python
pipeline/01_generate.py
```
 
The generated synthetic data:
```python
e.g., generate_sequence() → "ACDEFGHIK..."
```

We saved this data in (data/bronze/sequences) which is the bronze layer.

- Raw data
- No processing
- Just generated sequences

2. We process data with
```python
pipeline/02_score.py
```
 
- read raw data
- Apply score function
```python
e.g., score_sequence("ACDEFGHIK...") -> score = 0.63
```

We saved this data in (data/silver/sequences) which is the silver layer.

- Cleaned + processed data
- Now has features (scores)

3. We filter and select best data with
```python
pipeline/03_select.py
```
 
- Sort sequences by score
- Keep top ones


We store this data in (data/gold/top_sequences) which is the gold layer.

- High-quality data
- Ready for modeling / decision-making

4. ML model

```python
models/
```
we have :

- rl_generator.py → Reinforcement Learning (mutation/improvement)
- diffusion_generator.py → noise + denoise
- flow_matching.py → sequence interpolation

They generate new protein sequences.

5. ML model training

```python
pipeline/04_train_loop.py
```

Loop:

```plaintext
1. Generate sequences
2. Score them
3. Select best
4. Mutate (RL step)
5. Repeat
```

- RL = improves sequences over time
- Not traditional “fit(X, y)”
- More like optimization loop

6. experiment tracking

by using MLflow:
```python
tracking/mlflow_utils.py
```

what is tracked;

- Parameters (iterations, population)
- Metrics:
     - avg score
     - max score

We can see model improving over time.


7. Data engineering

by using 
```plaintext
PySpark
Delta Lake
```

What Spark does:
- Handles large datasets
- Reads/writes data
- Scales processing


Example:
```python
df.write.mode("overwrite").parquet(...)
```

8. Full workflow:

```plaintext
        ┌──────────────┐
        │ ML Models    │  (RL / Diffusion / Flow)
        └──────┬───────┘
               ↓
      01_generate.py
               ↓
        Bronze Layer
   data/bronze/sequences
               ↓
      02_score.py
               ↓
        Silver Layer
   data/silver/sequences
               ↓
      03_select.py
               ↓
         Gold Layer
   data/gold/top_sequences
               ↓
      04_train_loop.py
               ↓
          MLflow Tracking

```

The full ML system does:

- data generation
- pipelines
- storage layers
- iterative training
- tracking

# How to run

After cloning the  project and create environment:

```bash
git clone protein-generation-lakehouse
cd protein-generation-lakehouse
pip install -r requirements.txt
```

There was incompatibility error between JDK and pyspark. We created an env with python 3.10 and also downloaded JDK 17 from [Eclipse Adoptium](https://adoptium.net/temurin/releases/?version=17).
To check Java version:
```bash
java -version
```

After issues with pyspark, hadoop and Java version on windows (You can check full steps on [issue 1-3](https://github.com/SalmaKazemiRashed/Protein_generation_lakehouse/issues?q=is%3Aissue%20state%3Aclosed))
I had to move to WSL or linux.
We are following:
```bash
Generator Model
    ↓
Raw sequences
    ↓
Spark DataFrame
    ↓
Parquet / Delta Lake
    ↓
Feature engineering
    ↓
Training / evaluation

```

We have Sample Generated Proteins:

+----------+--------------------------------------------------+--------------------------------------------------+---------+---------------+-------------------+
|protein_id|sequence                                          |optimized_sequence                                |rl_reward|diffusion_score|flow_matching_score|
+----------+--------------------------------------------------+--------------------------------------------------+---------+---------------+-------------------+
|0         |ELGTFLEDDTQAWWIPNTQFHAIVWVVGYTRISFFQGYTKQWFKDWIETF|ELGTFLEDDTQAWWIPNTQFHAIVWVVGYTRISFFQGYTKQWFKDWIETH|0.82     |0.9            |0.662              |
|1         |TYQIAWGPRYCMFWQWINADNYQWPNNMGMHVRIQNGELKMSLAHKTCDQ|TYQIAWGPRYCMFWQWINADNYQWPNNMGMHVRIQNGWLKMSLAHKTCDQ|0.849    |0.615          |0.784              |
|2         |LTGVRRVAGMPLFFITVWYPLCDCGYVCKWHKIKFQACQNPPIYTMENNA|LTGVRRVAGMPLFFITVWYPLCDCGYVCPWHKIKFQACQNPPIYTMENNA|0.817    |0.866          |0.765              |
|3         |WMNADYTPTESLCNGVSVCCNFYIIAHSLIRRDCHSYFKEWTCLVWISSN|WMNADYTPTESLCNGVSVCCNFYIIAHSLIRRDCHSYFKEWTCLVWISSL|0.909    |0.77           |0.627              |
|4         |SLHTPPCMSMGKEQHWDPIMHLYIRNVTQCWDGPYNPNIMIESMMQTSFC|SLHTPPCMSMGKEQHWDPIMHLYIRNVTQCWDGPYNPNIMIESMMQTSFC|0.942    |0.91           |0.936              |
|5         |CWTRGRCAVPCISTRESHFIQQWEKTWPVHWTPQLIKDTLGSCCTKEFAN|CWTRGRCAVPDISTRESHFIQQWEKTWPVHWTPQLIKDTLGSCCTKEFAN|0.994    |0.85           |0.985              |
|6         |IINSDESDDTNCDCVFCPMTESGVQYTSTTTTNGLWTGGYGVIAILWELM|IINSDESDDTNCDCVFCPMTESGVQYTSTTTTNHLWTGGYGVIAILWELM|0.816    |0.987          |0.576              |
|7         |HAEVQITQRAACNDHNRNHVFEEPTIYDCQTEMAMVTFTREKTDAWYKKC|HAEVQITQRAACNVHNRNHVFEEPTIYDCQTEMAMVTFTREKTDAWYKKC|0.931    |0.672          |0.71               |
|8         |YHTDIYRGTAVYRRDKRFEQCYIGLAWKMQVSTTADEASHMNSFAWTSNP|YHTDIYRGTFVYRRDKRFEQCYIGLAWKMQVSTTADEASHMNSFAWTSNP|0.72     |0.657          |0.787              |
|9         |DMRHYWCCDMYCNKCFRRRGMDARFRQQTLWGQTKKAFGDKQHVTERLYQ|DMRHYWCCDMYCNKCFKRRGMDARFRQQTLWGQTKKAFGDKQHVTERLYQ|0.879    |0.693          |0.797              |
+----------+--------------------------------------------------+--------------------------------------------------+---------+---------------+-------------------+

start MLflow:
```bash
mlflow ui
```

and then Open: http://127.0.0.1:5000.

