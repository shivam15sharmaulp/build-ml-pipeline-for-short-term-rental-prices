# NYC Short-Term Rental Price Pipeline

![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/mlflow-pipeline-0194E2?logo=mlflow&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/w%26b-experiment%20tracking-FFBE00)
![Status](https://img.shields.io/badge/status-portfolio%20project-1f7a5c)

An end-to-end machine learning pipeline for estimating nightly short-term rental prices in New York City. This repository is my reproducible training workflow for downloading raw listing data, cleaning it, validating it, training a baseline model, and tracking every dataset and model artifact in Weights & Biases.

This project is less about squeezing out the last decimal point of model performance and more about showing a clean, repeatable ML workflow: versioned data, reproducible training, lightweight quality gates, experiment tracking, and a deployable model artifact.

## Author

Maintained and extended by Shivam Sharma.

This fork builds on the original Udacity starter repository, with my own implementation changes, project configuration, experiment runs, and documentation updates.

## Overview

The project is built around a simple question: given a listing's metadata and title, what is a reasonable nightly price?

Instead of treating this as a one-off notebook exercise, I structured it as a repeatable ML pipeline with clear handoffs between steps:

- raw data ingestion
- basic cleaning and filtering
- data quality checks
- train/validation/test splitting
- model training and artifact export
- optional regression testing on the promoted production model

The current implementation uses MLflow to orchestrate the steps, Hydra to manage configuration, and Weights & Biases to version artifacts and track experiments.

## Results Snapshot

Latest recorded held-out test performance from the production-tagged model in this repo:

| Metric | Value |
| --- | ---: |
| MAE | 33.29 |
| $R^2$ | 0.581 |

Those scores come from the production-model regression test tracked in the repository's W&B run metadata.

## Visual Overview

W&B artifact graph for the pipeline:

![Pipeline graph](images/wandb-pipeline-graph.png)

Example of model selection and experiment review in W&B:

![W&B model selection](images/wandb_select_best.gif)

## Pipeline Stages

### 1. Data ingestion

The pipeline starts by pulling a CSV sample and logging it to W&B as a versioned raw dataset artifact.

### 2. Basic cleaning

The cleaning step applies the same rules every run:

- filters listings to a configurable price range
- converts `last_review` to a datetime field
- drops listings outside expected NYC longitude and latitude bounds

The cleaned output is saved as `clean_sample.csv` and logged back to W&B.

### 3. Data validation

Before training, the project runs a lightweight quality gate over the cleaned dataset. The checks currently cover:

- expected column names and order
- allowed neighbourhood groups
- geographic boundaries
- row-count sanity checks
- price range enforcement
- KL-divergence drift check against a tagged reference dataset

### 4. Train/validation/test split

The cleaned dataset is split into train/validation and test artifacts using stratification on `neighbourhood_group`.

### 5. Random forest training

The training step builds a scikit-learn pipeline that combines structured preprocessing with basic NLP on listing titles:

- ordinal encoding for `room_type`
- one-hot encoding for `neighbourhood_group`
- imputation for numerical and date-derived features
- TF-IDF features from the `name` column
- `RandomForestRegressor` as the baseline model

Each run logs at least these outputs to W&B:

- MAE
- $R^2$
- feature importance visualization
- exported MLflow model artifact

### 6. Optional production-model regression test

Once a trained model is promoted with the `prod` alias in W&B, the pipeline can run a separate regression test step against the latest held-out test set.

## Tech Stack

- Python 3.13
- MLflow
- Hydra
- Weights & Biases
- pandas
- scikit-learn
- matplotlib
- conda

## Repository Layout

```text
.
├── main.py                     # Pipeline entrypoint
├── config.yaml                 # Central pipeline configuration
├── rf_config.json              # Serialized RF settings used by training runs
├── components/                 # Reusable MLflow components
├── src/
│   ├── basic_cleaning/         # Cleaning step
│   ├── data_check/             # Data validation tests
│   ├── eda/                    # Notebook-based exploration
│   └── train_random_forest/    # Model training step
└── images/                     # README assets
```

## Environment Setup

This repository is set up for macOS and Ubuntu-based environments and expects Python 3.13.

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

Authenticate with Weights & Biases before running the pipeline:

```bash
wandb login
```

## Running The Pipeline

Run the full pipeline:

```bash
mlflow run .
```

Run only selected steps:

```bash
mlflow run . -P steps=download,basic_cleaning,data_check
```

Override configuration values with Hydra options:

```bash
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.max_tfidf_features=30 modeling.random_forest.max_features=0.33"
```

Launch a simple hyperparameter sweep:

```bash
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
```

Run the production-model regression test after tagging a model with `prod` in W&B:

```bash
mlflow run . -P steps=test_regression_model
```

## Configuration

The pipeline is driven from `config.yaml`. The main knobs I tune there are:

- dataset sample selection
- cleaning bounds for price
- data drift threshold
- train/validation/test split settings
- TF-IDF vocabulary size
- random forest hyperparameters

Current defaults include:

- `etl.sample: sample1.csv`
- `etl.min_price: 10`
- `etl.max_price: 350`
- `modeling.max_tfidf_features: 30`
- `modeling.random_forest.max_features: 0.33`

## Experiment Tracking And Artifacts

W&B is used as the experiment system of record for both data and models. Across a normal run, the pipeline produces and consumes artifacts such as:

- `sample.csv`
- `clean_sample.csv`
- `trainval_data.csv`
- `test_data.csv`
- `random_forest_export`

I also use W&B aliases to manage lifecycle transitions:

- `latest` for the newest artifact version
- `reference` for the baseline data snapshot used in drift checks
- `prod` for the model selected for downstream testing

## Practical Notes

- The cleaning step includes geographic boundary filtering so obviously invalid NYC listings are removed before validation and training.
- The model is intentionally a baseline, but the training pipeline is fully reproducible and easy to extend.
- The `eda` step is there for exploratory work, but the actual training workflow is driven from MLflow components rather than notebooks.

## Key Learnings

- Turning notebook logic into pipeline steps forces cleaner decisions around inputs, outputs, and configuration.
- Data validation matters as much as model training. Simple checks on geography, price bounds, and distribution drift catch issues early.
- Even a baseline random forest becomes much more useful when it is paired with artifact versioning, reproducible runs, and model promotion rules.
- Listing titles carry real signal. Adding TF-IDF features from the `name` field improved the project beyond a purely structured baseline.

## Future Improvements

- Compare the current random forest baseline against gradient boosting or XGBoost-style models.
- Add richer location features such as distance-to-center or neighborhood-level aggregates.
- Introduce a stricter model promotion gate based on validation and test thresholds instead of manual review alone.
- Package the inference pipeline behind a small API or batch scoring job so the trained model is easier to consume downstream.

## Links

- GitHub: https://github.com/shivam15sharmaulp/build-ml-pipeline-for-short-term-rental-prices
- Weights & Biases: https://wandb.ai/ssharma4-shivam-llc/nyc_airbnb/overview/details

## License

[LICENSE.txt](LICENSE.txt)
