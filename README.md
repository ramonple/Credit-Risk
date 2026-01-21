# Credit Risk (End-to-End ML Pipeline)

This repository demonstrates an end-to-end workflow for credit risk modelling in Python, covering the full model lifecycle:
data preparation → feature engineering → feature selection → modelling → evaluation → deployment-style scoring → monitoring.

## What’s inside
- **Feature cleaning & engineering** (pipeline-style scripts)
- **Model training** (baseline + ML models)
- **Model evaluation** (metrics, stability / performance checks)
- **Deployment-style scoring** (batch scoring / reusable scoring logic)
- **Monitoring** (post-deployment drift / performance tracking)
- **Rule search & evaluation** (decision-rule exploration)

## Project structure 
- `feature_prepare/` feature engineering
- `ML_rules/` ML Rule construction and evalution
- `models/` modelling, evaluation and monitoring

## Quickstart
```bash
pip install -r requirements.txt
python src/machine_learning_modelling.py
