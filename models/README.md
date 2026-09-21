# Models

This directory contains serialized model artifacts used by the Streamlit prediction module.

## Current artifacts

- `best_xgboost_smote_model.pkl` — saved XGBoost classifier.
- `preprocessor.pkl` — fitted preprocessing transformer required before prediction.

The application loads these files from `models/` relative to `streamlit_app.py`.

> The binary model files themselves were moved without changing their contents. Only their repository location changed.
