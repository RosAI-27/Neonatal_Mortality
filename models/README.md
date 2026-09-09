# Models

This directory is reserved for future organization of trained model artifacts.

The current `.pkl` files remain at the repository root because `streamlit_app.py` currently loads them relative to the application file. This keeps the application unchanged and avoids breaking the deployed workflow.

## Current artifacts

- `best_xgboost_smote_model.pkl` — saved XGBoost classifier used by the prediction module.
- `preprocessor.pkl` — saved preprocessing transformer required before prediction.

Any future move of these artifacts should be accompanied by a deliberate update to the application's model-loading paths and a deployment test.