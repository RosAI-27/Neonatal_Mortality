# Project Structure

This repository contains the analysis workflow and the Streamlit application for the neonatal mortality study.

```text
Neonatal_Mortality/
├── README.md
├── requirements.txt
├── neonatal_mortality.ipynb        # Main analysis notebook
├── streamlit_app.py                # Streamlit application
├── neonatal_mortality_data.csv     # Prepared dataset used by the analysis
├── best_xgboost_smote_model.pkl    # Saved XGBoost model used by the app
├── preprocessor.pkl                # Saved preprocessing object used by the app
│
├── data/
│   └── README.md                   # Data documentation and handling notes
│
├── models/
│   └── README.md                   # Model artifact documentation
│
├── outputs/
│   └── README.md                   # Intended location for generated figures/results
│
└── docs/
    └── PROJECT_STRUCTURE.md        # Repository organization guide
```

## Why the main artifacts remain at the repository root

The notebook and Streamlit application currently use root-relative paths for the prepared dataset and saved model artifacts. Keeping those files in place on this refactor branch avoids changing the analytical code or introducing broken paths.

The `data/`, `models/`, and `outputs/` directories are therefore documentation-ready locations for future iterations. Once the application and notebook paths are deliberately migrated together, the binary/data artifacts can be moved without breaking reproducibility.

## Protected core files

The following files contain the project's main analytical/application logic and should not be casually rewritten during structural refactors:

- `neonatal_mortality.ipynb`
- `streamlit_app.py`
- `requirements.txt`
- `neonatal_mortality_data.csv`
- `best_xgboost_smote_model.pkl`
- `preprocessor.pkl`

Structural changes should preserve the current application entry point:

```bash
streamlit run streamlit_app.py
```
