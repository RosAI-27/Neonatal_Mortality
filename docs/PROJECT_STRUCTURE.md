# Project Structure

The repository separates the application, data, model artifacts, analysis notebook, documentation, and generated outputs.

```text
Neonatal_Mortality/
├── README.md
├── requirements.txt
├── streamlit_app.py                # Streamlit entry point
├── data/
│   └── neonatal_mortality_data.csv # Prepared analytical dataset
├── models/
│   ├── best_xgboost_smote_model.pkl
│   └── preprocessor.pkl
├── notebooks/
│   └── neonatal_mortality.ipynb    # Main analysis notebook
├── docs/
│   └── PROJECT_STRUCTURE.md
└── outputs/
    └── README.md
```

## Why this structure?

- **Root:** application entry point and dependency definition.
- **data/:** datasets used by the project.
- **models/:** serialized models and preprocessing artifacts.
- **notebooks/:** exploratory and experimental analysis.
- **outputs/:** generated figures, tables, and result files.
- **docs/:** project documentation.

## Deployment

The Streamlit entry point remains at the repository root:

```bash
streamlit run streamlit_app.py
```

The application loads model artifacts from `models/`.

## Refactor principle

The dataset and model artifacts were moved without modifying their contents. The notebook and Streamlit application received only the path changes required by the new directory structure.

The `main` branch is not modified by this refactor.
