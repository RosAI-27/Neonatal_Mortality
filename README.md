# Neonatal Mortality in Cameroon — EDS 2018

A data science project investigating factors associated with neonatal mortality in Cameroon using the 2018 Demographic and Health Survey (DHS/EDS) data. The project combines statistical analysis with machine learning and evaluates the effect of SMOTE on an imbalanced binary classification problem.

> **Research focus:** determinants of neonatal mortality in Cameroon using a machine-learning approach based on DHS 2018 data.

## Project overview

Neonatal mortality refers to the death of a live-born child during the first 28 days of life. Because neonatal deaths represent a small proportion of births in the analytical dataset, the target variable is highly imbalanced. This project therefore compares conventional machine-learning models with models trained using synthetic minority oversampling (SMOTE).

The workflow covers:

- data preparation and exploratory analysis;
- descriptive and bivariate analysis;
- stratified train/test splitting;
- machine-learning classification;
- comparison of models with and without SMOTE;
- evaluation using metrics suited to imbalanced classification;
- model interpretation;
- an interactive Streamlit application for exploring the results and making predictions with the saved model.

## Dataset

The analytical dataset is derived from the **Cameroon Demographic and Health Survey (EDS/DHS) 2018**, based on the Birth Recode data.

| Item | Description |
|---|---|
| Source | EDS/DHS Cameroon 2018 |
| Analytical observations | 33,988 births |
| Analytical variables | 15 |
| Target | `neonatal_mort` |
| Target distribution | 32,891 survivors / 1,097 neonatal deaths |
| Mortality proportion | 3.23% |
| Period represented | 2013–2018 |

The prepared CSV is included in the repository for reproducibility. The Streamlit application also allows the user to upload a compatible CSV through the sidebar.

> **Data note:** The repository contains a prepared analytical dataset rather than the original DHS `.SAV` file. The original survey remains subject to the DHS Program's data-access and usage conditions.

## Methodology

### 1. Data preparation

The analysis uses demographic, socioeconomic, maternal, child, and healthcare-related variables, including maternal age, parity, child's sex, maternal education, wealth, residence, region, perceived birth size, birth interval, and antenatal-care visits.

### 2. Train/test split

A stratified 80/20 split is used so that the minority-class proportion remains comparable between training and test sets.

### 3. Preprocessing

Numerical variables are imputed and standardized, while categorical variables are imputed and one-hot encoded. Preprocessing is performed within machine-learning pipelines to reduce the risk of data leakage.

### 4. Models

The project compares five classifiers:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

Each model is evaluated under two conditions:

1. without SMOTE (baseline);
2. with SMOTE applied to the training data only.

### 5. Evaluation

Performance is assessed using metrics that are informative for an imbalanced target, including:

- ROC-AUC;
- F1-score;
- recall (sensitivity);
- precision;
- balanced accuracy;
- average precision;
- confusion matrices and classification reports.

### 6. Data leakage control

SMOTE is applied **after the train/test split and only to the training data**. The test set remains untouched so that evaluation reflects performance on the original class distribution.

## Repository structure

```text
Neonatal_Mortality/
├── README.md
├── requirements.txt
├── neonatal_mortality.ipynb
├── streamlit_app.py
├── neonatal_mortality_data.csv
├── best_xgboost_smote_model.pkl
├── preprocessor.pkl
│
├── data/
│   └── README.md
├── models/
│   └── README.md
├── outputs/
│   └── README.md
└── docs/
    └── PROJECT_STRUCTURE.md
```

The `data/`, `models/`, and `outputs/` directories provide a clean structure for future iterations. The current dataset and model artifacts intentionally remain at the repository root because the existing notebook and Streamlit application use root-relative paths. This refactor therefore improves organization **without rewriting the analytical code or breaking the application**.

## Streamlit application

The repository includes an interactive Streamlit dashboard presenting eight sections:

1. **Overview** — key indicators and study context;
2. **Descriptive analysis** — distributions and mortality rates by selected variables;
3. **Bivariate tests** — statistical associations with neonatal mortality;
4. **Machine Learning — Without SMOTE**;
5. **Machine Learning — With SMOTE**;
6. **SMOTE comparison**;
7. **Results & conclusions**;
8. **Prediction tool** using the saved XGBoost model and preprocessing object.

The application can operate with demonstration data when no CSV is uploaded, while the prepared dataset can be supplied through the file uploader for analysis.

## Installation and usage

### Requirements

- Python 3.9 or later
- pip or conda

### 1. Clone the repository

```bash
git clone https://github.com/RosAI-27/Neonatal_Mortality.git
cd Neonatal_Mortality
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run streamlit_app.py
```

The application will normally be available at `http://localhost:8501`.

## Model artifacts

The prediction module relies on two serialized artifacts:

- `best_xgboost_smote_model.pkl` — trained XGBoost classifier;
- `preprocessor.pkl` — fitted preprocessing transformer.

These files are kept at the root of the repository in the current version so that the existing application can load them without code changes.

## Results at a glance

The analysis shows that class imbalance is a major consideration: neonatal deaths account for only **3.23%** of observations. In the recorded model comparison, SMOTE substantially changes sensitivity for some models but does not automatically improve every performance metric. This is why model selection should not rely on accuracy alone.

The notebook records the detailed experimental results, including the comparison of AUC, F1, recall, balanced accuracy, precision, and average precision across models.

## Academic context

**Project title:** *The Determinants of Neonatal Mortality in Cameroon: A Machine Learning Approach Using DHS 2018*

This work was developed as part of a Master's-level Data Science project at Saint Jean Institut University.

**Author:** BAPFUBUSA SIAPZE Rose Ange  
**Program:** Master 1 Data Science  
**Institution:** Saint Jean Institut University  
**Supervisor:** Pr. NGUEFACK  
**Academic year:** 2025–2026

The project was also presented at the **5th Cameroonian Statistical Days**, where it received **3rd place** recognition.

## References

- Cameroon Demographic and Health Survey (EDS/DHS), 2018.
- Mosley, W. H., & Chen, L. C. (1984). An analytical framework for the study of child survival in developing countries.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique.

## Disclaimer

This project is intended for academic and research purposes. A machine-learning prediction should not be interpreted as a clinical diagnosis or as a substitute for professional medical judgment.
