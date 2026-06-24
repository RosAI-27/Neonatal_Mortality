# 🏥 Déterminants de la Mortalité Néonatale au Cameroun

Application Streamlit d'analyse des données EDS Cameroun 2018 avec comparaison SMOTE vs Sans SMOTE.

---

## 📋 Prérequis

- Python 3.9+
- pip ou conda

## 🚀 Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-username/neonatal-mortality-cameroon.git
cd neonatal-mortality-cameroon
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Avec venv
python -m venv venv

# Activation Linux/Mac
source venv/bin/activate

# Activation Windows
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

**requirements.txt :**
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.1.0
imbalanced-learn>=0.11.0
joblib>=1.3.0
pyreadstat>=1.2.0
```

### 4. Préparer les données

Placez le fichier de données dans le dossier `data/` :
- `neonatal_mortality_data.csv` (séparateur `;`)

Ou utilisez les données de démonstration intégrées.

### 5. Lancer l'application

```bash
streamlit run streamlit_app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

---

## 🌐 Déploiement en ligne

### Option A : Streamlit Community Cloud (Gratuit)

1. Poussez votre code sur GitHub (inclure `streamlit_app.py`, `requirements.txt`, et les modèles `.pkl` si disponibles)
2. Connectez-vous sur [share.streamlit.io](https://share.streamlit.io)
3. Cliquez sur **"New app"** → Sélectionnez votre dépôt
4. Spécifiez le fichier principal : `streamlit_app.py`
5. Cliquez **Deploy**

**Structure du dépôt GitHub recommandée :**
```
neonatal-mortality-cameroon/
├── streamlit_app.py          # Application principale
├── requirements.txt          # Dépendances
├── best_xgboost_smote_model.pkl   # Modèle (optionnel)
├── preprocessor.pkl               # Préprocesseur (optionnel)
├── data/
│   └── neonatal_mortality_data.csv
└── README.md
```

### Option B : Heroku

```bash
# Installer Heroku CLI
heroku login
heroku create neonatal-mortality-cm

# Créer Procfile
echo "web: streamlit run streamlit_app.py --server.port=$PORT" > Procfile

# Créer runtime.txt
echo "python-3.11.6" > runtime.txt

git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### Option C : Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
```

```bash
# Build et run
docker build -t neonatal-app .
docker run -p 8501:8501 neonatal-app
```

---

## 📁 Fichiers optionnels pour l'outil de prédiction

Pour activer l'outil de prédiction avancé (Module 8), placez ces fichiers dans le même dossier que `streamlit_app.py` :

- `best_xgboost_smote_model.pkl` — Modèle XGBoost entraîné avec SMOTE
- `preprocessor.pkl` — Pipeline de prétraitement (StandardScaler + OneHotEncoder)

**Génération des modèles (extrait du code Python) :**
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import joblib

# Prétraitement
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['maternal_age', 'maternal_age_sq', 'parity', 'region']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), 
     ['sex_child', 'education', 'wealth', 'residence', 'baby_size', 'birth_interval', 'anc_visits'])
])

# Pipeline SMOTE + XGBoost
pipeline = ImbPipeline([
    ('prep', preprocessor),
    ('smote', SMOTE(k_neighbors=5, random_state=42)),
    ('model', XGBClassifier(n_estimators=200, scale_pos_weight=1, max_depth=5, learning_rate=0.1))
])

# Entraînement
pipeline.fit(X_train, y_train)

# Sauvegarde
joblib.dump(pipeline.named_steps['model'], 'best_xgboost_smote_model.pkl')
joblib.dump(pipeline.named_steps['prep'], 'preprocessor.pkl')
```

---

## 🔧 Structure de l'application

| Module | Description |
|--------|-------------|
| 📊 Vue d'ensemble | KPIs, contexte, distribution des classes |
| 🔬 Analyse descriptive | Statistiques pondérées par sous-groupes |
| 📈 Tests bivariés | Chi-deux, t-tests, force d'association |
| 🤖 ML — Sans SMOTE | Baseline (5 algorithmes) |
| ⚖️ ML — Avec SMOTE | Sur-échantillonnage synthétique |
| 🔄 Comparaison SMOTE | Analyse détaillée des différences |
| 📋 Résultats & Conclusions | Odds Ratios, recommandations |
| 🩺 Outil de prédiction | Score de risque néonatal personnalisé |

---

## ⚠️ Notes importantes

- **Data Leakage** : SMOTE est appliqué UNIQUEMENT après le split Train/Test, jamais avant.
- **Déséquilibre** : La classe minoritaire représente 3.2% — justifiant l'approche SMOTE.
- **Données** : L'application fonctionne avec des données de démonstration si aucun CSV n'est uploadé.

---

## 📚 Références

- EDS Cameroun 2018 (DHS Program) — CMBR71FL.SAV
- Mosley & Chen (1984) — Cadre conceptuel de la survie infantile
- Chawla et al. (2002) — SMOTE

---

**Auteur** : BAPFUBUSA SIAPZE Rose Ange — Master 1 Data Science, Saint Jean Institut University  
**Encadrant** : Pr. NGUEFACK  
**Année académique** : 2025-2026
