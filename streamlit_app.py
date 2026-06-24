#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Application Streamlit : Déterminants de la Mortalité Néonatale au Cameroun
Analyse EDS 2018 avec comparaison SMOTE vs Sans SMOTE

Auteur : BAPFUBUSA SIAPZE Rose Ange — Master 1 Data Science
Encadrant : Pr. NGUEFACK
================================================================================
Structure modulaire :
  - Module 1 : Vue d'ensemble (KPIs, contexte, distribution)
  - Module 2 : Analyse descriptive (statistiques pondérées)
  - Module 3 : Tests bivariés (Chi-deux, t-tests)
  - Module 4 : Machine Learning Sans SMOTE
  - Module 5 : Machine Learning Avec SMOTE
  - Module 6 : Comparaison systématique
  - Module 7 : Résultats & Conclusions (OR, recommandations)
  - Module 8 : Outil de prédiction (modèle XGBoost + SMOTE)

Déploiement : streamlit run streamlit_app.py
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

# Configuration matplotlib pour Streamlit
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (10, 5)

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="Mortalité Néonatale — EDS Cameroun 2018",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS PERSONNALISÉ
# =============================================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a5276;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #5d6d7e;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .section-header {
        background-color: #1a5276;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    .info-box {
        background-color: #d6eaf8;
        border-left: 4px solid #2980b9;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fdebd0;
        border-left: 4px solid #e67e22;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d5f5e3;
        border-left: 4px solid #27ae60;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .smote-highlight {
        background-color: #f5eef8;
        border-left: 4px solid #8e44ad;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FONCTIONS DE CHARGEMENT (mise en cache)
# =============================================================================
@st.cache_data
def load_data(file):
    """Charge les données depuis le fichier uploadé."""
    return pd.read_csv(file, sep=";")

@st.cache_data
def generate_demo_data():
    """Génère des données réalistes calées sur les statistiques EDS 2018."""
    np.random.seed(42)
    n = 5000

    neonatal_mort = np.random.choice([0, 1], size=n, p=[0.968, 0.032])

    education_options = ["Aucun", "Primaire", "Secondaire", "Superieur"]
    wealth_options = ["Poorest", "Poorer", "Middle", "Richer", "Richest"]
    residence_options = ["Urbain", "Rural"]
    sex_options = ["Masculin", "Feminin"]
    region_options = list(range(1, 13))
    baby_size_options = ["Tres gros", "Plus gros", "Normal", "Petit", "Tres petit"]
    birth_interval_options = ["Premiere", "<24 mois", "24-35 mois", ">=36 mois"]
    anc_options = ["Aucune", "1-3", "4-7", "8+", "Missing"]

    df = pd.DataFrame({
        "neonatal_mort": neonatal_mort,
        "maternal_age": np.random.normal(28, 7, n).clip(15, 49).astype(int),
        "maternal_age_sq": lambda d: d["maternal_age"] ** 2,
        "parity": np.random.choice(range(1, 13), n, p=[0.18, 0.17, 0.15, 0.13, 0.10, 0.08, 0.07, 0.05, 0.04, 0.02, 0.01, 0.002]),
        "sex_child": np.random.choice(sex_options, n, p=[0.49, 0.51]),
        "education": np.random.choice(education_options, n, p=[0.28, 0.37, 0.32, 0.03]),
        "wealth": np.random.choice(wealth_options, n, p=[0.19, 0.24, 0.24, 0.18, 0.15]),
        "residence": np.random.choice(residence_options, n, p=[0.44, 0.56]),
        "region": np.random.choice(region_options, n),
        "baby_size": np.random.choice(baby_size_options, n, p=[0.03, 0.06, 0.15, 0.03, 0.01]),
        "birth_interval": np.random.choice(birth_interval_options, n, p=[0.28, 0.21, 0.26, 0.25]),
        "anc_visits": np.random.choice(anc_options, n, p=[0.05, 0.15, 0.25, 0.05, 0.50]),
    })
    df["maternal_age_sq"] = df["maternal_age"] ** 2

    # Biais réalistes
    mask_small = df["baby_size"] == "Tres petit"
    df.loc[mask_small, "neonatal_mort"] = np.random.choice([0, 1], size=mask_small.sum(), p=[0.87, 0.13])

    return df

@st.cache_resource
def load_model_and_preprocessor():
    """Charge le modèle XGBoost et le préprocesseur sauvegardés."""
    model_path = Path(__file__).parent / "best_xgboost_smote_model.pkl"
    prep_path = Path(__file__).parent / "preprocessor.pkl"

    if model_path.exists() and prep_path.exists():
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        return model, preprocessor, True
    return None, None, False


# =============================================================================
# EN-TÊTE
# =============================================================================
st.markdown('<div class="main-title">🏥 Déterminants de la Mortalité Néonatale au Cameroun</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyse EDS Cameroun 2018 • Régression Logistique & Machine Learning avec SMOTE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"><b>Auteur :</b> BAPFUBUSA SIAPZE Rose Ange — Master 1 Data Science, Saint Jean Institut University</div>', unsafe_allow_html=True)

# =============================================================================
# SIDEBAR — Navigation
# =============================================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Flag_of_Cameroon.svg/225px-Flag_of_Cameroon.svg.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisir une section :",
    [
        "📊 Vue d'ensemble",
        "🔬 Analyse descriptive",
        "📈 Tests bivariés",
        "🤖 Machine Learning — Sans SMOTE",
        "⚖️ Machine Learning — Avec SMOTE",
        "🔄 Comparaison SMOTE",
        "📋 Résultats & Conclusions",
        "🩺 Outil de prédiction",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Source :** EDS Cameroun 2018 (CMBR71FL.SAV)")
st.sidebar.markdown("**N :** 33 988 naissances (2013–2018)")
st.sidebar.markdown("**Méthode :** Régression logistique pondérée + 5 algorithmes ML")
st.sidebar.markdown("---")

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================
uploaded_file = st.sidebar.file_uploader(
    "📂 Charger neonatal_mortality_data.csv",
    type=["csv"],
    help="Fichier CSV exporté (séparateur ;). Sinon, données de démonstration utilisées.",
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    data_source = "📂 Données réelles chargées"
else:
    df = generate_demo_data()
    data_source = "⚠️ Données de démonstration (uploadez votre CSV pour l'analyse réelle)"

st.sidebar.markdown(f"**{data_source}**")
st.sidebar.markdown(f"N = {len(df):,} observations")

# Chargement du modèle
model, preprocessor, model_loaded = load_model_and_preprocessor()
if model_loaded:
    st.sidebar.markdown("🧠 Modèle XGBoost (SMOTE) : **Chargé**")
else:
    st.sidebar.markdown("🧠 Modèle XGBoost (SMOTE) : *Non trouvé*")

# =============================================================================
# MODULE 1 — VUE D'ENSEMBLE
# =============================================================================
if page == "📊 Vue d'ensemble":
    st.markdown('<div class="section-header">📊 Vue d\'ensemble — Contexte et statistiques clés</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown('<div class="warning-box">⚠️ <b>Données de démonstration.</b> Uploadez votre fichier CSV dans la sidebar pour l\'analyse réelle.</div>', unsafe_allow_html=True)

    # KPIs
    total = len(df)
    deces = int(df["neonatal_mort"].sum())
    taux = df["neonatal_mort"].mean() * 100
    survie = total - deces

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{total:,}</div>
            <div class="metric-label">Naissances analysées</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#922b21,#e74c3c)">
            <div class="metric-value">{deces:,}</div>
            <div class="metric-label">Décès néonataux</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#1e8449,#27ae60)">
            <div class="metric-value">{survie:,}</div>
            <div class="metric-label">Survivants</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#7d3c98,#9b59b6)">
            <div class="metric-value">{taux:.2f}%</div>
            <div class="metric-label">Taux de mortalité</div>
        </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Distribution des classes (Déséquilibre)")
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ["Survie (0)", "Décès (1)"]
        sizes = [survie, deces]
        colors = ["#2980b9", "#e74c3c"]
        explode = (0, 0.08)
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, explode=explode,
            autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11}
        )
        ax.set_title("Distribution classes cible", fontweight="bold")
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="info-box">🔑 <b>Déséquilibre extrême :</b> la classe minoritaire (décès) représente ~3.2% — justification de SMOTE.</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Contexte de l'étude")
        st.markdown("""
| Paramètre | Valeur |
|-----------|--------|
| **Source** | EDS Cameroun 2018 (DHS) |
| **Fichier** | CMBR71FL.SAV (Birth Recode) |
| **Période** | 2013-2018 (5 ans) |
| **Cible ODD 3.2** | ≤ 12‰ naissances vivantes (2030) |
| **Taux national** | ~31.8 ‰ (pondéré) |
| **Variables** | 11 déterminants |
| **Méthodes** | Régression logistique pondérée + 5 algo ML + SMOTE |
        """)

        st.markdown("#### Approche méthodologique")
        st.markdown("""
```
EDS 2018 (N=33 988)
   ↓
Prétraitement (imputation, encodage, standardisation)
   ↓
Split stratifié 80/20 ←── CRUCIAL : préserve le ratio 3.2%/96.8%
   ↓
Sans SMOTE          Avec SMOTE (UNIQUEMENT sur Train)
   ↓                      ↓
5 modèles ML         5 modèles ML
   ↓                      ↓
Comparaison systématique (AUC-ROC, F1, Recall, Balanced Acc)
   ↓
Meilleur modèle + Importance variables + Recommandations
```
        """)

    st.markdown("---")
    st.markdown("#### Cadre conceptuel — Mosley & Chen (1984)")
    cols = st.columns(5)
    categories = [
        ("🧬", "Biologiques", "Taille bébé, parité, sexe, âge maternel"),
        ("🌍", "Environnementaux", "Milieu résidence, région"),
        ("🏥", "Soins", "ANC, intervalle inter-génésique"),
        ("🥗", "Nutritionnels", "Taille perçue (proxy RCIU/prématurité)"),
        ("📚", "Socio-économiques", "Éducation, richesse"),
    ]
    for col, (icon, cat, desc) in zip(cols, categories):
        with col:
            st.markdown(f"**{icon} {cat}**  \n_{desc}_")

    st.markdown('<div class="smote-highlight">💜 <b>SMOTE (Synthetic Minority Over-sampling TEchnique)</b> : Technique de sur-échantillonnage synthétique appliquée UNIQUEMENT sur le Train set après le split stratifié. Le Test set reste intact pour éviter toute fuite de données (data leakage).</div>', unsafe_allow_html=True)


# =============================================================================
# MODULE 2 — ANALYSE DESCRIPTIVE
# =============================================================================
elif page == "🔬 Analyse descriptive":
    st.markdown('<div class="section-header">🔬 Analyse descriptive pondérée</div>', unsafe_allow_html=True)

    cat_vars = {
        "baby_size": "Taille du bébé",
        "birth_interval": "Intervalle inter-génésique",
        "education": "Éducation maternelle",
        "wealth": "Quintile de richesse",
        "residence": "Milieu de résidence",
        "sex_child": "Sexe de l'enfant",
        "anc_visits": "Visites prénatales (ANC)",
    }

    var_sel = st.selectbox("Sélectionner une variable :", list(cat_vars.keys()),
                           format_func=lambda x: cat_vars[x])

    col1, col2 = st.columns([1.2, 1])

    with col1:
        taux_var = df.groupby(var_sel)["neonatal_mort"].agg(["sum", "count", "mean"]).reset_index()
        taux_var.columns = [var_sel, "Décès", "Effectif", "Taux"]
        taux_var["Taux (%)"] = (taux_var["Taux"] * 100).round(2)
        taux_var = taux_var.sort_values("Taux (%)", ascending=False)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors_bar = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(taux_var)))
        bars = ax.barh(taux_var[var_sel], taux_var["Taux (%)"], color=colors_bar, edgecolor="white")
        ax.set_xlabel("Taux de mortalité néonatale (%)")
        ax.set_title(f"Taux de mortalité par {cat_vars[var_sel]}", fontweight="bold")
        for bar, val in zip(bars, taux_var["Taux (%)"]):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(f"**Tableau : {cat_vars[var_sel]}**")
        st.dataframe(taux_var[[var_sel, "Effectif", "Décès", "Taux (%)"]].reset_index(drop=True),
                     use_container_width=True)
        st.markdown(f"""
        <div class="info-box">
        <b>Max :</b> {taux_var.iloc[0][var_sel]} → {taux_var.iloc[0]["Taux (%)"]:.1f}%<br>
        <b>Min :</b> {taux_var.iloc[-1][var_sel]} → {taux_var.iloc[-1]["Taux (%)"]:.1f}%
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Distribution des variables continues")
    col_a, col_b = st.columns(2)
    with col_a:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        df[df["neonatal_mort"] == 0]["maternal_age"].hist(alpha=0.6, label="Survie", color="#2980b9", bins=20, ax=ax2)
        df[df["neonatal_mort"] == 1]["maternal_age"].hist(alpha=0.6, label="Décès", color="#e74c3c", bins=20, ax=ax2)
        ax2.set_title("Âge maternel selon l'issue", fontweight="bold")
        ax2.set_xlabel("Âge maternel (ans)")
        ax2.legend()
        st.pyplot(fig2)
        plt.close()
    with col_b:
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        df[df["neonatal_mort"] == 0]["parity"].hist(alpha=0.6, label="Survie", color="#2980b9", bins=15, ax=ax3)
        df[df["neonatal_mort"] == 1]["parity"].hist(alpha=0.6, label="Décès", color="#e74c3c", bins=15, ax=ax3)
        ax3.set_title("Parité selon l'issue", fontweight="bold")
        ax3.set_xlabel("Nombre d'enfants (parité)")
        ax3.legend()
        st.pyplot(fig3)
        plt.close()


# =============================================================================
# MODULE 3 — TESTS BIVARIÉS
# =============================================================================
elif page == "📈 Tests bivariés":
    st.markdown('<div class="section-header">📈 Tests d\'association bivariée</div>', unsafe_allow_html=True)

    bivar_data = {
        "Variable": [
            "Taille bébé", "Intervalle inter-génésique", "ANC visits",
            "Sexe de l'enfant", "Éducation maternelle", "Parité",
            "Âge maternel", "Résidence", "Région administrative",
            "Quintile de richesse",
        ],
        "Type de test": [
            "Chi-deux", "Chi-deux", "Chi-deux",
            "Chi-deux", "Chi-deux", "Test t",
            "Test t", "Chi-deux", "Chi-deux",
            "Chi-deux",
        ],
        "Statistique": [
            "χ²=185.73", "χ²=114.39", "χ²=23.90",
            "χ²=14.33", "χ²=31.88", "t=-11.50",
            "t=-1.28", "χ²=11.83", "χ²=73.47",
            "χ²=18.47",
        ],
        "p-value": [
            "<2.2×10⁻¹⁶", "1.2×10⁻²⁴", "8.4×10⁻⁵",
            "1.5×10⁻⁴", "1.0×10⁻⁶", "<10⁻¹⁶",
            "0.20", "5.8×10⁻⁴", "1.1×10⁻¹⁰",
            "9.9×10⁻⁴",
        ],
        "Significatif (p<0.05)": [
            "✅ ✅ ✅", "✅ ✅ ✅", "✅ ✅",
            "✅ ✅", "✅ ✅", "✅ ✅ ✅",
            "❌", "✅", "✅ ✅",
            "✅ ✅",
        ],
    }
    bivar_df = pd.DataFrame(bivar_data)
    st.dataframe(bivar_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="info-box">ℹ️ <b>Décision :</b> Toutes les variables sauf âge maternel sont significativement associées à la mortalité néonatale (p < 0.05). L\'âge maternel est conservé pour le terme quadratique.</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    pvals_approx = [1e-20, 1e-12, 5.8e-5, 1.5e-4, 1e-6, 1e-16, 0.20, 5.8e-4, 1.1e-10, 9.9e-4]
    names = bivar_df["Variable"].tolist()
    log_pvals = [-np.log10(p) for p in pvals_approx]
    colors_p = ["#e74c3c" if p >= -np.log10(0.05) else "#95a5a6" for p in log_pvals]
    ax.barh(names, log_pvals, color=colors_p)
    ax.axvline(x=-np.log10(0.05), color="red", linestyle="--", label="Seuil p=0.05")
    ax.axvline(x=-np.log10(0.20), color="orange", linestyle="--", label="Seuil p=0.20")
    ax.set_xlabel("−log₁₀(p-value)")
    ax.set_title("Force d'association bivariée", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# =============================================================================
# MODULE 4 — ML SANS SMOTE (BASELINE)
# =============================================================================
elif page == "🤖 Machine Learning — Sans SMOTE":
    st.markdown('<div class="section-header">🤖 Résultats Machine Learning — Sans SMOTE (Baseline)</div>', unsafe_allow_html=True)

    perf_no_smote = {
        "Modèle": ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"],
        "AUC-ROC": [0.684, 0.655, 0.693, 0.695, 0.684],
        "F1-Score": [0.106, 0.129, 0.017, 0.167, 0.123],
        "Recall": [0.648, 0.320, 0.009, 0.174, 0.498],
        "Precision": [0.058, 0.077, 0.095, 0.161, 0.070],
        "Balanced Acc": [0.647, 0.600, 0.503, 0.572, 0.639],
    }
    perf_df = pd.DataFrame(perf_no_smote)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        metrics = ["AUC-ROC", "F1-Score", "Recall"]
        colors_m = ["#2980b9", "#8e44ad", "#e67e22"]
        for ax, metric, color in zip(axes, metrics, colors_m):
            bars = ax.bar(range(len(perf_df)), perf_df[metric], color=color, alpha=0.85, edgecolor="white")
            ax.set_xticks(range(len(perf_df)))
            ax.set_xticklabels(["LR", "RF", "GB", "XGB", "LGBM"], fontsize=8)
            ax.set_title(metric, fontweight="bold")
            ax.set_ylim(0, 1)
            for bar, val in zip(bars, perf_df[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", fontsize=7)
        plt.suptitle("Performance Sans SMOTE (Baseline)", fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col2:
        st.dataframe(perf_df.style.highlight_max(axis=0, color="#d5f5e3"), use_container_width=True)

    st.markdown('<div class="warning-box">⚠️ <b>Observation clé :</b> Sans SMOTE, le F1-Score est très faible (0.02 à 0.17) — conséquence directe du déséquilibre 3.2%/96.8%. Les modèles apprennent à prédire la classe majoritaire (survie). La Logistic Regression offre le meilleur Recall (0.65) grâce à <code>class_weight=\'balanced\'</code>.</div>', unsafe_allow_html=True)


# =============================================================================
# MODULE 5 — ML AVEC SMOTE
# =============================================================================
elif page == "⚖️ Machine Learning — Avec SMOTE":
    st.markdown('<div class="section-header">⚖️ Résultats Machine Learning — Avec SMOTE</div>', unsafe_allow_html=True)

    st.markdown("""
    **SMOTE (Synthetic Minority Over-sampling TEchnique)** génère des exemples synthétiques de la classe minoritaire 
    en interpolant entre les k voisins les plus proches dans l'espace des features.
    
    **Pipeline utilisé :** Prétraitement → Split 80/20 → SMOTE (k=5, ratio=1:1, **uniquement sur Train**) → Modèle
    
    <div class="smote-highlight">
    🔒 <b>Principe fondamental anti-data-leakage :</b><br>
    1. Split stratifié 80/20 D'ABORD (préservation du ratio 3.2%)<br>
    2. SMOTE APRES, UNIQUEMENT sur le Train set<br>
    3. Le Test set reste INTACT (jamais suréchantillonné)
    </div>
    """, unsafe_allow_html=True)

    perf_smote = {
        "Modèle": ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"],
        "AUC-ROC": [0.686, 0.652, 0.669, 0.668, 0.668],
        "F1-Score": [0.102, 0.113, 0.091, 0.100, 0.113],
        "Recall": [0.653, 0.256, 0.055, 0.064, 0.073],
        "Precision": [0.055, 0.075, 0.261, 0.230, 0.246],
        "Balanced Acc": [0.641, 0.574, 0.529, 0.528, 0.533],
    }
    perf_smote_df = pd.DataFrame(perf_smote)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        metrics = ["AUC-ROC", "F1-Score", "Recall"]
        colors_m = ["#8e44ad", "#922b21", "#c0392b"]
        for ax, metric, color in zip(axes, metrics, colors_m):
            bars = ax.bar(range(len(perf_smote_df)), perf_smote_df[metric], color=color, alpha=0.85, edgecolor="white")
            ax.set_xticks(range(len(perf_smote_df)))
            ax.set_xticklabels(["LR", "RF", "GB", "XGB", "LGBM"], fontsize=8)
            ax.set_title(metric, fontweight="bold")
            ax.set_ylim(0, 1)
            for bar, val in zip(bars, perf_smote_df[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", fontsize=7)
        plt.suptitle("Performance Avec SMOTE", fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col2:
        st.dataframe(perf_smote_df.style.highlight_max(axis=0, color="#d5f5e3"), use_container_width=True)

    st.markdown("""
    <div class="info-box">
    📌 <b>Après SMOTE :</b> Le Train set contient ~53 000 observations équilibrées (50%/50%).<br>
    • Le Recall de la Logistic Regression reste le plus élevé (0.65)<br>
    • L'AUC-ROC est globalement stable par rapport à la baseline<br>
    • La Precision diminue car plus de faux positifs sont générés
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MODULE 6 — COMPARAISON SMOTE
# =============================================================================
elif page == "🔄 Comparaison SMOTE":
    st.markdown('<div class="section-header">🔄 Comparaison SMOTE vs Sans SMOTE — Analyse détaillée</div>', unsafe_allow_html=True)

    comparison = {
        "Modèle": ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"],
        "AUC (Sans)": [0.684, 0.655, 0.693, 0.695, 0.684],
        "AUC (Avec)": [0.686, 0.652, 0.669, 0.668, 0.668],
        "Δ AUC": [0.002, -0.003, -0.024, -0.027, -0.016],
        "F1 (Sans)": [0.106, 0.129, 0.017, 0.167, 0.123],
        "F1 (Avec)": [0.102, 0.113, 0.091, 0.100, 0.113],
        "Δ F1": [-0.004, -0.016, 0.074, -0.067, -0.010],
        "Recall (Sans)": [0.648, 0.320, 0.009, 0.174, 0.498],
        "Recall (Avec)": [0.653, 0.256, 0.055, 0.064, 0.073],
        "Δ Recall": [0.005, -0.064, 0.046, -0.110, -0.425],
    }
    comp_df = pd.DataFrame(comparison)

    st.dataframe(comp_df.style.format({c: "{:.3f}" for c in comp_df.columns if c not in ["Modèle"]}),
                use_container_width=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(comp_df))
    width = 0.35
    metrics_comp = [
        ("AUC (Sans)", "AUC (Avec)", "AUC-ROC"),
        ("F1 (Sans)", "F1 (Avec)", "F1-Score"),
        ("Recall (Sans)", "Recall (Avec)", "Recall"),
    ]
    for ax, (m_no, m_s, title) in zip(axes, metrics_comp):
        ax.bar(x - width/2, comp_df[m_no], width, label="Sans SMOTE", color="steelblue", alpha=0.85)
        ax.bar(x + width/2, comp_df[m_s], width, label="Avec SMOTE", color="coral", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(["LR", "RF", "GB", "XGB", "LGBM"])
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    plt.suptitle("Comparaison SMOTE vs Sans SMOTE — 5 modèles", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
        <b>📌 Résultats clés :</b><br>
        • <b>AUC-ROC</b> : SMOTE n'améliore pas significativement l'AUC (Δ ≤ +0.002)<br>
        • <b>Gradient Boosting</b> : meilleur gain F1 (+0.074) et Recall (+0.046)<br>
        • <b>XGBoost</b> : meilleure AUC sans SMOTE (0.695)<br>
        • <b>Logistic Regression</b> : plus stable, meilleur Recall global
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ Pourquoi SMOTE est limité ici :</b><br>
        1. Déséquilibre extrême (3.2%) → ratio synthétique ~24:1 crée des artefacts<br>
        2. Absence de variables biométriques continues (poids, APGAR)<br>
        3. Un seul facteur (taille "Très petit") capte ~8.3% de l'importance<br>
        4. Les exemples synthétiques ne reflètent pas la vraie distribution
        </div>""", unsafe_allow_html=True)


# =============================================================================
# MODULE 7 — RÉSULTATS & CONCLUSIONS
# =============================================================================
elif page == "📋 Résultats & Conclusions":
    st.markdown('<div class="section-header">📋 Résultats consolidés et conclusions</div>', unsafe_allow_html=True)

    st.markdown("#### Odds Ratios — Régression logistique pondérée")
    or_data = {
        "Variable": [
            "Taille bébé — Très petit", "Intervalle — Première", "Intervalle — <24 mois",
            "ANC — Missing", "Sexe masculin", "Parité (par enfant)",
            "Région (par unité)", "Âge maternel (quadratique)", "Âge maternel (linéaire)",
        ],
        "OR": [6.32, 1.72, 1.68, 1.57, 1.27, 1.20, 1.02, 1.001, 0.882],
        "IC_95%_Inf": [4.67, 1.44, 1.42, 1.09, 1.12, 1.16, 1.00, 1.001, 0.827],
        "IC_95%_Sup": [8.55, 2.07, 1.99, 2.28, 1.44, 1.24, 1.05, 1.002, 0.940],
        "p_value": ["<10⁻³²", "<10⁻⁸", "<10⁻⁹", "0.016", "<10⁻⁴", "<10⁻²⁹", "0.020", "0.002", "<10⁻⁴"],
        "Interpretation": [
            "↑ Risque ×6.3 (FACTEUR DOMINANT)", "↑ Risque +72% (1ère naissance)", "↑ Risque +68% (intervalle court)",
            "↑ Risque +57% (données manquantes)", "↑ Risque +27% (sexe masculin)", "↑ Risque +20% par enfant",
            "↑ Risque +2.4% par région", "Effet quadratique positif", "Effet protecteur par année"
        ],
    }
    or_df = pd.DataFrame(or_data)
    st.dataframe(or_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(or_df))
    colors_or = ["#e74c3c" if or_val > 1 else "#27ae60" for or_val in or_df["OR"]]
    ax.errorbar(or_df["OR"], y_pos,
                xerr=[or_df["OR"] - or_df["IC_95%_Inf"], or_df["IC_95%_Sup"] - or_df["OR"]],
                fmt="o", color="#2c3e50", ecolor="#95a5a6", capsize=4, markersize=7)
    ax.axvline(x=1, color="#e74c3c", linestyle="--", alpha=0.7, label="OR=1 (référence)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(or_df["Variable"], fontsize=9)
    ax.set_xlabel("Odds Ratio (échelle log)")
    ax.set_title("Forest Plot — Odds Ratios significatifs", fontweight="bold")
    ax.set_xscale("log")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("#### Conclusions principales")
    conclusions = [
        ("🔴 FACTEUR DOMINANT", "Taille 'Très petit' (OR=6.3, p<10⁻³²) — proxy de prématurité/RCIU. Le risque est multiplié par 6.3."),
        ("🔴 Parité élevée", "OR=1.20 par enfant supplémentaire (p<10⁻²⁹) — épuisement maternel, dilution des ressources."),
        ("🟡 Intervalle court", "<24 mois : OR=1.68 (p<10⁻⁹) — risque accru lié à l'épuisement nutritionnel."),
        ("🟡 Sexe masculin", "OR=1.27 — vulnérabilité biologique (maturité pulmonaire plus tardive)."),
        ("🟢 Effet protecteur âge", "Âge maternel : effet protecteur linéaire (OR=0.88) mais quadratique positif (risque aux extrêmes)."),
        ("🔵 SMOTE", "N'améliore pas significativement l'AUC globale, mais permet un meilleur équilibre F1/Recall sur Gradient Boosting."),
    ]
    for emoji_label, desc in conclusions:
        st.markdown(f"**{emoji_label} :** {desc}")

    st.markdown("---")
    st.markdown("#### Recommandations opérationnelles")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **🏥 Niveau clinique :**
        * Score de risque néonatal : parité + taille perçue + région
        * Dépistage systématique des petits nouveau-nés
        * Incubation préventive et monitoring cardiorespiratoire
        * Renforcement réanimation néonatale dans les districts à haut risque
        """)
    with col_b:
        st.markdown("""
        **📊 Niveau méthodologique :**
        * Intégrer variables biométriques continues dans futures EDS (poids, APGAR)
        * Explorer SMOTEENN ou Focal Loss comme alternatives à SMOTE
        * Développer modèle validé sur plusieurs vagues DHS
        * Modélisation par survie (Cox) pour capter la dynamique temporelle
        """)


# =============================================================================
# MODULE 8 — OUTIL DE PRÉDICTION
# =============================================================================
elif page == "🩺 Outil de prédiction":
    st.markdown('<div class="section-header">🩺 Outil de prédiction du risque néonatal</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.markdown('<div class="warning-box">⚠️ <b>Modèle non chargé.</b> L\'outil fonctionne en mode simplifié (formule logistique). Placez <code>best_xgboost_smote_model.pkl</code> et <code>preprocessor.pkl</code> dans le même dossier que cette application.</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">ℹ️ Cet outil calcule un <b>score de risque</b> basé sur les déterminants identifiés. À des fins éducatives uniquement.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👶 Caractéristiques du nouveau-né**")
        baby_size = st.selectbox("Taille perçue du bébé", ["Tres gros", "Plus gros", "Normal", "Petit", "Tres petit"],
                                  help="⚠️ 'Très petit' = facteur de risque dominant (OR=6.3)")
        sex_child = st.selectbox("Sexe de l'enfant", ["Feminin", "Masculin"])

    with col2:
        st.markdown("**👩 Caractéristiques maternelles**")
        maternal_age = st.slider("Âge maternel (ans)", 15, 49, 28)
        parity = st.slider("Parité (nombre d'enfants)", 1, 15, 3,
                           help="Plus la parité est élevée, plus le risque augmente (OR=1.20 par enfant)")
        birth_interval = st.selectbox("Intervalle inter-génésique",
                                       ["Premiere", "<24 mois", "24-35 mois", ">=36 mois"])

    with col3:
        st.markdown("**🏥 Soins et contexte**")
        anc = st.selectbox("Visites prénatales (ANC)", ["Aucune", "1-3", "4-7", "8+", "Missing"])
        residence = st.selectbox("Milieu de résidence", ["Urbain", "Rural"])
        region = st.selectbox("Région", [f"Région {i}" for i in range(1, 13)])

    st.markdown("---")

    if st.button("🔍 Calculer le score de risque", type="primary", use_container_width=True):

        # Si le modèle est chargé, utiliser le modèle XGBoost
        if model_loaded:
            # Préparer les données d'input
            input_data = pd.DataFrame({
                'maternal_age': [maternal_age],
                'maternal_age_sq': [maternal_age ** 2],
                'parity': [parity],
                'sex_child': [sex_child],
                'education': ['Secondaire'],  # Valeur par défaut
                'wealth': ['Middle'],          # Valeur par défaut
                'residence': [residence],
                'region': [int(region.replace('Région ', ''))],
                'baby_size': [baby_size],
                'birth_interval': [birth_interval],
                'anc_visits': [anc]
            })

            # Prétraitement
            input_processed = preprocessor.transform(input_data)
            probability = float(model.predict_proba(input_processed)[0, 1])
        else:
            # Mode simplifié : formule logistique basée sur les OR
            log_odds_base = -3.5
            size_or = {"Tres gros": 0.8, "Plus gros": 0.6, "Normal": 1.0, "Petit": 1.27, "Tres petit": 6.32}
            log_odds = log_odds_base + np.log(size_or[baby_size])

            if sex_child == "Masculin":
                log_odds += np.log(1.27)
            log_odds += np.log(1.20) * (parity - 1)
            log_odds += np.log(0.88) * (maternal_age - 28) + np.log(1.001) * ((maternal_age**2) - (28**2))

            interval_or = {"Premiere": 1.72, "<24 mois": 1.68, "24-35 mois": 0.51, ">=36 mois": 0.46}
            log_odds += np.log(interval_or[birth_interval])

            if anc == "Missing":
                log_odds += np.log(1.57)
            if "6" in region:
                log_odds += np.log(0.44)

            probability = 1 / (1 + np.exp(-log_odds))

        prob_pct = probability * 100

        # Affichage des résultats
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            color = "#e74c3c" if prob_pct > 10 else "#e67e22" if prob_pct > 5 else "#27ae60"
            st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,{color},{color}cc)">
                <div class="metric-value">{prob_pct:.1f}%</div>
                <div class="metric-label">Probabilité estimée de décès néonatal</div>
            </div>""", unsafe_allow_html=True)

        with col_r2:
            taux_national = 3.18
            ratio = prob_pct / taux_national
            st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#7d3c98,#9b59b6)">
                <div class="metric-value">×{ratio:.1f}</div>
                <div class="metric-label">Ratio vs taux national (3.18%)</div>
            </div>""", unsafe_allow_html=True)

        with col_r3:
            risk_level = "🔴 RISQUE ÉLEVÉ" if prob_pct > 10 else "🟡 RISQUE MODÉRÉ" if prob_pct > 5 else "🟢 RISQUE FAIBLE"
            st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#1a5276,#2980b9)">
                <div class="metric-value" style="font-size:1.3rem">{risk_level}</div>
                <div class="metric-label">Classification du risque</div>
            </div>""", unsafe_allow_html=True)

        # Facteurs de risque identifiés
        st.markdown("---")
        st.markdown("**📊 Facteurs de risque identifiés dans ce profil :**")
        risk_factors = []
        if baby_size == "Tres petit":
            risk_factors.append(("🔴", "Taille très petite", "OR = 6,32 — Facteur dominant"))
        if parity >= 6:
            risk_factors.append(("🟠", f"Parité élevée ({parity})", "OR = 1,20 par enfant"))
        if birth_interval in ["<24 mois", "Premiere"]:
            risk_factors.append(("🟠", f"Intervalle : {birth_interval}", "Risque accru"))
        if sex_child == "Masculin":
            risk_factors.append(("🟡", "Sexe masculin", "OR = 1,27"))
        if anc == "Missing":
            risk_factors.append(("🟡", "ANC non renseigné", "OR = 1,57"))

        if risk_factors:
            cols_factors = st.columns(min(len(risk_factors), 3))
            for idx, (emoji, factor, detail) in enumerate(risk_factors):
                with cols_factors[idx % len(cols_factors)]:
                    st.markdown(f"""
                    <div style="padding: 0.8rem; background: white; border-radius: 8px;
                                border-left: 4px solid #e74c3c; margin-bottom: 0.5rem;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                        <span style="font-size: 1.3rem;">{emoji}</span> <b>{factor}</b><br>
                        <span style="font-size: 0.85rem; color: #666;">{detail}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("✅ Aucun facteur de risque majeur identifié dans ce profil.")

        # Recommandations
        if probability > 0.05:
            st.markdown("""
            <div class="warning-box">
            🚨 <b>Recommandations urgentes :</b><br>
            • Incubation / réanimation néonatale immédiate<br>
            • Monitorage cardiorespiratoire continu<br>
            • Bilan infectieux et glycémique rapide
            </div>
            """, unsafe_allow_html=True)
        elif baby_size == "Tres petit":
            st.markdown("""
            <div class="warning-box">
            ⚠️ <b>Attention particulière :</b><br>
            La taille "Très petit" est le facteur de risque le plus discriminant (OR=6.32).
            Surveillance renforcée conseillée malgré une probabilité modérée.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
            ✅ <b>Surveillance standard :</b><br>
            Aucun facteur de risque majeur identifié. Protocole de soins néonatals habituel.
            </div>
            """, unsafe_allow_html=True)

        st.caption("**⚠️ Disclaimer :** Cet outil est indicatif, basé sur un modèle statistique. La décision clinique doit toujours intégrer l'évaluation complète par un professionnel de santé qualifié.")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#95a5a6;font-size:0.85rem'>"
    "© 2025 BAPFUBUSA SIAPZE Rose Ange — Saint Jean Institut University — Master 1 Data Science<br>"
    "Données : EDS Cameroun 2018 (DHS Program) — Usage académique uniquement</div>",
    unsafe_allow_html=True,
)
