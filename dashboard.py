import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Configuration de la page
st.set_page_config(page_title="WhereIsUnicorn - VC Dashboard", layout="wide")

st.title("🦄 WhereIsUnicorn : Intelligence Prédictive pour VCs")
st.markdown("---")

# 1. Chargement de la "Table de Vérité" (Processed Data)
@st.cache_data
def load_data():
    # Cherche d'abord le chemin "officiel", puis tout fichier commençant par processed_ dans data/processed
    base = Path(__file__).resolve().parent
    candidates = [
        base / "data" / "processed" / "processed_startups.csv",
    ]

    # Auto-détection souple (pratique si le fichier a été renommé)
    processed_dir = base / "data" / "processed"
    if processed_dir.exists():
        for f in processed_dir.iterdir():
            if f.name.startswith("processed") and f.suffix == ".csv":
                candidates.append(f)

    for path in candidates:
        if path.exists():
            return pd.read_csv(path)

    st.error("Fichier processed introuvable dans data/processed/. "
             "Vérifie que processed_startups.csv est bien présent.")
    return pd.DataFrame()

df = load_data()

# --- SECTION 1 : LE TOP 5 DES PÉPITES ---
st.subheader("🚀 Top 5 des Licornes Potentielles (Non encore acquises)")

# On filtre les entreprises qui ne sont pas encore un succès et on trie par score
# Note : on utilise le cluster 'élite' identifié dans ton notebook
top_5 = df[df['is_success'] == 0].sort_values(by=['relationships', 'total_funding_usd'], ascending=False).head(5)

cols = st.columns(5)
for i, (index, row) in enumerate(top_5.iterrows()):
    with cols[i]:
        st.metric(label=row['name'], value=f"{int(row['relationships'])} Rel.", delta="Haut Potentiel")
        st.caption(f"Secteur : {row['category_code']}")

# --- SECTION 2 : TESTER UNE STARTUP (Connexion API) ---
st.sidebar.header("🔍 Analyser un nouveau dossier")
# Le choix du mode est en dehors du form pour que le changement déclenche le rerun immédiat
mode = st.sidebar.radio("Type de prédiction", ["Succès (supervisé)", "Cluster (non supervisé)"])

with st.sidebar.form("prediction_form"):
    name = st.text_input("Nom de la startup")

    # Champs communs
    funding = st.number_input("Total levé (USD)", min_value=0.0)
    rel = st.slider("Nombre de relations", 0, 100, 10)
    investor_count = st.number_input("Nombre d'investisseurs", min_value=0, value=5)

    if mode == "Succès (supervisé)":
        cat = st.selectbox("Secteur", df['category_code'].unique())
        country = st.selectbox("Pays", df['country_code'].unique())
        cluster_profile = st.select_slider("Cluster profil (issu du KMeans)", options=[0, 1, 2], value=2)
    else:
        # Champs spécifiques au clustering (6 features attendues)
        funding_rounds = st.number_input("Nombre de tours de financement", min_value=0.0, value=1.0)
        funding_rounds_count = st.number_input("Nombre total de tours (count)", min_value=0.0, value=1.0)

    submit = st.form_submit_button("Lancer la prédiction")

if submit:
    if mode == "Succès (supervisé)":
        payload = {
            "category_code": str(cat),
            "country_code": str(country),
            "funding_total_usd": float(funding),
            "relationships": int(rel),
            "total_funding_usd": float(funding),
            "investor_count": int(investor_count),
            "cluster_profile": int(cluster_profile),
        }
        endpoint = "http://127.0.0.1:8000/predict"
    else:
        payload = {
            "funding_total_usd": float(funding),
            "funding_rounds": float(funding_rounds),
            "relationships": float(rel),
            "total_funding_usd": float(funding),
            "funding_rounds_count": float(funding_rounds_count),
            "investor_count": float(investor_count),
        }
        endpoint = "http://127.0.0.1:8000/cluster"

    try:
        response = requests.post(endpoint, json=payload, timeout=5)
    except requests.RequestException as exc:
        st.error(f"Appel API impossible : {exc}. Vérifie que l'API tourne sur 127.0.0.1:8000.")
    else:
        if response.status_code != 200:
            st.error(f"API joignable mais réponse {response.status_code} : {response.text}")
        else:
            res = response.json()
            st.subheader(f"Résultat pour {name or 'Startup'}")
            if mode == "Succès (supervisé)":
                if "confidence_score" not in res:
                    st.error(f"Réponse inattendue de l'API : {res}")
                else:
                    prob = float(res['confidence_score'].replace('%',''))
                    fig = px.pie(
                        values=[prob, 100 - prob],
                        names=['Confiance', 'Risque'],
                        hole=0.7,
                        color_discrete_sequence=['#00CC96', '#EF553B'],
                    )
                    st.plotly_chart(fig)
                    st.write(f"**Verdict :** {res.get('recommendation', '')}")
            else:
                if "cluster_label" not in res:
                    st.error(f"Réponse inattendue de l'API : {res}")
                else:
                    st.success(f"Cluster assigné : {res['cluster_label']} — {res.get('cluster_profile', '')}")
