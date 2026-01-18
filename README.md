# WhereIsUnicorn

API + dashboard pour prédire le succès des startups (XGBoost supervisé) et leur cluster (KMeans non supervisé).

## Structure
- `data/raw/` : CSV bruts (`objects`, `funding_rounds`, `investments`, `acquisitions`).
- `data/processed/processed_startups.csv` : table nettoyée (df_final).
- `models/unicorn_model.pkl` : pipeline scikit-learn + XGBoost (supervisé).
- `models/unicorn_clusters.pkl` : modèle KMeans (non supervisé).
- `api/main.py` : FastAPI avec deux routes (`/predict`, `/cluster`).
- `dashboard.py` : Streamlit pour top 5 + appels API.
- `requirements.txt` : dépendances (scikit-learn épinglé en 1.6.1).

## Installation
```bash
pip install -r requirements.txt
```

## Lancer l’API (terminal 1)
```bash
cd api
uvicorn main:app --reload --port 8000
```
Docs interactives : http://127.0.0.1:8000/docs

## Tester l’API en ligne de commande
- Supervisé :
```bash
python - <<'PY'
import requests
payload={"category_code":"web","country_code":"USA","funding_total_usd":39750000,"relationships":17,"total_funding_usd":39750000,"investor_count":4,"cluster_profile":2}
r=requests.post("http://127.0.0.1:8000/predict",json=payload,timeout=5)
print(r.status_code, r.text)
PY
```
- Cluster :
```bash
python - <<'PY'
import requests
payload={"funding_total_usd":39750000,"funding_rounds":3,"relationships":17,"total_funding_usd":39750000,"funding_rounds_count":3,"investor_count":4}
r=requests.post("http://127.0.0.1:8000/cluster",json=payload,timeout=5)
print(r.status_code, r.text)
PY
```

## Lancer le dashboard (terminal 2)
```bash
streamlit run dashboard.py
```
- Sidebar : choisir le mode
  - **Succès (supervisé)** : secteur (`category_code`), pays, funding, relations, investisseurs, cluster_profile (0/1/2).
  - **Cluster (non supervisé)** : funding, funding_rounds, funding_rounds_count, relations, investisseurs.
- Le top 5 est calculé depuis `data/processed/processed_startups.csv`.

## Points clés
- Vérifie la présence des artefacts : `models/unicorn_model.pkl` et `models/unicorn_clusters.pkl`.
- scikit-learn doit être en 1.6.1 (épinglé dans `requirements.txt`) pour charger les pickles.
- L’API suppose que tu lances `uvicorn` depuis `api/` (chemins relatifs déjà réglés).

## Modèles et entraînement (résumé)
- **Supervisé (`unicorn_model.pkl`)**  
  - Pipeline scikit-learn : `ColumnTransformer` (StandardScaler sur numériques, OrdinalEncoder sur catégorielles) + `XGBoostClassifier`.  
  - Features d’entrée attendues par l’API : `category_code`, `country_code`, `funding_total_usd`, `relationships`, `total_funding_usd`, `investor_count`, `cluster_profile`.  
  - Donnée cible : `is_success` (1 = acquisition/IPO, 0 sinon).  
  - Perf indicative notebook : ~80% accuracy, ~57% recall sur la classe succès (dataset déséquilibré).

- **Non supervisé (`unicorn_clusters.pkl`)**  
  - KMeans (k=3) sur 6 features numériques : `funding_total_usd`, `funding_rounds`, `relationships`, `total_funding_usd`, `funding_rounds_count`, `investor_count`.  
  - PCA utilisée en exploration pour visualiser les clusters (non requis en prod).  
  - Interprétation : cluster 2 = “élite” (taux de succès historique ~54%), cluster 0 faible (~4%), cluster 1 intermédiaire.

- **Pipeline de données**  
  - Bruts en `data/raw/` (objects, funding_rounds, investments, acquisitions).  
  - Nettoyage/agrégation dans le notebook → `data/processed/processed_startups.csv` (df_final) utilisé pour le top 5 et pour les tests API.

## Dépannage rapide
- Bandeau rouge dans le dashboard : l’API ne répond pas ou renvoie une erreur → regarde les logs `uvicorn` et la réponse détaillée dans le dashboard.
- Erreur de modèle non chargé : assure-toi que les `.pkl` sont dans `models/` et que les versions de dépendances sont installées (`pip install -r requirements.txt`).
