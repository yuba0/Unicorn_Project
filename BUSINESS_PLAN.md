# WhereIsUnicorn – Business Plan (Synthèse)

## Contexte métier
Les fonds de Venture Capital (VC) reçoivent des centaines de dossiers par mois. Le tri initial repose souvent sur l’intuition, ce qui augmente le risque de rater des pépites ou d’investir trop tard. Objectif : fournir un score de succès et un profil de cluster pour prioriser les startups à investiguer.

## Jeu de données source
- Origine : jeu de données type Crunchbase (dispo sur Kaggle) contenant des historiques de levées et relations.
- Volume : ~64k entités, 4 fichiers bruts (`objects`, `funding_rounds`, `investments`, `acquisitions`) totalisant ~300 Mo.
- Variables clés : secteur (`category_code`), pays (`country_code`), montants levés (`funding_total_usd`, `total_funding_usd`), réseaux (`relationships`), nombre d’investisseurs, statut (acquisition/IPO/operating).

## Objectif Data Science (précis)
Prédire la probabilité de succès (acquisition/IPO) d’une startup et identifier son cluster de profil. Formulé : « Estimer P(succès) à horizon 3-5 ans à partir des signaux financiers et relationnels historiques, et positionner la startup dans un cluster de maturité/élitisme. »

## Solution proposée
- Modèle supervisé : pipeline scikit-learn (encodage + StandardScaler) + XGBoostClassifier → `unicorn_model.pkl`.
- Modèle non supervisé : KMeans (k=3) sur 6 features financières et réseau → `unicorn_clusters.pkl`.
- API FastAPI : deux routes (`/predict`, `/cluster`), santé via `/`.
- Dashboard Streamlit : top 5 des pépites (données processed) + formulaires de requête API (supervisé/cluster).

## Architecture data
- Raw : `data/raw/` (CSV bruts Kaggle/Crunchbase).
- Processed : `data/processed/processed_startups.csv` (df_final nettoyée/agrégée).
- Models : `models/` (artefacts .pkl).
- Services : `api/main.py` (FastAPI), `dashboard.py` (Streamlit).

## Impact métier
- Priorisation du deal-flow : réduction du temps de screening (focus sur top N).
- Réduction du risque d’oubli : recall ~57% sur les succès → moins de pépites ratées.
- Narratif VC : preuve quantitative que le réseau (relationships, investor_count) pèse plus que le seul montant levé.

## Limites
- Données historiques : biais temporel (critères 2010 ≠ 2026).
- Variables manquantes : pas de qualité produit, équipe fondatrice, traction récente.
- LFS/données lourdes : les bruts ne sont pas versionnés (GitHub 100 Mo) ; besoin d’un stockage externe.
- Version de dépendances : scikit-learn épinglé (1.6.1) pour compatibilité pickle.

## Perspectives
- Rafraîchissement régulier : ingestion incrémentale des levées et acquisitions.
- Enrichissement NLP : actualités, sentiment, thèmes (BERTopic/BERT) + base vectorielle.
- Explicabilité : SHAP pour justifier les décisions auprès des investisseurs.
- Monitoring : suivi drift, réentraînement trimestriel, alertes Slack/Email.
- Packaging : Docker + CI/CD, hébergement sur un PaaS (Render, Railway) ou cloud (AWS/GCP/Azure).

## Roadmap (90 jours)
1. Semaine 1-2 : pipeline d’ingestion raw→processed automatisé, tests API/dash end-to-end.
2. Semaine 3-4 : ajout SHAP, métriques de monitoring (latence, taux d’erreur, drift).
3. Semaine 5-8 : enrichissement NLP (news, sentiment), stockage vectoriel, nouvelle version du modèle.
4. Semaine 9-12 : Docker + CI/CD, déploiement staging, collecte de feedback utilisateur VC.

## KPI à suivre
- Métier : taux de rappel sur les succès, précision sur le top N, temps moyen de screening économisé.
- Produit : latence API p95, taux d’erreur, uptime.
- Modèle : drift des distributions, AUC/PR, calibration des probabilités.

## Modèle économique (SaaS B2B)
- Abonnement mensuel par analyste/équipe VC, paliers par volume de requêtes API.
- Option “signals” push : alertes quand une startup migre vers le cluster élite.
- Services premium : intégration CRM, export rapports d’investissement, support prioritaire.
