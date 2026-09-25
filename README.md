# Prêt à Dépenser — API de scoring crédit (déploiement & monitoring)

Projet OpenClassrooms **« Déployez et monitorez votre modèle de scoring »** (MLOps 2/2).
Mise en production du modèle de scoring XGBoost versionné avec MLflow au projet précédent :
API FastAPI, conteneurisation Docker, pipeline CI/CD GitHub Actions, journalisation des
prédictions dans PostgreSQL, métriques Prometheus/Grafana, analyse de dérive (Evidently),
dashboard Streamlit et optimisation post-déploiement.

> Documentation complète en cours de rédaction : voir les sections ajoutées au fil des phases.

## Démarrage rapide

```bash
uv sync --all-extras --all-groups     # installe l'environnement (Python 3.13)
make ci                               # lint + typage + tests
make run                              # API sur http://localhost:8000 (Swagger : /docs)
```
