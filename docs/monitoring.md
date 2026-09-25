# Interpréter le monitoring

## 1. Grafana — « Scoring API — monitoring technique » (http://localhost:3000)

| Panneau | Ce qu'il mesure | Quand s'inquiéter |
|---|---|---|
| Requêtes / s | Débit sur `/predict` | Chute brutale (client en panne) ou pic anormal |
| Latence p99 (SLO 180 ms) | 99 % des requêtes répondent sous cette valeur | Orange > 120 ms, rouge > 180 ms |
| Taux d'erreur 5xx | Erreurs internes / total | > 1 % : modèle ou dépendance en panne |
| Taux de refus | Part de décisions « Refusé » (référence 32 %) | Hors de 25–40 % : dérive ou changement de population |
| Latence p50/p95/p99 | Distribution des temps de réponse, ligne rouge = SLO | p95 qui se rapproche du p99 : saturation |
| Inférence p99 par backend | Temps du modèle seul | Hausse sans hausse de trafic : contention CPU |
| Requêtes par code HTTP | 200 / 422 / 5xx | Beaucoup de 422 : contrat d'entrée cassé côté client |
| Heatmap des scores | Distribution des probabilités dans le temps | Déplacement de la masse vers le haut : population plus risquée |
| Journalisation | Taille de file, lots, abandons, erreurs DB | Abandons > 0 : base trop lente, augmenter `LOG_BATCH_SIZE`/le pool |
| Part < 180 ms | Conformité SLO | < 99 % |

Métriques brutes : `curl localhost:8000/metrics`. Prometheus : http://localhost:9090.

## 2. Streamlit — suivi métier et dérive (http://localhost:8501)

- **Métier** : volume, taux de refus avec intervalle de confiance (Wilson), distribution des
  scores vs référence, tableau par scénario.
- **Dérive des données** : PSI par variable (0,10 à surveiller, 0,25 dérive), test KS avec
  correction de Bonferroni, comparaison d'une variable. La dérive « dataset » est déclarée si
  plus de 30 % des colonnes sont en alerte et qu'au moins 200 lignes sont disponibles.
- **Performance** : sans labels, coût et AUC *attendus* (probabilités calibrées par isotonique
  sur la référence) ; avec labels différés (`make labels`), métriques *réalisées*. Un écart
  attendu/réalisé signale un changement de la relation entrées → défaut (concept drift).
- **Seuil de décision** : courbe coût métier `(10·FN + FP)/n` en fonction du seuil, sur la
  référence et sur la production labellisée ; poids FN ajustable.
- **Technique** : latences serveur stockées (p50/p95/p99), inférence par backend, dernières erreurs.

## 3. Notebook `notebooks/drift_analysis.ipynb`

Analyse complète et reproductible (Evidently + PSI/KS + performance + seuil) sur les données
stockées, scénario par scénario. Ré-exécuter :

```bash
make simulate SCENARIO=drift_ext_source N=800 && make labels
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/drift_analysis.ipynb
```

## 4. Que faire en cas d'alerte ?

| Signal | Cause probable | Action |
|---|---|---|
| PSI ≥ 0,25 sur EXT_SOURCE_* | Fournisseur de score externe modifié / en panne | Vérifier le taux de NaN ; contacter le fournisseur ; surveiller le coût réalisé |
| PSI ≥ 0,25 sur AMT_* / ratios | Nouvelle offre, inflation, changement de segment | Confirmer avec le métier ; envisager ré-entraînement |
| Taux de refus hors IC | Dérive des entrées ou du score | Ouvrir la page Dérive ; ne pas toucher au seuil sans labels |
| Coût réalisé > référence + 10 % | Concept drift | Ré-entraîner (`scripts/retrain_stub.py`), re-calibrer le seuil |
| 422 en hausse | Contrat cassé côté client | Lire `api_errors.detail` (champ, type) |
| p99 > 180 ms | Contention CPU, base lente | Vérifier `XGB_NTHREAD=1`, la file de journalisation, les ressources du conteneur |
