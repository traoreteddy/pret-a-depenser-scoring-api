"""Dashboard Streamlit — suivi métier et dérive du modèle de scoring en production.

Lance : uv run streamlit run dashboard/app.py   (DATABASE_URL lu dans .env)
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from scoring_api.monitoring.drift import (  # noqa: E402
    MONITORED_COLUMNS,
    PSI_ALERT,
    PSI_WARN,
    dataset_summary,
    drift_table,
    refusal_rate_ci,
)
from scoring_api.monitoring.performance import (  # noqa: E402
    expected_metrics,
    fit_calibrator,
    realized_metrics,
    reliability_table,
)
from scoring_api.monitoring.production import (  # noqa: E402
    expand_inputs,
    load_errors,
    load_predictions,
    scenario_list,
)
from scoring_api.monitoring.reference import load_reference, load_reference_metrics  # noqa: E402
from scoring_api.monitoring.threshold import cost_curve, optimal_threshold  # noqa: E402

st.set_page_config(
    page_title="Prêt à Dépenser — monitoring du scoring", page_icon="📊", layout="wide"
)


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip()
    return ""


DB_URL = _database_url()


@st.cache_data(ttl=60, show_spinner=False)
def _reference() -> tuple[pd.DataFrame, dict]:  # type: ignore[type-arg]
    return load_reference(), load_reference_metrics()


@st.cache_data(ttl=30, show_spinner="Lecture de PostgreSQL…")
def _production(hours: float, scenario: str | None) -> pd.DataFrame:
    since = datetime.now(UTC) - timedelta(hours=hours)
    return load_predictions(DB_URL, since=since, scenario=scenario)


@st.cache_data(ttl=30, show_spinner=False)
def _scenarios() -> list[str]:
    return scenario_list(DB_URL)


@st.cache_data(ttl=60, show_spinner=False)
def _errors(hours: float) -> pd.DataFrame:
    return load_errors(DB_URL, since=datetime.now(UTC) - timedelta(hours=hours))


# ----------------------------------------------------------------------------- barre latérale
st.sidebar.title("Prêt à Dépenser")
st.sidebar.caption("Monitoring du modèle `scoring_credit` v1")
page = st.sidebar.radio(
    "Page", ["Métier", "Dérive des données", "Performance", "Seuil de décision", "Technique"]
)
hours = st.sidebar.slider("Fenêtre (heures)", min_value=1, max_value=24 * 30, value=24 * 7)
if not DB_URL:
    st.error("DATABASE_URL non défini (variable d'environnement ou fichier .env).")
    st.stop()
scen_options = ["(tous)", *_scenarios()]
scen_choice = st.sidebar.selectbox("Scénario (en-tête X-Scenario)", scen_options)
scenario = None if scen_choice == "(tous)" else scen_choice
st.sidebar.markdown("---")
st.sidebar.caption("Référence : 20 000 dossiers de X_test (P6). Seuils PSI : 0,10 / 0,25.")

reference, ref_metrics = _reference()
preds = _production(float(hours), scenario)
threshold = (
    float(preds["threshold"].iloc[-1]) if not preds.empty else float(ref_metrics["threshold"])
)

if preds.empty:
    st.warning(
        "Aucune prédiction sur la fenêtre choisie. Lancez `make simulate` pour générer du trafic."
    )
    st.stop()

# ----------------------------------------------------------------------------- Métier
if page == "Métier":
    st.title("Suivi métier")
    n = len(preds)
    refused = int((preds["decision"] == "Refusé").sum())
    rate, lo, hi = refusal_rate_ci(refused, n)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dossiers scorés", f"{n:,}".replace(",", " "))
    c2.metric(
        "Taux de refus",
        f"{rate:.1%}",
        delta=f"{rate - ref_metrics['refusal_rate']:+.1%} vs réf. {ref_metrics['refusal_rate']:.1%}",
        delta_color="inverse",
    )
    c3.metric("IC 95 % du taux de refus", f"[{lo:.1%} ; {hi:.1%}]")
    c4.metric(
        "Score moyen",
        f"{preds['proba_defaut'].mean():.3f}",
        delta=f"{preds['proba_defaut'].mean() - ref_metrics['proba_mean']:+.3f} vs réf.",
        delta_color="inverse",
    )

    st.subheader("Distribution des probabilités de défaut")
    fig = go.Figure()
    fig.add_histogram(
        x=reference["proba_defaut"],
        histnorm="probability density",
        name="Référence (X_test)",
        opacity=0.55,
        nbinsx=40,
    )
    fig.add_histogram(
        x=preds["proba_defaut"],
        histnorm="probability density",
        name="Production",
        opacity=0.55,
        nbinsx=40,
    )
    fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"seuil {threshold:.2f}")
    fig.update_layout(barmode="overlay", height=380, margin={"t": 30})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Volume et taux de refus dans le temps")
    ts = (
        preds.set_index("ts")
        .resample("15min")
        .agg(
            n=("proba_defaut", "size"),
            refus=("decision", lambda s: float((s == "Refusé").mean()) if len(s) else np.nan),
        )
        .dropna()
    )
    if not ts.empty:
        fig2 = px.bar(ts.reset_index(), x="ts", y="n", labels={"n": "dossiers / 15 min", "ts": ""})
        fig2.add_scatter(
            x=ts.index,
            y=ts["refus"] * ts["n"].max(),
            name="taux de refus (échelle relative)",
            yaxis="y",
            mode="lines",
        )
        fig2.add_hline(
            y=ref_metrics["refusal_rate"] * ts["n"].max(),
            line_dash="dot",
            annotation_text="refus réf. 32 %",
        )
        fig2.update_layout(height=320, margin={"t": 30})
        st.plotly_chart(fig2, use_container_width=True)

    if scenario is None and preds["scenario"].notna().any():
        st.subheader("Par scénario simulé")
        g = preds.groupby(preds["scenario"].fillna("(réel)")).agg(
            dossiers=("proba_defaut", "size"),
            score_moyen=("proba_defaut", "mean"),
            taux_refus=("decision", lambda s: float((s == "Refusé").mean())),
        )
        st.dataframe(
            g.style.format({"score_moyen": "{:.3f}", "taux_refus": "{:.1%}"}),
            use_container_width=True,
        )

# ----------------------------------------------------------------------------- Dérive
elif page == "Dérive des données":
    st.title("Dérive des données (data drift)")
    current = expand_inputs(preds)
    if current.empty:
        st.warning("Aucune entrée échantillonnée (raw_input) sur la fenêtre.")
        st.stop()
    table = drift_table(reference, current)
    summary = dataset_summary(table, len(current))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Lignes analysées",
        f"{summary['n_current']:,}".replace(",", " "),
        help="Entrées échantillonnées (raw_input conservé)",
    )
    c2.metric("Colonnes en dérive (PSI ≥ 0,25)", summary["n_psi_drift"])
    c3.metric("Colonnes à surveiller (PSI ≥ 0,10)", summary["n_psi_warn"])
    c4.metric("PSI du score", f"{summary['score_psi']:.3f}")
    if not summary["sufficient_data"]:
        st.info("Moins de 200 lignes : les tests ne sont pas concluants.")
    elif summary["dataset_drift"]:
        st.error("Dérive du jeu de données : plus de 30 % des colonnes en alerte.")
    elif summary["n_psi_drift"]:
        st.warning("Dérive localisée sur certaines variables (voir tableau).")
    else:
        st.success("Aucune dérive détectée.")

    st.subheader("PSI par variable")
    top = table.head(25)
    fig = px.bar(
        top,
        x="psi",
        y="colonne",
        orientation="h",
        color="statut_psi",
        color_discrete_map={
            "stable": "#2a9d8f",
            "à surveiller": "#e9c46a",
            "dérive": "#e76f51",
            "n/a": "#adb5bd",
        },
    )
    fig.add_vline(x=PSI_WARN, line_dash="dot")
    fig.add_vline(x=PSI_ALERT, line_dash="dash")
    fig.update_layout(height=650, yaxis={"autorange": "reversed"}, margin={"t": 20})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        table.style.format(
            {
                "psi": "{:.3f}",
                "ks_stat": "{:.3f}",
                "ks_p": "{:.2e}",
                "moy_ref": "{:.3f}",
                "moy_cur": "{:.3f}",
                "nan_ref": "{:.1%}",
                "nan_cur": "{:.1%}",
            }
        ),
        use_container_width=True,
        height=400,
    )

    st.subheader("Comparer une variable")
    col = st.selectbox("Variable", [c for c in MONITORED_COLUMNS if c in current.columns])
    fig3 = go.Figure()
    fig3.add_histogram(
        x=reference[col], histnorm="probability density", name="Référence", opacity=0.55, nbinsx=40
    )
    fig3.add_histogram(
        x=current[col], histnorm="probability density", name="Production", opacity=0.55, nbinsx=40
    )
    fig3.update_layout(barmode="overlay", height=340, margin={"t": 20})
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------------- Performance
elif page == "Performance":
    st.title("Performance du modèle en production")
    st.caption(
        "Sans labels : estimation à partir des probabilités (CBPE simplifié, modèle calibré sur la référence). Avec labels différés : métriques réalisées."
    )
    y_ref, p_ref = reference["TARGET"].to_numpy(), reference["proba_defaut"].to_numpy()
    cal = fit_calibrator(y_ref, p_ref)
    rt = reliability_table(y_ref, p_ref)
    exp = expected_metrics(preds["proba_defaut"].to_numpy(), threshold, calibrator=cal)
    labelled = preds[preds["target"].notna()]
    real = (
        realized_metrics(
            labelled["target"].to_numpy(), labelled["proba_defaut"].to_numpy(), threshold
        )
        if len(labelled) >= 50
        else None
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Coût métier attendu / dossier",
        f"{exp['cout_attendu']:.3f}",
        delta=f"{exp['cout_attendu'] - ref_metrics['business_cost']:+.3f} vs réf. {ref_metrics['business_cost']:.3f}",
        delta_color="inverse",
    )
    c2.metric(
        "AUC attendue",
        f"{exp['auc_attendue']:.3f}",
        delta=f"{exp['auc_attendue'] - ref_metrics['auc']:+.3f} vs réf. {ref_metrics['auc']:.3f}",
    )
    c3.metric(
        "Taux de défaut attendu",
        f"{exp['taux_defaut_attendu']:.1%}",
        delta=f"{exp['taux_defaut_attendu'] - ref_metrics['default_rate']:+.1%} vs réf.",
        delta_color="inverse",
    )
    if real:
        st.subheader(f"Métriques réalisées ({real['n']} dossiers labellisés)")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric(
            "Coût réalisé",
            f"{real['cout']:.3f}",
            delta=f"{real['cout'] - exp['cout_attendu']:+.3f} vs attendu",
            delta_color="inverse",
        )
        d2.metric(
            "AUC réalisée",
            f"{real['auc']:.3f}",
            delta=f"{real['auc'] - exp['auc_attendue']:+.3f} vs attendue",
        )
        d3.metric("Rappel (défauts refusés)", f"{real['rappel']:.1%}")
        d4.metric("Taux de défaut réel", f"{real['taux_defaut']:.1%}")
    else:
        st.info("Pas assez de labels différés (min. 50) : lancez `make labels`.")

    st.subheader("Calibration du modèle (référence)")
    fig = go.Figure()
    fig.add_scatter(
        x=rt["proba_moyenne"], y=rt["taux_observe"], mode="lines+markers", name="modèle"
    )
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", name="parfait")
    fig.update_layout(
        xaxis_title="probabilité prédite",
        yaxis_title="taux de défaut observé",
        height=360,
        margin={"t": 20},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Le modèle XGBoost entraîné avec scale_pos_weight surestime la probabilité brute (classes rééquilibrées) : le calibrateur isotonique corrige avant l'estimation sans labels."
    )

# ----------------------------------------------------------------------------- Seuil
elif page == "Seuil de décision":
    st.title("Calibrage du seuil de décision")
    st.caption(
        "Coût métier = (10 × faux négatifs + 1 × faux positifs) / n, comme au P6. Le seuil 0,48 minimise ce coût sur la validation croisée."
    )
    cost_fn = st.slider("Poids d'un défaut non détecté (FN)", 1, 20, 10)
    curve_ref = cost_curve(
        reference["TARGET"].to_numpy(), reference["proba_defaut"].to_numpy(), cost_fn=float(cost_fn)
    )
    t_ref, c_ref = optimal_threshold(curve_ref)
    labelled = preds[preds["target"].notna()]
    fig = go.Figure()
    fig.add_scatter(x=curve_ref["seuil"], y=curve_ref["cout"], name="référence (X_test)")
    if len(labelled) >= 200:
        curve_prod = cost_curve(
            labelled["target"].to_numpy(),
            labelled["proba_defaut"].to_numpy(),
            cost_fn=float(cost_fn),
        )
        t_prod, c_prod = optimal_threshold(curve_prod)
        fig.add_scatter(
            x=curve_prod["seuil"],
            y=curve_prod["cout"],
            name=f"production labellisée (n={len(labelled)})",
        )
        st.metric(
            "Seuil optimal en production",
            f"{t_prod:.2f}",
            delta=f"{t_prod - t_ref:+.2f} vs référence {t_ref:.2f}",
        )
    fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"seuil en prod {threshold:.2f}")
    fig.add_vline(x=t_ref, line_dash="dot", annotation_text=f"optimum réf. {t_ref:.2f}")
    fig.update_layout(
        xaxis_title="seuil de refus",
        yaxis_title="coût métier moyen / dossier",
        height=400,
        margin={"t": 30},
    )
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    row = curve_ref.iloc[(curve_ref["seuil"] - threshold).abs().idxmin()]
    c1.metric("Coût au seuil courant (réf.)", f"{row['cout']:.3f}")
    c2.metric("Taux de refus au seuil courant", f"{row['taux_refus']:.1%}")
    c3.metric("Rappel au seuil courant", f"{row['rappel']:.1%}")
    st.dataframe(
        curve_ref.iloc[::5].style.format(
            {
                "seuil": "{:.2f}",
                "cout": "{:.4f}",
                "taux_refus": "{:.1%}",
                "rappel": "{:.1%}",
                "precision": "{:.1%}",
            }
        ),
        use_container_width=True,
        height=300,
    )

# ----------------------------------------------------------------------------- Technique
else:
    st.title("Suivi technique (depuis PostgreSQL)")
    st.caption(
        "Latences serveur enregistrées avec chaque prédiction. Pour le temps réel : Grafana (port 3000)."
    )
    lat = preds["latency_total_ms"].dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("p50 (ms)", f"{np.percentile(lat, 50):.2f}")
    c2.metric("p95 (ms)", f"{np.percentile(lat, 95):.2f}")
    c3.metric(
        "p99 (ms)", f"{np.percentile(lat, 99):.2f}", delta="objectif < 180 ms", delta_color="off"
    )
    errs = _errors(float(hours))
    c4.metric("Erreurs 4xx/5xx", len(errs))
    fig = px.histogram(
        preds, x="latency_total_ms", nbins=60, labels={"latency_total_ms": "latence serveur (ms)"}
    )
    fig.update_layout(height=320, margin={"t": 20})
    st.plotly_chart(fig, use_container_width=True)
    by_backend = preds.groupby("model_backend")["latency_inference_ms"].describe(
        percentiles=[0.5, 0.95, 0.99]
    )[["count", "50%", "95%", "99%", "max"]]
    st.subheader("Temps d'inférence par backend (ms)")
    st.dataframe(by_backend.style.format("{:.2f}"), use_container_width=True)
    if not errs.empty:
        st.subheader("Dernières erreurs")
        errs["detail"] = errs["detail"].apply(
            lambda d: "; ".join(
                f"{'.'.join(map(str, e.get('loc', [])))}: {e.get('msg')}" for e in (d or [])
            )
        )
        st.dataframe(
            errs[["ts", "status_code", "error_type", "scenario", "detail"]],
            use_container_width=True,
            height=300,
        )
