import os
import json
import time
from typing import List, Dict, Optional

import requests
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# Configuration
st.set_page_config(
    page_title="Credit Scoring – Dashboard",
    page_icon="📊",
    layout="wide",
)

# Meilleur theme pour plus de lisibilité
alt.theme.enable("opaque")

# API URL
API_URL = (
    st.secrets.get("API_URL") 
    or os.getenv("API_URL")
    or "https://credit-scoring-api-8j5p.onrender.com"
    )

# Helpers
@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_metadata() -> Dict:
    r = requests.get(f"{API_URL}/metadata", timeout=15)
    r.raise_for_status()
    return r.json()

def api_predict(payload: Dict) -> Dict:
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    if r.status_code != 200:
        # montre l’erreur renvoyée par l’API
        try:
            detail = r.json().get("detail", "")
        except Exception:
            detail = r.text
        raise RuntimeError(f"API /predict a échoué ({r.status_code}) : {detail}")
    return r.json()

def api_explain(payload: Dict) -> Dict:
    r = requests.post(f"{API_URL}/explain", json={"features": payload, "top_k": 10}, timeout=20)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", "")
        except Exception:
            detail = r.text
        raise RuntimeError(f"API /explain a échoué ({r.status_code}) : {detail}")
    return r.json()

def ratio_to_threshold(prob: float, thr: float) -> str:
    gap = prob - thr
    if gap >= 0:
        return f"+{gap:.3f} au-dessus du seuil"
    return f"{gap:.3f} en-dessous du seuil"

def describe_feature_value(df: pd.DataFrame, col: str, value: float) -> str:
    if col not in df.columns:
        return "n/a"
    s = pd.to_numeric(df[col], errors="coerce")
    q50 = s.quantile(0.5)
    q25 = s.quantile(0.25)
    q75 = s.quantile(0.75)
    pos = "≈ médiane"
    if value < q25:
        pos = "< Q1 (25e)"
    elif value > q75:
        pos = "> Q3 (75e)"
    return f"{value:.4g} ({pos}, médiane={q50:.4g})"

def color_for_decision(decision_text: str) -> str:
    return "✅" if "Accepter" in decision_text else "❌"

# Sidebar / Sources / Parameters
st.sidebar.header("⚙️ Paramètres")
st.sidebar.caption("Configurer la source des données et l’API.")

uploaded = st.sidebar.file_uploader("Charger un CSV de clients", type=["csv"])

try:
    meta = fetch_metadata()
    EXPECTED_FEATURES: List[str] = meta.get("expected_features", [])
    BEST_THR: float = float(meta.get("threshold", 0.5))
    MODEL_VERSION = meta.get("model_version", "v1")
except Exception as e:
    st.error(f"Impossible de joindre l’API ({API_URL}). Détail: {e}")
    st.stop()

# Charge les données
df = load_data(uploaded)

if df.empty:
    st.warning("Aucun CSV chargé. Charger un dataset (avec `SK_ID_CURR`).")
else:
    if "SK_ID_CURR" not in df.columns:
        st.error("Le CSV doit contenir la colonne `SK_ID_CURR`.")
        st.stop()

# Header
st.title("📊 Dashboard – Credit Scoring")
st.write(
    f"Modèle **{MODEL_VERSION}**, **seuil métier = {BEST_THR:.3f}**. "
    f"API : `{API_URL}`"
)

# Colonne gauche / droite
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1) Sélection du client")
    try:
        client_ids = df["SK_ID_CURR"].astype(str).tolist()
        client_id = st.selectbox("Choisir un SK_ID_CURR", client_ids, index=0 if client_ids else None)
        sample = df[df["SK_ID_CURR"].astype(str) == client_id].head(1)
        if sample.empty:
            st.error("Client introuvable dans le CSV.")
            st.stop()
    except Exception as e:
        st.error(f"Erreur lors de la sélection / chargement du client : {e}")
        st.stop()

    
    features_row = sample.iloc[0].to_dict()
    # Retire l’ID et tout ce qui n’est pas attendu
    features_payload = {}
    for c in EXPECTED_FEATURES:
        val = features_row.get(c, np.nan)
        try:
            val = float(val)
        except Exception:
            val = np.nan
        
        if pd.isna(val) or np.isinf(val):
            features_payload[c] = None
        else:
            features_payload[c] = float(val)

    with st.spinner("Appel de l’API /predict…"):
        try:
            resp = api_predict({"features": features_payload})
        except Exception as e:
            st.error(str(e))
            st.stop()

    prob = float(resp["probability"])
    decision = resp["decision"]
    predicted_class = resp["predicted_class"]

    # Score
    st.subheader("2) Score & Décision")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.metric(
            label="Probabilité de défaut",
            value=f"{prob:.3%}",
            delta=ratio_to_threshold(prob, BEST_THR),
            help="Probabilité d'être classé 'mauvais payeur' selon le modèle."
        )
    with c2:
        st.metric(
            label="Seuil métier",
            value=f"{BEST_THR:.3%}",
            help="Au-dessus du seuil → refus ; en-dessous → acceptation."
        )
    with c3:
        st.metric(
            label="Décision",
            value=f"{color_for_decision(decision)} {decision}",
            help="Interprétation lisible du résultat."
        )

    # Jauge simple barre horizontale
    st.caption("Écart vs seuil (plus lisible qu’une seule couleur).")
    maxw = 1.0
    bar_df = pd.DataFrame({"x":[0, BEST_THR, prob], "label": ["0", "seuil", "proba"]})
    chart = alt.Chart(pd.DataFrame({"x":[0,1]})).mark_bar().encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0,1])), y=alt.value(16)
    ).properties(height=16)

    rule_thr = alt.Chart(pd.DataFrame({"x":[BEST_THR]})).mark_rule(size=3).encode(x="x:Q")
    rule_prob = alt.Chart(pd.DataFrame({"x":[prob]})).mark_rule(size=3, strokeDash=[4,2]).encode(x="x:Q")
    st.altair_chart(chart + rule_thr + rule_prob, use_container_width=True)


with col_right:
    st.subheader("3) Explication du score (local)")

    with st.expander("Explication du score (SHAP local)", expanded=True):
        try:
            exp = api_explain(features_payload)
            contribs = pd.DataFrame(exp["contribution"])

            chart = alt.Chart(contribs).mark_bar().encode(
                x = alt.X("abs_shap:Q", title="Importance locale (|SHAP| sur probabilité)"),
                y = alt.Y("feature:N", sort="-x", title=None),
                tooltip = ["feature", alt.Tooltip("value:Q", format=".4g"),
                        alt.Tooltip("shap:Q", format=".4g"), alt.Tooltip("abs_shap:Q", format=".4g")]
            ).properties(height=24*len(contribs), title="Top contributions locales")
            st.altair_chart(chart, use_container_width=True)

            st.caption(
                f"Base value (attendu moyen) : **{exp['base_value']:.3f}** - "
                f"Prédiction : **{exp['prediction']:.3f}** (Δ ≈ somme des SHAP)."
            )
        except Exception as e:
            st.error(f"Erreur lors de l’appel à l’API /explain : {e}")


    K = st.slider("Nombre de variables à afficher", 3, 15, 7, help="Top K features les plus atypiques selon un z-score simple.")
    # calcul d'un zscore sur les features numériques pour trouver ce qui 'déborde'
    
    numeric_cols = [c for c in EXPECTED_FEATURES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    z = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        mu = s.mean()
        sd = s.std(ddof=0) or 1.0
        try:
            val = float(sample[c].iloc[0])
        except Exception:
            val = np.nan
        if np.isnan(val):
            continue
        z[c] = abs((val - mu) / sd)

    top_items = sorted(z.items(), key=lambda t: t[1], reverse=True)[:K]
    if not top_items:
        st.info("Pas assez de colonnes numériques ou valeurs indisponibles pour calculer une explication légère.")
    else:
        expl_df = pd.DataFrame(top_items, columns=["Feature", "z_abs"]).sort_values("z_abs", ascending=True)
        base = alt.Chart(expl_df).mark_bar().encode(
            x=alt.X("z_abs:Q", title="Écart standardisé (|z|)"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=["Feature", alt.Tooltip("z_abs:Q", format=".2f")]
        ).properties(height=24*len(expl_df), title="Variables les plus atypiques (approx. explication locale)")
        st.altair_chart(base, use_container_width=True)

# Informations client
st.subheader("4) Informations descriptives du client")
cols_show = st.multiselect(
    "Variables à afficher dans le panneau (recherche possible)",
    options=[c for c in df.columns if c != "TARGET"],
    default=["SK_ID_CURR", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED"][: min(5, len(df.columns))]
)
if cols_show:
    pretty = sample[cols_show].T.reset_index()
    pretty.columns = ["Variable", "Valeur"]
    st.dataframe(pretty, use_container_width=True)

# Comparaisons aux pairs
st.subheader("5) Comparaison client vs population / groupe similaire")

var = st.selectbox(
    "Choisir une variable numérique",
    [c for c in df.columns if c != "TARGET" and pd.api.types.is_numeric_dtype(df[c])],
    help="Histogramme/boîte à moustaches + valeur du client."
)

similar_filters = st.multiselect(
    "Filtrer un groupe 'similaire' (optionnel) – colonnes catégorielles",
    options=[c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])],
    help="Sélectionne 1 à N colonnes catégorielles, puis choisis leur modalité ci-dessous."
)

group_df = df.copy()
if similar_filters:
    st.caption("Choisis la modalité pour chaque filtre sélectionné :")
    for cat_col in similar_filters:
        choices = ["[Tous]"] + sorted(df[cat_col].astype(str).unique().tolist())
        val = st.selectbox(f"{cat_col}", choices, index=0)
        if val != "[Tous]":
            group_df = group_df[group_df[cat_col].astype(str) == val]

# Histogramme population/groupe + ligne verticale client
if var:
    base_df = pd.DataFrame({"value": pd.to_numeric(group_df[var], errors="coerce")}).dropna()
    client_val = pd.to_numeric(sample[var], errors="coerce").values[0]

    # histogramme accessible
    hist = alt.Chart(base_df).mark_bar().encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=40), title=var),
        y=alt.Y("count()", title="Effectif"),
        tooltip=[alt.Tooltip("count()", title="Effectif")],
    ).properties(height=260, title=f"Distribution de {var} (groupe sélectionné)")

    client_rule = alt.Chart(pd.DataFrame({"value":[client_val]})).mark_rule(size=3, strokeDash=[4,2]).encode(
        x="value:Q"
    )

    st.altair_chart(hist + client_rule, use_container_width=True)
    st.caption(f"Valeur client : {describe_feature_value(df, var, client_val)}")

# Footer
st.divider()
st.caption("Dashboard Streamlit pour le projet 8 – Credit Scoring – OpenClassrooms")