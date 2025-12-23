import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------- CONFIG ----------
API_URL = (st.secrets.get("API_URL") or os.getenv("API_URL") or "https://credit-scoring-api-8j5p.onrender.com")
CLIENTS_CSV = "../Sources/clients.csv"
DEFAULT_FEATURES_TO_SHOW = ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","PAYMENT_RATE","DAYS_EMPLOYED"]

st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")

# ---------- ACCESSIBILITÉ ----------
with st.sidebar:
    st.header("Réglages d’accessibilité")
    big_font = st.checkbox("Texte agrandi", value=True)
    high_contrast = st.checkbox("Couleurs haut contraste", value=True)
    show_value_labels = st.checkbox("Afficher les valeurs exactes", value=True)

FONT_SIZE = "18px" if big_font else "14px"
st.markdown(f"<style> * {{ font-size:{FONT_SIZE}; }} </style>", unsafe_allow_html=True)

# Palette colorblind-safe (bleu / orange / gris)
COLOR_POS = "#1f77b4"  # bleu
COLOR_NEG = "#ff7f0e"  # orange
COLOR_NEU = "#4d4d4d"  # gris sombre
if high_contrast:
    COLOR_POS, COLOR_NEG, COLOR_NEU = "#00429d", "#d1495b", "#111111"

# ---------- HELPERS ----------
@st.cache_data(show_spinner=False)
def load_clients():
    df = pd.read_csv(CLIENTS_CSV)
    return df

def api_health():
    r = requests.get(f"{API_URL}/health", timeout=10)
    r.raise_for_status()
    return r.json()

def api_predict(features: dict, threshold: float | None = None):
    payload = {"features": features}
    if threshold is not None:
        payload["threshold"] = float(threshold)
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def gauge_distance(prob, thr):
    # jauge simple : barre horizontale + repères
    pct_prob = min(max(prob, 0.0), 1.0)
    pct_thr = min(max(thr, 0.0), 1.0)
    cols = st.columns([8,1])
    with cols[0]:
        st.progress(pct_prob, text=f"Probabilité de défaut : {pct_prob:.3f} (seuil = {pct_thr:.3f})")
    with cols[1]:
        gap = prob - thr
        sign = "au-dessus" if gap >= 0 else "en dessous"
        st.markdown(f"**Δ = {gap:+.3f}**<br/><span style='color:{COLOR_NEU}'>({sign} du seuil)</span>", unsafe_allow_html=True)

def feature_hist(df, feature, client_value, segment_mask=None):
    # histogramme Comparaison client vs cohort/segment
    data = df[[feature]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if segment_mask is not None:
        data = data[segment_mask]
    base = alt.Chart(data).mark_bar().encode(
        alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=40), title=feature),
        alt.Y('count()', title="Effectif"),
        tooltip=[alt.Tooltip(f"{feature}:Q", title=feature), alt.Tooltip('count():Q', title="n")]
    )
    rule = alt.Chart(pd.DataFrame({"v":[client_value]})).mark_rule(color=COLOR_NEG, size=3).encode(x=f"v:Q")
    text = alt.Chart(pd.DataFrame({"v":[client_value]})).mark_text(
        text=f"{feature} client", dy=-8, color=COLOR_NEG
    ).encode(x="v:Q")
    chart = base + rule + text
    st.altair_chart(chart.properties(height=220), use_container_width=True)
    if show_value_labels:
        st.caption(f"Valeur client : **{client_value}**")

# ---------- DATA ----------
df_clients = load_clients()
health = api_health()
THRESHOLD = float(health.get("threshold", 0.5))
st.success(f"API OK · seuil métier = **{THRESHOLD:.3f}**")

# ---------- UI : sélection client ----------
st.header("Dashboard Scoring Crédit")
left, right = st.columns([1,2])

with left:
    st.subheader("Sélection client")
    id_col_candidates = [c for c in df_clients.columns if str(c).lower() in ("sk_id_curr","client_id","id")]
    id_col = id_col_candidates[0] if id_col_candidates else df_clients.columns[0]
    client_ids = df_clients[id_col].tolist()
    client_id = st.selectbox("Client", client_ids, index=0)
    row = df_clients[df_clients[id_col] == client_id].iloc[0]
    st.write("**Informations principales**")
    # Montre quelques features lisibles (adapte à ton dataset)
    info_cols = [c for c in DEFAULT_FEATURES_TO_SHOW if c in df_clients.columns]
    st.dataframe(pd.DataFrame(row[info_cols]).rename(columns={0:"valeur"}))

with right:
    st.subheader("Score & décision")
    # Prépare le payload features -> API
    features_payload = {c: (None if pd.isna(row[c]) else float(row[c])) for c in df_clients.columns if c != id_col and pd.api.types.is_numeric_dtype(df_clients[c])}
    res = api_predict(features_payload)     # seuil par défaut (artifacts.json de l’API)
    prob = float(res["probability"])
    yhat = int(res["predicted_class"])
    decision = res.get("decision", "Accepter" if yhat==0 else "Refuser")
    # Jauge distance au seuil
    gauge_distance(prob, THRESHOLD)
    # Cartouche décision
    bg = "#222" if high_contrast else "#2b2b2b"
    col = "#00b050" if yhat==0 else "#d1495b"
    st.markdown(
        f"<div style='padding:12px;border-radius:8px;background:{bg};color:white'>"
        f"<b>Décision</b> : <span style='color:{col}'>{decision}</span> "
        f"— p(defaut)={prob:.3f} ; seuil={THRESHOLD:.3f}"
        f"</div>", unsafe_allow_html=True
    )

st.markdown("---")

# ---------- COMPARAISON ----------
st.subheader("Comparaison client vs cohorte / segment")
cols = st.columns(3)
with cols[0]:
    feature_to_plot = st.selectbox("Variable à comparer", [c for c in DEFAULT_FEATURES_TO_SHOW if c in df_clients.columns])
with cols[1]:
    segment_col = st.selectbox("Filtre (segment)", ["(aucun)"] + [c for c in df_clients.columns if str(df_clients[c].dtype) in ("int64","float64","object")][:15])
with cols[2]:
    segment_val = None
    seg_mask = None
    if segment_col != "(aucun)":
        unique_vals = df_clients[segment_col].dropna().unique().tolist()[:50]
        segment_val = st.selectbox("Valeur de segment", unique_vals)
        seg_mask = df_clients[segment_col] == segment_val

client_val = row[feature_to_plot]
feature_hist(df_clients, feature_to_plot, client_val, segment_mask=seg_mask)

# ---------- WHAT-IF (optionnel) ----------
st.markdown("---")
st.subheader("What-if : modifier des caractéristiques et recalculer")
cols = st.columns(len(DEFAULT_FEATURES_TO_SHOW))
edited = {}
for i, f in enumerate(DEFAULT_FEATURES_TO_SHOW):
    if f in df_clients.columns:
        default = None if pd.isna(row[f]) else float(row[f] and pd.api.types.is_numeric_dtype(df_clients[f]))
        with cols[i]:
            edited[f] = st.number_input(f, value=default if default is not None else 0.0)

if st.button("Recalculer"):
    payload = features_payload.copy()
    payload.update({k: (None if v is None else float(v)) for k, v in edited.items()})
    res2 = api_predict(payload, threshold=THRESHOLD)
    prob2 = float(res2["probability"]); yhat2 = int(res2["predicted_class"])
    decision2 = res2.get("decision", "Accepter" if yhat2==0 else "Refuser")
    st.info(f"**Nouveau score** → p(defaut)={prob2:.3f} ; décision : **{decision2}** (seuil={THRESHOLD:.3f})")
