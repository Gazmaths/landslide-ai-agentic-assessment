#!/usr/bin/env python3
import os
import streamlit as st
import pandas as pd

from src.settings import settings
from src.data_io import load_csv_or_none

# Please go ahead and use your existing class for now; you can then migrate to src/ later at your own pace.
try:
    from landslide_agent_app import LandslideAgentPoly, build_context_text
except Exception as e:
    st.error(f"Couldn't import from landslide_agent_app.py. Make sure it is in the same folder. Error: {e}")
    st.stop()

st.set_page_config(page_title="Landslide Agent (Modular Starter)", layout="wide")
st.title("Landslide Susceptibility Agent — Modular Starter")
st.caption("This entrypoint keeps UI short and imports your core logic from the src/ package or existing file.")

with st.sidebar:
    st.header("🔑 Keys & Settings")
    # Prefer env var / secrets; give users a password field as a fallback (not persisted).
    # DO NOT store the key in code or commit it.
    default_key = settings.openai_api_key or st.secrets.get("OPENAI_API_KEY", "")
    openai_key = st.text_input("OpenAI API Key (optional)", value=default_key, type="password")

    st.header("📂 Data")
    prob_path = st.text_input("Probability CSV", "./landslide_prob.csv")
    truth_path = st.text_input("Ground-truth CSV", "./landslide_truth.csv")

    st.header("🔎 Query")
    mode = st.radio("Query by", ["Address", "Coordinates"], horizontal=True)
    address = st.text_input("Address", "Boone, North Carolina") if mode == "Address" else ""
    lat = st.number_input("Latitude", value=36.2168, format="%.6f") if mode == "Coordinates" else None
    lon = st.number_input("Longitude", value=-81.6746, format="%.6f") if mode == "Coordinates" else None

    k_neighbors = st.slider("IDW k-nearest points", 3, 24, 8, 1)

    run = st.button("Run Analysis")

# Apply key for downstream use (your landslide_agent_app reads env or input)
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

prob_df = load_csv_or_none(prob_path)
truth_df = load_csv_or_none(truth_path)

if prob_df is not None:
    st.success(f"Loaded prob CSV: {len(prob_df):,} rows")
else:
    st.info("No probability CSV found. You'll still get nearest-truth info if truth data is present.")

if truth_df is not None:
    st.success(f"Loaded truth CSV: {len(truth_df):,} rows")
else:
    st.warning("Ground-truth CSV not found. Impact/date will be unavailable.")

agent = LandslideAgentPoly(prob_df, truth_df)

coords = None
if run:
    if mode == "Address":
        from landslide_agent_app import geocode_address
        coords = geocode_address(address) if address.strip() else None
        if not coords:
            st.error("Couldn't geocode address. Try a more specific location or switch to coordinates.")
    else:
        coords = (float(lat), float(lon))

result = None
if run and coords:
    qlat, qlon = coords
    result = agent.sample_probability_from_surface(qlat, qlon, k=k_neighbors)

    cols = st.columns(4)
    with cols[0]:
        pv = result.get("probability")
        st.metric("Probability", f"{pv:.3f}" if pv is not None else "N/A")
    with cols[1]:
        lvl, rationale, _ = agent.get_combined_risk_level(result.get("probability"), result.get("truth_dist_m"))
        st.metric("Risk Level", lvl)
    with cols[2]:
        st.metric("Impact (nearest)", result.get("impact", "unknown").title())
    with cols[3]:
        st.metric("Movement Date", result.get("movement_date") or "N/A")

    with st.expander("Details"):
        st.json(result)

st.subheader("Chatbot (Context Preview)")
ctx = "NO_ANALYSIS_YET"
if result and coords:
    ctx = build_context_text(agent, result, coords[0], coords[1], address if mode=="Address" else "")
st.text_area("APP_CONTEXT (read-only)", value=ctx, height=160)

st.markdown("> Tip: Deploy safely with `st.secrets['OPENAI_API_KEY']` or an env variable. Never hardcode keys.")
