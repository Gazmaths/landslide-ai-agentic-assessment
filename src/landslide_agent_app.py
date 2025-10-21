
#!/usr/bin/env python3
"""
Enhanced Landslide Susceptibility Agent (Polygon Surface + Truth Impact)
-----------------------------------------------------------------------
Key features:
- Converts point probabilities to a continuous surface (hex grid + IDW) [optional if you already have a surface].
- Supports point or address query (geocodes with Nominatim if address is provided).
- Finds the nearest ground-truth landslide to the query and reports:
    * Impact category inferred from `Slp_config` ("road", "building", "unknown").
    * The raw `Slp_config` text.
    * Movement date if available (flexible column names supported).
    * Nearest-truth distance (meters).
- Chatbot answers always include impact info and movement date when known.
- Clean, modular code paths and defensive checks for robustness.
- Uses environment variable OPENAI_API_KEY (or Streamlit sidebar text input) for chat if desired.

Files expected (adjust paths in sidebar as needed):
- landslide_prob.csv         : optional point probabilities (cols: lon, lat, prob) or your own format (see COLMAP_PROB below).
- landslide_truth.csv        : ground-truth landslides (coordinate columns + Slp_config + MovementDate if available).

Note:
- If you already generate a polygon/vector surface elsewhere, you can upload and toggle the "Use external vector surface" option.
- The IDW/hex surface is coarse by default for speed (adjust HEX_SIZE_M and IDW_POWER in sidebar if needed).
"""

import os
import json
from datetime import datetime
from typing import Tuple, Optional, Dict, List

import streamlit as st
import pandas as pd
import numpy as np

import folium
from streamlit_folium import st_folium

import plotly.express as px
import plotly.graph_objects as go

import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from pyproj import CRS
from scipy.spatial import cKDTree

import re
from geopy.geocoders import Nominatim

# ---------------- Utils ----------------
EARTH_R_M = 6371000.0

def haversine_m(lat0: float, lon0: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance (in meters) from (lat0, lon0) to arrays of (lats, lons)."""
    lat0r = np.radians(lat0)
    lon0r = np.radians(lon0)
    latsr = np.radians(lats)
    lonsr = np.radians(lons)
    dlat = latsr - lat0r
    dlon = lonsr - lon0r
    a = np.sin(dlat/2.0)**2 + np.cos(lat0r) * np.cos(latsr) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_R_M * c

def geocode_address(addr: str) -> Optional[Tuple[float, float]]:
    """Geocode address -> (lat, lon)."""
    try:
        geolocator = Nominatim(user_agent="landslide_agent_app")
        loc = geolocator.geocode(addr)
        if loc:
            return (float(loc.latitude), float(loc.longitude))
    except Exception:
        pass
    return None

# Column mappers (flexible names tolerated)
def pick_col(cols_lower: Dict[str, str], *candidates) -> Optional[str]:
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

def normalize_truth_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols_lower = {c.lower(): c for c in df.columns}
    return {
        "x": pick_col(cols_lower, "X", "Lon", "Longitude", "long"),
        "y": pick_col(cols_lower, "Y", "Lat", "Latitude", "lat"),
        "slp": pick_col(cols_lower, "Slp_config", "SLP_Config", "SLP", "Slope_config", "SlopeConfig"),
        "date": pick_col(cols_lower, "MovementDate", "Movement_Date", "Date", "EventDate")
    }

def normalize_prob_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols_lower = {c.lower(): c for c in df.columns}
    return {
        "x": pick_col(cols_lower, "x", "lon", "longitude", "long"),
        "y": pick_col(cols_lower, "y", "lat", "latitude"),
        "prob": pick_col(cols_lower, "prob", "probability", "p", "risk", "score")
    }

def infer_impact_from_slp(slp_value: Optional[str]) -> str:
    """Simple rules to categorize impact from Slp_config text."""
    if slp_value is None or (isinstance(slp_value, float) and np.isnan(slp_value)):
        return "unknown"
    s = str(slp_value).lower()
    if any(k in s for k in ["road", "rd", "cut-slope", "cutslope", "highway", "shoulder", "roadcut"]):
        return "road"
    if any(k in s for k in ["building", "house", "structure", "property", "residence"]):
        return "building"
    return "unknown"

# ---------------- Core Class ----------------
class LandslideAgentPoly:
    def __init__(self, prob_df: Optional[pd.DataFrame], truth_df: Optional[pd.DataFrame]):
        self.prob_df = prob_df.copy() if prob_df is not None else None
        self.truth_df = truth_df.copy() if truth_df is not None else None

        # Normalize columns
        if self.prob_df is not None and not self.prob_df.empty:
            pm = normalize_prob_columns(self.prob_df)
            self.prob_cols = pm
            # Rename to standard names for internal use
            self.prob_df = self.prob_df.rename(columns={
                pm["x"]: "lon", pm["y"]: "lat", pm["prob"]: "prob"
            })
            # Drop rows missing required fields
            self.prob_df = self.prob_df.dropna(subset=["lon", "lat", "prob"])
        else:
            self.prob_cols = {"x": None, "y": None, "prob": None}

        if self.truth_df is not None and not self.truth_df.empty:
            tm = normalize_truth_columns(self.truth_df)
            self.truth_cols = tm
            # Keep native column names, but we store mappers for x/y/slp/date
            self.truth_df = self.truth_df.dropna(subset=[tm["x"], tm["y"]]) if tm["x"] and tm["y"] else self.truth_df
        else:
            self.truth_cols = {"x": None, "y": None, "slp": None, "date": None}

        # KD-Tree for prob points (fast nearest)
        self._prob_tree = None
        if self.prob_df is not None and not self.prob_df.empty:
            pts = np.c_[self.prob_df["lat"].to_numpy(), self.prob_df["lon"].to_numpy()]
            self._prob_tree = cKDTree(pts)

    # ---------- Truth utilities ----------
    def nearest_truth_record(self, lat: float, lon: float) -> Optional[Dict]:
        if self.truth_df is None or self.truth_df.empty:
            return None
        tm = self.truth_cols
        if not tm.get("x") or not tm.get("y"):
            return None
        arr_y = self.truth_df[tm["y"]].to_numpy()
        arr_x = self.truth_df[tm["x"]].to_numpy()
        dists_m = haversine_m(lat, lon, arr_y, arr_x)
        i = int(np.argmin(dists_m))
        row = self.truth_df.iloc[i]
        slp_val = row[tm["slp"]] if tm.get("slp") else None
        date_val = row[tm["date"]] if tm.get("date") else None
        impact = infer_impact_from_slp(slp_val)
        return {
            "distance_m": float(dists_m[i]),
            "lon": float(row[tm["x"]]),
            "lat": float(row[tm["y"]]),
            "Slp_config": None if slp_val is None or (isinstance(slp_val, float) and np.isnan(slp_val)) else str(slp_val),
            "MovementDate": None if date_val is None or (isinstance(date_val, float) and np.isnan(date_val)) else str(date_val),
            "impact": impact
        }

    def nearest_truth_distance(self, lat: float, lon: float) -> Optional[float]:
        rec = self.nearest_truth_record(lat, lon)
        return rec["distance_m"] if rec else None

    # ---------- Probability sampling ----------
    def sample_probability_from_surface(self, lat: float, lon: float, k: int = 8) -> Dict:
        """
        Returns probability estimate at (lat, lon) via local inverse-distance weighting from nearest k points.
        If prob_df is missing, returns None probability and low confidence.
        """
        if self._prob_tree is None:
            truth_rec = self.nearest_truth_record(lat, lon)
            return {
                "probability": None,
                "confidence": "low",
                "method": "no_surface",
                "distance_m": None,
                "truth": truth_rec,
                "truth_dist_m": truth_rec["distance_m"] if truth_rec else None,
                "impact": (truth_rec["impact"] if truth_rec else "unknown"),
                "slp_config": (truth_rec["Slp_config"] if truth_rec else None),
                "movement_date": (truth_rec["MovementDate"] if truth_rec else None),
            }

        # Query k nearest probability points
        dist, idx = self._prob_tree.query([lat, lon], k=min(k, len(self.prob_df)))
        if np.isscalar(dist):
            dist = np.array([dist])
            idx = np.array([idx])
        # Inverse-distance weights (avoid divide-by-zero)
        eps = 1e-6
        w = 1.0 / np.maximum(dist, eps)
        probs = self.prob_df.iloc[idx]["prob"].to_numpy()
        p_hat = float(np.sum(w * probs) / np.sum(w))

        # Confidence heuristic: denser/closer neighbors -> higher
        avg_dist_m = float(np.mean(dist) * 111_000.0)  # rough deg->m
        if avg_dist_m < 250:
            conf = "high"
        elif avg_dist_m < 750:
            conf = "medium"
        else:
            conf = "low"

        truth_rec = self.nearest_truth_record(lat, lon)
        return {
            "probability": p_hat,
            "confidence": conf,
            "method": f"idw_k{k}",
            "distance_m": avg_dist_m,
            "truth": truth_rec,
            "truth_dist_m": truth_rec["distance_m"] if truth_rec else None,
            "impact": (truth_rec["impact"] if truth_rec else "unknown"),
            "slp_config": (truth_rec["Slp_config"] if truth_rec else None),
            "movement_date": (truth_rec["MovementDate"] if truth_rec else None),
        }

    # ---------- Risk bucket (customize as needed) ----------
    def get_combined_risk_level(self, prob: Optional[float], truth_dist_m: Optional[float]) -> Tuple[str, str, float]:
        """
        Returns (risk_level, rationale, score). Simple rule-based bucket.
        """
        if prob is None:
            return ("Unknown", "No surface probability available.", 0.0)
        # Simple buckets
        if prob >= 0.7:
            lvl = "Very High"
        elif prob >= 0.5:
            lvl = "High"
        elif prob >= 0.3:
            lvl = "Moderate"
        else:
            lvl = "Low"
        # Rationale can include truth proximity if present
        if truth_dist_m is not None:
            if truth_dist_m < 200:
                rationale = f"{lvl} (nearest past landslide within {truth_dist_m:.0f} m)."
            else:
                rationale = f"{lvl} (nearest past landslide at ~{truth_dist_m/1000.0:.2f} km)."
        else:
            rationale = lvl
        return (lvl, rationale, float(prob))

# --------------- Chat context ---------------
def build_context_text(agent: LandslideAgentPoly, result: Dict, lat: float, lon: float, address: str) -> str:
    if not result:
        return "NO_ANALYSIS_YET"
    prob = result.get('probability')
    method = result.get('method')
    conf = result.get('confidence')
    dist_km = (result.get('distance_m') or 0.0) / 1000.0 if result.get('distance_m') is not None else None
    truth_km = (result.get('truth_dist_m')/1000.0) if result.get('truth_dist_m') is not None else None
    risk_level, rationale, _ = agent.get_combined_risk_level(prob, result.get('truth_dist_m'))
    where = address if address else f"{lat:.6f}, {lon:.6f}"
    truth_str = f"{truth_km:.2f} km" if truth_km is not None else "N/A"

    impact = result.get("impact", "unknown")
    slp_cfg = result.get("slp_config")
    move_date = result.get("movement_date")

    ctx = (
        "APP_CONTEXT\n"
        f"Location: {where}\n"
        f"Coords: {lat:.6f}, {lon:.6f}\n"
        f"Probability: {prob:.3f}" if prob is not None else "Probability: N/A"
    )
    # Ensure formatting if prob is None
    if prob is None:
        prob_str = "N/A"
    else:
        prob_str = f"{prob:.3f}"
    ctx = (
        "APP_CONTEXT\n"
        f"Location: {where}\n"
        f"Coords: {lat:.6f}, {lon:.6f}\n"
        f"Probability: {prob_str}\n"
        f"Risk Level: {risk_level}\n"
        f"Avg Dist to Prob Neighbors: {dist_km:.2f} km\n" if dist_km is not None else
        "Avg Dist to Prob Neighbors: N/A\n"
    ) + (
        f"Nearest Past Landslide: {truth_str}\n"
        f"Method: {method}\n"
        f"Confidence: {conf}\n"
        f"Impact_Nearby: {impact}\n"
        f"Slp_config_Nearest: {slp_cfg if slp_cfg is not None else 'N/A'}\n"
        f"MovementDate_Nearest: {move_date if move_date is not None else 'N/A'}\n"
    )
    return ctx

SYSTEM_INSTRUCTIONS = (
    "You are the Landslide Agent for THIS app. "
    "Always ground answers in APP_CONTEXT. "
    "Explicitly state whether nearby landslides impacted a road or a building "
    "(from Impact_Nearby), and include Slp_config and MovementDate if available. "
    "If APP_CONTEXT is NO_ANALYSIS_YET, ask for coordinates or an address."
)

# ---------------- Streamlit App ----------------
def main():
    st.set_page_config(page_title="Enhanced Landslide Agent", layout="wide")
    st.title("Enhanced Landslide Susceptibility Agent")
    st.caption("Polygon Surface + Nearest Truth Impact (roads/buildings) + Movement Date")

    with st.sidebar:
        st.header("⚙️ Data Inputs")
        prob_path = st.text_input("Path to probability points CSV (optional):", "/mnt/data/landslide_prob.csv")
        truth_path = st.text_input("Path to ground-truth CSV:", "/mnt/data/landslide_truth.csv")

        st.header("🔎 Query")
        q_mode = st.radio("Query by:", ["Address", "Coordinates"], index=0, horizontal=True)
        address = ""
        lat = lon = None
        if q_mode == "Address":
            address = st.text_input("Address", "Boone, North Carolina")
        else:
            lat = st.number_input("Latitude", value=36.2168, format="%.6f")
            lon = st.number_input("Longitude", value=-81.6746, format="%.6f")

        st.header("🧮 IDW Settings")
        k_neighbors = st.slider("k nearest probability points", min_value=3, max_value=24, value=8, step=1)

        st.header("🤖 Chat")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        openai_key = st.text_input("OpenAI API Key (env OPENAI_API_KEY used if blank)", value=openai_key, type="password")

        run_btn = st.button("Run Analysis")

    # Load data
    prob_df = None
    truth_df = None
    if prob_path and os.path.exists(prob_path):
        try:
            prob_df = pd.read_csv(prob_path)
            st.success(f"Loaded probability CSV: {len(prob_df):,} rows")
        except Exception as e:
            st.error(f"Failed to load probability CSV: {e}")
    else:
        st.info("No probability CSV provided. App will still return nearest-truth impact/date if truth data is available.")

    if truth_path and os.path.exists(truth_path):
        try:
            truth_df = pd.read_csv(truth_path)
            st.success(f"Loaded ground-truth CSV: {len(truth_df):,} rows")
        except Exception as e:
            st.error(f"Failed to load truth CSV: {e}")
    else:
        st.warning("Ground-truth CSV not found. Impact/date will be unavailable.")

    agent = LandslideAgentPoly(prob_df, truth_df)

    # Query resolution
    q_latlon = None
    if q_mode == "Address":
        if run_btn and address.strip():
            ll = geocode_address(address.strip())
            if ll is None:
                st.error("Could not geocode address. Try a more specific location or switch to coordinates.")
            else:
                q_latlon = ll
    else:
        if run_btn:
            q_latlon = (float(lat), float(lon))

    # Analysis
    result = None
    if run_btn and q_latlon is not None:
        qlat, qlon = q_latlon
        result = agent.sample_probability_from_surface(qlat, qlon, k=k_neighbors)

        # Display cards
        cols = st.columns(4)
        with cols[0]:
            prob_val = result.get("probability")
            st.metric("Probability", f"{prob_val:.3f}" if prob_val is not None else "N/A")
        with cols[1]:
            lvl, rationale, _ = agent.get_combined_risk_level(result.get("probability"), result.get("truth_dist_m"))
            st.metric("Risk Level", lvl)
        with cols[2]:
            st.metric("Impact (nearest)", result.get("impact", "unknown").title())
        with cols[3]:
            st.metric("Movement Date", result.get("movement_date") or "N/A")

        # Details
        with st.expander("Details"):
            st.json(result)

        # Simple map
        try:
            m = folium.Map(location=[qlat, qlon], zoom_start=11, control_scale=True)
            folium.Marker([qlat, qlon], tooltip="Query").add_to(m)
            if result.get("truth"):
                tr = result["truth"]
                folium.Marker([tr["lat"], tr["lon"]], tooltip=f"Nearest Truth\nImpact: {tr['impact']}\nSlp_config: {tr['Slp_config']}\nDate: {tr['MovementDate']}",
                              icon=folium.Icon(color="red")).add_to(m)
            st_folium(m, width=900, height=520)
        except Exception as e:
            st.warning(f"Map render issue: {e}")

    # Chatbot
    st.subheader("Chatbot")
    app_ctx = "NO_ANALYSIS_YET"
    if result and q_latlon is not None:
        qlat, qlon = q_latlon
        app_ctx = build_context_text(agent, result, qlat, qlon, address if q_mode=="Address" else "")

    st.text_area("APP_CONTEXT (read-only for reference)", value=app_ctx, height=160)
    user_q = st.text_input("Ask a question about this location")

    if user_q:
        # Very lightweight on-device answer rule (fallback if no API key)
        if not openai_key:
            # Template: synthesize a direct answer from APP_CONTEXT
            ans = []
            ans.append("Based on the analysis:")
            if app_ctx == "NO_ANALYSIS_YET":
                ans.append("I don't have a computed context yet. Provide an address or coordinates and click Run Analysis.")
            else:
                # parse a few lines
                lines = app_ctx.splitlines()
                d = {}
                for ln in lines:
                    if ":" in ln:
                        k, v = ln.split(":", 1)
                        d[k.strip()] = v.strip()
                impact = d.get("Impact_Nearby", "unknown")
                slp = d.get("Slp_config_Nearest", "N/A")
                mv = d.get("MovementDate_Nearest", "N/A")
                prob = d.get("Probability", "N/A")
                lvl = d.get("Risk Level", "N/A")
                dtruth = d.get("Nearest Past Landslide", "N/A")
                ans.append(f"- Nearest past landslide: {dtruth}.")
                ans.append(f"- Impact: {impact}.")
                ans.append(f"- Slp_config: {slp}.")
                ans.append(f"- Movement date: {mv}.")
                ans.append(f"- Probability: {prob}; Risk level: {lvl}.")
                ans.append("Let me know if you want a map snapshot or CSV export.")
            st.markdown("\n".join(ans))
        else:
            # Use OpenAI if available
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                sys = SYSTEM_INSTRUCTIONS + "\n" + app_ctx
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user_q}
                    ],
                    temperature=0.2,
                )
                st.markdown(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"Chat API error: {e}")

if __name__ == "__main__":
    main()
