### App Version with Account Rep and Sales Territory Filters and To Kill For Quadrant rankings

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="GB Off-Trade Outlet Scores (MVP)", layout="wide")

# --- file paths ---
CSV_PATH = r"uk_outlet_scores_weekly_synthetic.csv"
TERR_PATH = r"sales_territories_synthetic_v2.csv"

NUMERIC_COLS = [
    "sales_value_gbp","nine_l_cases_total","sales_growth_4w","performance_index",
    "avg_on_hand_units","oos_rate","sku_range_onhand","cases_ordered_total","order_value_gbp",
    "recency_weeks","order_freq_4w","promo_flag","promo_types_count","avg_discount_pct","promo_uplift_pct",
    "footfall_count","population_1km","median_income_gbp","competitor_outlets_1km","nearby_venues_score",
    "alcohol_index","region_growth_index","avg_rating","review_count","recent_sentiment",
    "expected_value_gbp","headroom_ratio","range_gap_ratio","demand_index","external_index",
    "performance_score","potential_score","unified_commercial_score"
]

@st.cache_data
def load_weekly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week_start_date"])
    # Coerce numerics safely (bad/missing values -> NaN)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Light clean
    for c in ["channel","region","format_size","OutletID"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    # Scores presence hint
    missing = [c for c in ["performance_score","potential_score","unified_commercial_score"] if c not in df.columns]
    if missing:
        st.warning(f"Missing score columns: {missing}. The app will still load, but some views will be empty.")
    return df

@st.cache_data
def load_territories(path: str) -> pd.DataFrame:
    """
    Expect raw columns: outlet_id, sales_territory, account_rep
    But handle common variants too (OutletID/OutletCode, Territory/Rep).
    Standardise to: OutletID, sales_territory, account_rep
    """
    try:
        t = pd.read_csv(path)
    except Exception as e:
        st.warning(f"Territory file not found or unreadable: {e}. Proceeding without territories.")
        return pd.DataFrame(columns=["OutletID","sales_territory","account_rep"])

    # normalise headers
    cols_norm = {c: c.strip() for c in t.columns}
    t = t.rename(columns=cols_norm)

    # map likely variants -> standard names
    rename_map = {}
    # Outlet ID variants
    if "OutletID" not in t.columns:
        if "outlet_id" in t.columns: rename_map["outlet_id"] = "OutletID"
        elif "OutletCode" in t.columns: rename_map["OutletCode"] = "OutletID"
        elif "outletcode" in t.columns: rename_map["outletcode"] = "OutletID"
    # Territory variants
    if "sales_territory" not in t.columns:
        if "Territory" in t.columns: rename_map["Territory"] = "sales_territory"
        elif "territory" in t.columns: rename_map["territory"] = "sales_territory"
    # Rep variants
    if "account_rep" not in t.columns:
        if "Rep" in t.columns: rename_map["Rep"] = "account_rep"
        elif "rep" in t.columns: rename_map["rep"] = "account_rep"
        elif "sales_rep" in t.columns: rename_map["sales_rep"] = "account_rep"

    if rename_map:
        t = t.rename(columns=rename_map)

    # keep only the columns we need
    keep = [c for c in ["OutletID","sales_territory","account_rep"] if c in t.columns]
    t = t[keep].copy()

    # enforce dtypes
    for c in ["OutletID","sales_territory","account_rep"]:
        if c in t.columns:
            t[c] = t[c].astype(str)

    return t

# --- load data ---
df = load_weekly(CSV_PATH)
territories = load_territories(TERR_PATH)

# --- merge territories (left join on OutletID) ---
if not territories.empty and "OutletID" in df.columns:
    df["OutletID"] = df["OutletID"].astype(str)
    territories["OutletID"] = territories["OutletID"].astype(str)
    df = df.merge(territories, on="OutletID", how="left")
else:
    # ensure columns exist for downstream filters (even if empty)
    if "sales_territory" not in df.columns:
        df["sales_territory"] = pd.Series(dtype=str)
    if "account_rep" not in df.columns:
        df["account_rep"] = pd.Series(dtype=str)

st.title("🏪 GB Off-Trade Outlet Scores — MVP")
st.caption("Weekly leaderboard, outlet prioritisation matrix, and outlet drilldown. Now with Territory & Account Rep filters.")

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")

# Week filter (default to most recent)
weeks = sorted(df["week_start_date"].dt.date.unique())
default_week = weeks[-1] if len(weeks) else None
selected_weeks = st.sidebar.multiselect("Week start", weeks, default=[default_week] if default_week else [])

regions  = sorted(df["region"].dropna().unique())          if "region" in df.columns else []
channels = sorted(df["channel"].dropna().unique())         if "channel" in df.columns else []
formats  = sorted(df["format_size"].dropna().unique())     if "format_size" in df.columns else []
terrs    = sorted(df["sales_territory"].dropna().unique()) if "sales_territory" in df.columns else []
reps     = sorted(df["account_rep"].dropna().unique())     if "account_rep" in df.columns else []

sel_regions  = st.sidebar.multiselect("Region", regions, default=regions)
sel_channels = st.sidebar.multiselect("Outlet Type", channels, default=channels)
sel_formats  = st.sidebar.multiselect("Store Size", formats, default=formats)

sel_territory = st.sidebar.selectbox("Sales Territory", ["All"] + terrs, index=0) if terrs else "All"
sel_rep       = st.sidebar.selectbox("Sales Rep", ["All"] + reps, index=0)      if reps else "All"

min_unified = st.sidebar.slider("Min Unified Score", 0, 100, 0, 1)

# ---------------- Apply filters ----------------
f = df.copy()
if selected_weeks:
    f = f[f["week_start_date"].dt.date.isin(selected_weeks)]
if sel_regions:
    f = f[f["region"].isin(sel_regions)]
if sel_channels:
    f = f[f["channel"].isin(sel_channels)]
if sel_formats:
    f = f[f["format_size"].isin(sel_formats)]
if sel_territory != "All" and "sales_territory" in f.columns:
    f = f[f["sales_territory"] == sel_territory]
if sel_rep != "All" and "account_rep" in f.columns:
    f = f[f["account_rep"] == sel_rep]
if "unified_commercial_score" in f.columns:
    f = f[f["unified_commercial_score"] >= min_unified]

# ---------------- KPIs ----------------
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Outlets", f["OutletID"].nunique() if "OutletID" in f.columns else 0)
with c2: st.metric("Avg Unified", round(f["unified_commercial_score"].mean(),1) if "unified_commercial_score" in f.columns and len(f) else 0)
with c3: st.metric("Avg Performance", round(f["performance_score"].mean(),1) if "performance_score" in f.columns and len(f) else 0)
with c4: st.metric("Avg Potential", round(f["potential_score"].mean(),1) if "potential_score" in f.columns and len(f) else 0)

st.divider()

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard","🎯 Outlet Prioritisation Matrix","🟰 Outlet Drilldown"])

with tab1:
    st.subheader("Leaderboard (by Unified Score)")
    top_n = st.slider("Show top N", 10, 200, 50, 10)
    cols = [
        "week_start_date","OutletID","region","channel","format_size",
        "sales_territory","account_rep",
        "performance_score","potential_score","unified_commercial_score",
        "sales_value_gbp","nine_l_cases_total","sales_growth_4w","oos_rate","promo_uplift_pct"
    ]
    if len(f):
        show = f[[c for c in cols if c in f.columns]].sort_values("unified_commercial_score", ascending=False).head(top_n)
        st.dataframe(show, use_container_width=True, hide_index=True)

        # Route export (latest week, top N per rep)
        latest_week = f["week_start_date"].max()
        latest = f[f["week_start_date"] == latest_week].copy()
        n_per_rep = st.slider("Top N per rep (latest week)", 5, 100, 25, 5)

        if "account_rep" in latest.columns and latest["account_rep"].notna().any():
            latest_for_routing = latest if sel_rep == "All" else latest[latest["account_rep"] == sel_rep]
            route = (latest_for_routing
                     .sort_values("unified_commercial_score", ascending=False)
                     .groupby("account_rep", group_keys=False)
                     .head(n_per_rep))

            route_cols = [
                "week_start_date","account_rep","sales_territory","OutletID","region","channel","format_size",
                "unified_commercial_score","performance_score","potential_score",
                "sales_value_gbp","nine_l_cases_total","oos_rate","promo_flag","promo_uplift_pct",
                "sku_range_onhand","order_freq_4w","recency_weeks"
            ]
            route = route[[c for c in route_cols if c in route.columns]]

            st.download_button(
                "Download route plan (CSV, latest week)",
                data=route.to_csv(index=False),
                file_name=("route_plan_" + (sel_rep if sel_rep!="All" else "all_reps") + ".csv").replace(" ", "_"),
                mime="text/csv"
            )
    else:
        st.info("No rows match current filters.")

with tab2:
    st.subheader("Outlet Prioritisation Matrix (Potential vs Performance)")
    needed = {"performance_score", "potential_score"}
    if needed.issubset(f.columns) and len(f):

        # Ensure numeric, clipped to [0, 100]
        f["_perf"] = pd.to_numeric(f["performance_score"], errors="coerce").clip(0, 100)
        f["_pot"]  = pd.to_numeric(f["potential_score"], errors="coerce").clip(0, 100)

        # ---- Tier rules (non-overlapping; order matters) ----
        conditions = [
            (f["_perf"] >= 70) & (f["_pot"] >= 70),                             # To Kill For
            (f["_perf"].between(50, 69.9999)) & (f["_pot"] >= 70),              # Grow Upper (band B)
            (f["_perf"].between(25, 49.9999)) & (f["_pot"] >= 50),              # Grow Upper (band A)
            (f["_perf"] >= 50) & (f["_pot"].between(50, 69.9999)),              # Grow Lower (big square)
            (f["_perf"] >= 50) & (f["_pot"] < 50),                              # Maintain
            (f["_perf"] < 25)  & (f["_pot"] >= 50),                             # Nurture
            (f["_perf"] < 50)  & (f["_pot"] < 50),                              # Minimise
        ]
        choices = [
            "To Kill For", "Grow Upper", "Grow Upper", "Grow Lower",
            "Maintain", "Nurture", "Minimise",
        ]
        f["_tier"] = np.select(conditions, choices, default="Other").astype(str)

        # ---- Background rectangles (explicit colors) ----
        boxes = pd.DataFrame([
            [  0,  50,   0,  50, "Minimise",     "#bdc3c7"],  # grey
            [  0,  50,  50, 100, "Maintain",     "#3498db"],  # blue
            [ 50,  70,  50, 100, "Grow Lower",   "#1e8449"],  # dark green
            [ 70, 100,  50,  70, "Grow Upper",   "#2ecc71"],  # lighter green (band B)
            [ 50, 100,  25,  50, "Grow Upper",   "#2ecc71"],  # lighter green (band A)
            [ 70, 100,  70, 100, "To Kill For",  "#d4ac0d"],  # mustard
            [ 50, 100,   0,  25, "Nurture",      "#a8e6cf"],  # mint
        ], columns=["x1","x2","y1","y2","tier","fill"])

        rects = alt.Chart(boxes).mark_rect(opacity=0.18).encode(
            x=alt.X("x1:Q", title="Potential Score (0–100)", scale=alt.Scale(domain=[0,100])),
            x2="x2:Q",
            y=alt.Y("y1:Q", title="Performance Score (0–100)", scale=alt.Scale(domain=[0,100])),
            y2="y2:Q",
            color=alt.Color("fill:N", scale=None, legend=None)
        )

        # ---- Labels (tier names only) ----
        labels = pd.DataFrame({
            "xc": (boxes["x1"] + boxes["x2"]) / 2.0,
            "yc": (boxes["y1"] + boxes["y2"]) / 2.0,
            "tier": boxes["tier"]
        })
        texts = alt.Chart(labels).mark_text(
            align="center", baseline="middle", fontSize=14, fontWeight="bold", color="#2c3e50"
        ).encode(x="xc:Q", y="yc:Q", text="tier:N")

        # ---- Scatter points ----
        tier_order  = ["Minimise","Maintain","Nurture","Grow Lower","Grow Upper","To Kill For"]
        tier_colors = alt.Scale(
            domain=tier_order,
            range=["#bdc3c7","#3498db","#a8e6cf","#1e8449","#2ecc71","#d4ac0d"]
        )
        scatter = alt.Chart(f).mark_circle(size=90, opacity=0.95).encode(
            x=alt.X("_pot:Q",  title="Potential Score (0–100)",   scale=alt.Scale(domain=[0,100])),
            y=alt.Y("_perf:Q", title="Performance Score (0–100)", scale=alt.Scale(domain=[0,100])),
            color=alt.Color("_tier:N", title="Tier", scale=tier_colors, sort=tier_order),
            tooltip=[
                "OutletID","region","channel","format_size",
                alt.Tooltip("performance_score", title="Performance", format=".1f"),
                alt.Tooltip("potential_score",  title="Potential",  format=".1f"),
                alt.Tooltip("unified_commercial_score", title="Unified", format=".1f"),
                alt.Tooltip("_tier", title="Tier"),
            ]
        ).properties(height=560)

        chart = (rects + texts + scatter).interactive()
        st.altair_chart(chart, use_container_width=True)

        # ---- Summary by tier (table below chart) ----
        tier_counts = (
            f["_tier"].value_counts()
             .reindex(tier_order, fill_value=0)
             .reset_index()
             .rename(columns={"index":"Tier","_tier":"Outlets"})
        )
        st.caption("Outlets by Tier")
        st.dataframe(tier_counts, hide_index=True, use_container_width=True)

        # Cleanup
        f.drop(columns=["_perf","_pot","_tier"], inplace=True, errors="ignore")

    else:
        st.info("Need performance_score and potential_score to render outlet prioritisation matrix.")




with tab3:
    st.subheader("Outlet Drilldown")
    if len(f):
        latest_week = f["week_start_date"].max()
        latest = f[f["week_start_date"]==latest_week]
        options = (latest["OutletID"] + " — " + latest["region"] + " • " + latest["channel"]).unique()
        sel = st.selectbox("Select outlet", options=sorted(options))
        if sel:
            sel_id = sel.split(" — ")[0]
            hist = f[f["OutletID"]==sel_id].sort_values("week_start_date")
            o = latest[latest["OutletID"]==sel_id].iloc[0]

            rep_txt = ""
            if "sales_territory" in o and "account_rep" in o:
                rep_txt = f"  \n**Territory:** {o['sales_territory']} • **Rep:** {o['account_rep']}"
            st.markdown(f"**Outlet:** {o['OutletID']}  \n{o.get('region','')} • {o.get('channel','')} • {o.get('format_size','')}{rep_txt}")

            k1, k2, k3 = st.columns(3)
            with k1: st.metric("Unified", round(o.get("unified_commercial_score",0),1))
            with k2: st.metric("Performance", round(o.get("performance_score",0),1))
            with k3: st.metric("Potential", round(o.get("potential_score",0),1))

            if {"performance_score","potential_score","unified_commercial_score"}.issubset(hist.columns):
                ts = hist[["week_start_date","performance_score","potential_score","unified_commercial_score"]] \
                        .melt("week_start_date", var_name="Metric", value_name="Score")
                line = alt.Chart(ts).mark_line(point=True).encode(
                    x=alt.X("week_start_date:T", title="Week"),
                    y=alt.Y("Score:Q", title="Score (0–100)"),
                    color="Metric"
                )
                st.altair_chart(line, use_container_width=True)

            driver_cols = [
                "sales_value_gbp","nine_l_cases_total","sales_growth_4w",
                "oos_rate","sku_range_onhand","order_freq_4w","recency_weeks",
                "promo_flag","promo_uplift_pct","performance_index",
                "expected_value_gbp","headroom_ratio","range_gap_ratio","demand_index","external_index",
                "footfall_count","population_1km","median_income_gbp","competitor_outlets_1km","nearby_venues_score",
                "alcohol_index","region_growth_index","avg_rating","review_count","recent_sentiment"
            ]
            have = ["week_start_date"] + [c for c in driver_cols if c in hist.columns]
            with st.expander("Drivers & context", expanded=True):
                st.dataframe(hist[have].sort_values("week_start_date", ascending=False),
                             use_container_width=True, hide_index=True)
    else:
        st.info("No rows in the current selection.")