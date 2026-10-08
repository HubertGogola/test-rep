import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import theme, components, data_pipeline as dp
from src.cache import get_bundle

bundle = get_bundle()
df = bundle["df"]

components.page_header(
    "Player analysis",
    "Cohort analysis",
    "Exploratory analysis of the synthetic cohort: distributions, correlation structure, "
    "and data completeness \u2014 the same checks the thesis runs before any modelling.",
    badges=["demo"],
)

st.sidebar.markdown("#### Filters")
squads = st.sidebar.multiselect("Squad", sorted(df.squad.unique()), default=sorted(df.squad.unique()))
roles = st.sidebar.multiselect("Role", sorted(df.role.unique()), default=sorted(df.role.unique()))
rounds = st.sidebar.multiselect("Round", sorted(df["round"].unique()), default=sorted(df["round"].unique()))
filtered = df[df.squad.isin(squads) & df.role.isin(roles) & df["round"].isin(rounds)]
if filtered.empty:
    st.warning("No observations match the selected filters.")
    st.stop()

st.write("")
components.stat_row([
    ("Observations", f"{len(filtered)}"),
    ("Players", f"{filtered.player_id.nunique()}"),
    ("Median appearances / player", f"{int(filtered.groupby('player_id').size().median())}"),
    ("Median minutes", f"{filtered.minutes.median():.0f}"),
])

st.write("")
st.markdown("### Distribution by group")
group_dim = st.selectbox("Group by", ["squad", "role", "round"], format_func=lambda c: c.capitalize())
metric = st.selectbox("Metric", dp.MOTOR_COLS, format_func=lambda c: dp.MOTOR_LABELS[c])

left, right = st.columns([1.2, 1])
with left:
    fig = px.box(filtered, x=group_dim, y=metric, points="outliers",
                 title=f"{dp.MOTOR_LABELS[metric]} by {group_dim}")
    theme.apply_plot_theme(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)
with right:
    summary = filtered.groupby(group_dim)[metric].agg(["mean", "median", "std", "count"]).round(1)
    st.dataframe(summary, use_container_width=True)

st.write("")
st.markdown("### Correlation structure of motor variables")
corr = filtered[dp.MOTOR_COLS].corr()
corr.index = [dp.MOTOR_LABELS[c] for c in corr.index]
corr.columns = [dp.MOTOR_LABELS[c] for c in corr.columns]
fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale=theme.PLOT_DIVERGING,
                      zmin=-1, zmax=1, title="Correlation matrix (synthetic cohort)")
theme.apply_plot_theme(fig_corr, height=430)
st.plotly_chart(fig_corr, use_container_width=True)
components.note(
    "In the thesis's own motor dataset, HSR and sprint distance were the most strongly correlated "
    "pair \u2014 look for the same pattern here. PII's correlation with raw total distance is not "
    "directly comparable between the two datasets: the formula places distance in the denominator, "
    "so some negative association is expected by construction, and how strongly that shows up "
    "empirically depends on how correlated distance happens to be with the formula's numerator "
    "terms in a given dataset. The thesis's cohort showed a very weak overall relationship "
    "(correlation \u22480.07); this independent synthetic cohort is not expected to reproduce that "
    "exact figure."
)

st.write("")
st.markdown("### Data completeness")
c1, c2 = st.columns(2)
with c1:
    completeness = filtered.groupby("round")["technical_detail_available"].mean().reindex(["Autumn", "Spring"])
    fig_comp = go.Figure(go.Bar(x=completeness.index, y=completeness.values * 100, marker_color=theme.ACCENT))
    fig_comp.update_layout(title="Share of appearances with detailed technical data",
                            yaxis=dict(title="% of appearances", range=[0, 100]))
    theme.apply_plot_theme(fig_comp, height=360)
    st.plotly_chart(fig_comp, use_container_width=True)
    st.caption(
        "Motor (GPS) variables are complete for every qualifying appearance. Detailed "
        "technical/tactical variables are not \u2014 mirroring the uneven completeness documented "
        "in the thesis, which is why the main dynamic analysis is built on motor data."
    )
with c2:
    fig_min = px.histogram(filtered, x="minutes", nbins=30, title="Minutes played (qualifying appearances, \u226520)")
    theme.apply_plot_theme(fig_min, height=360)
    st.plotly_chart(fig_min, use_container_width=True)

st.write("")
st.markdown("### Appearances per player")
counts = filtered.groupby("player_id").size().sort_values(ascending=False)
fig_app = go.Figure(go.Bar(x=list(range(1, len(counts) + 1)), y=counts.values, marker_color=theme.ACCENT))
fig_app.update_layout(
    title="Appearances per player (sorted)",
    xaxis_title="Player (ranked)", yaxis_title="Qualifying appearances",
)
fig_app.add_hline(y=8, line_dash="dash", line_color=theme.THESIS_TONE,
                   annotation_text="Minimum appearances used for the Dynamic Player DNA cohort")
theme.apply_plot_theme(fig_app, height=380)
st.plotly_chart(fig_app, use_container_width=True)
