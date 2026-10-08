import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import theme, components, data_pipeline as dp, thesis_results as T
from src.cache import get_bundle, bootstrap_player_ci

bundle = get_bundle()
df = bundle["df"]

components.page_header(
    "Player analysis",
    "Player DNA",
    "A full fictional profile: relative percentile positioning, soft profile membership, "
    "match-to-match stability, and the players whose contextual profile is most similar.",
    badges=["demo"],
)

players = sorted(df.player_id.unique())
player_id = st.selectbox("Fictional player", players, index=0)

p = df[df.player_id == player_id].sort_values("appearance_no")
first_row = p.iloc[0]

st.write("")
components.stat_row([
    ("Squad", str(first_row.squad)),
    ("Role", str(first_row.role)),
    ("Age", f"{first_row.age:.1f}"),
    ("Appearances", f"{len(p)}"),
    ("Autumn / Spring", f"{(p['round']=='Autumn').sum()} / {(p['round']=='Spring').sum()}"),
])

st.write("")
reliability = components.reliability_label(len(p))
components.stat_row([
    ("Stability", f"{first_row.stability:.3f}"),
    ("Mean hybridity", f"{first_row.mean_hybridity:.3f}"),
    ("Dominant-profile probability", f"{first_row.mean_dominant_prob:.3f}"),
    ("Estimate reliability", reliability),
])
components.note(
    "Stability describes match-to-match repeatability of the soft GMM profile; it is not a measure "
    "of player quality. Hybridity describes how evenly a single appearance mixes the two "
    "motor-activity profiles. Reliability reflects how many appearances the estimate is based on, "
    "not accuracy against any ground truth."
)

st.write("")
st.markdown("### Profile")
tab_motor, tab_technical, tab_combined = st.tabs(["Motor profile", "Technical/tactical profile", "Combined profile"])

def _render_radar(dim_map, reversed_dims, caption):
    radar_df = dp.percentile_profile(df, player_id, dim_map, reversed_dims)
    if radar_df.empty or radar_df["percentile"].isna().all():
        st.info("Not enough data to build this radar for the selected player.")
        return
    left, right = st.columns([1.1, 1])
    with left:
        fig = components.radar_chart(
            radar_df["dimension"].tolist(),
            [(player_id, radar_df["percentile"].tolist())],
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        bar = go.Figure(go.Bar(
            x=radar_df["percentile"], y=radar_df["dimension"], orientation="h",
            marker_color=theme.ACCENT,
        ))
        bar.update_layout(xaxis=dict(range=[0, 100], title="Percentile vs. reference group"), yaxis=dict(title=""))
        theme.apply_plot_theme(bar, height=330)
        st.plotly_chart(bar, use_container_width=True)
    st.caption(caption)

with tab_motor:
    _render_radar(
        T.MOTOR_RADAR_DIMENSIONS, set(),
        "Reference group: players of the same squad category and broad role. Percentile 50 is "
        "average for that group, not an absolute benchmark.",
    )

with tab_technical:
    coverage = p["technical_detail_available"].mean()
    if coverage < 0.34:
        st.info(
            f"Only {coverage:.0%} of this player's appearances include the detailed technical "
            "breakdown (the synthetic data mirrors the thesis's own uneven technical-data "
            "completeness). The radar below is based only on appearances where it is available."
        )
    _render_radar(
        T.TECHNICAL_RADAR_DIMENSIONS, T.TECHNICAL_RADAR_REVERSED,
        "Safety is reversed (fewer key losses / errors \u2192 higher percentile). Built only from "
        "appearances with the detailed technical breakdown available.",
    )

with tab_combined:
    _render_radar(
        T.COMBINED_RADAR_DIMENSIONS, T.COMBINED_RADAR_REVERSED,
        "A synthetic summary across technical and motor dimensions. Limited to eight dimensions "
        "by design, for readability.",
    )

st.write("")
st.markdown("### DNA fingerprint")
st.caption(
    "Each column is one appearance. Column colour is a continuous gradient of the GMM "
    "elevated-activity membership probability; the small square above it is that appearance's "
    "fitted HMM state. The dotted line marks the Autumn / Spring boundary."
)
st.plotly_chart(components.dna_fingerprint_figure(p), use_container_width=True)
components.note(
    "States are fitted on this player's full cohort by a Gaussian Hidden Markov Model and ranked "
    "by overall motor-activity level. They are deliberately not named by subjective 'form' "
    "(e.g. good/bad) \u2014 see the HMM & State Transitions page for the transition matrix and "
    "model-selection diagnostics."
)

with st.expander("Profile composition as a stacked area (alternate view)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p.appearance_no, y=p.gmm_lower_prob, mode="lines", stackgroup="one",
        name="Lower-activity profile", line=dict(width=0.5, color=theme.TEXT_SECONDARY),
        fillcolor="rgba(154,163,156,0.35)",
    ))
    fig.add_trace(go.Scatter(
        x=p.appearance_no, y=p.gmm_higher_prob, mode="lines", stackgroup="one",
        name="Elevated-activity profile", line=dict(width=0.5, color=theme.ACCENT),
        fillcolor="rgba(79,157,110,0.55)",
    ))
    for _, boundary_row in p[p["round"] != p["round"].shift(1)].iloc[1:].iterrows():
        fig.add_vline(x=boundary_row.appearance_no - 0.5, line_dash="dot", line_color="rgba(255,255,255,0.25)")
    fig.update_layout(
        title="GMM soft membership by appearance (stacked to 1.0)",
        xaxis_title="Appearance", yaxis_title="Membership probability", yaxis_range=[0, 1],
    )
    theme.apply_plot_theme(fig, height=340)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Bootstrap confidence intervals for this player's stability and hybridity"):
    ci = bootstrap_player_ci(player_id, n_boot=300)
    c1, c2 = st.columns(2)
    with c1:
        lo, hi = ci["stability_ci"]
        st.metric("Stability (95% CI)", f"{first_row.stability:.3f}", f"[{lo:.3f}, {hi:.3f}]")
    with c2:
        lo, hi = ci["hybridity_ci"]
        st.metric("Hybridity (95% CI)", f"{first_row.mean_hybridity:.3f}", f"[{lo:.3f}, {hi:.3f}]")
    st.caption(
        "Block bootstrap over consecutive appearances (block size 4), 300 resamples. Wider "
        "intervals indicate less precise estimates, typically for players with fewer appearances."
    )

st.write("")
st.markdown("### Similar players")
contextual = st.checkbox(
    "Restrict to the same round and squad category too (thesis-style contextual match)",
    value=True, key="similarity_contextual",
)
sim = dp.role_restricted_similarity(df, player_id, dp.MOTOR_COLS, same_round=contextual, same_squad=contextual)
fell_back = False
if sim.empty and contextual:
    # Graceful fallback: the strict round+squad+role pool can be thin for
    # smaller squads: FPLxx is fictional either way, so relaxing to
    # role-only still keeps the comparison meaningful rather than empty.
    sim = dp.role_restricted_similarity(df, player_id, dp.MOTOR_COLS, same_round=False, same_squad=False)
    fell_back = True

if contextual and not fell_back:
    st.caption("Restricted to the same round, squad category and broad role, matching the thesis's own approach.")
elif fell_back:
    st.caption(
        "Too few players share this player's exact round + squad + role combination, so this falls "
        "back to the same broad role only."
    )
else:
    st.caption("Restricted to players in the same broad role (across all rounds and squad categories).")

if sim.empty:
    st.info("No comparable players found.")
else:
    st.dataframe(
        sim.rename(columns={
            "player_id": "Player", "squad": "Squad", "role": "Role",
            "cosine_similarity": "Cosine similarity", "euclidean_distance": "Euclidean distance",
        }).round(3),
        use_container_width=True, hide_index=True,
    )
    top = sim.iloc[0]
    breakdown = dp.similarity_feature_breakdown(
        df, player_id, top.player_id, dp.MOTOR_COLS,
        same_round=(contextual and not fell_back), same_squad=(contextual and not fell_back),
    )
    if not breakdown.empty:
        closest_dims = ", ".join(dp.MOTOR_LABELS[c] for c in breakdown["feature"].iloc[:3])
        components.note(
            f"<b>{top.player_id}</b> is the closest match (cosine similarity "
            f"{top.cosine_similarity:.3f}). The dimensions contributing least to the difference "
            f"(i.e. most similar), measured in the same standardized space the similarity score "
            f"itself is computed in: {closest_dims}.",
            strong_prefix="Why these players are similar",
        )

st.write("")
components.download_csv_button(p, f"{player_id}_synthetic_demo.csv", "Download this player's synthetic appearances (CSV)")
