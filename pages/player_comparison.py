import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import theme, components, data_pipeline as dp, thesis_results as T
from src.cache import get_bundle

bundle = get_bundle()
df = bundle["df"]

components.page_header(
    "Player analysis",
    "Player comparison",
    "Set two fictional players side by side: motor and percentile profile, soft profile composition, "
    "stability and hybridity, and trajectory over appearances.",
    badges=["demo"],
)

players = sorted(df.player_id.unique())
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    player_a = st.selectbox("Player A", players, index=0)
with c3:
    restrict_role = st.checkbox("Only compare within the same broad role", value=True)

role_a = df[df.player_id == player_a].role.iloc[0]
# Build Player B's options BEFORE rendering the widget, so an incompatible
# pairing can never actually be selected in the first place -- rather than
# letting the person pick one and then rejecting it with a warning.
if restrict_role:
    candidates_b = [p for p in players if p != player_a and df[df.player_id == p].role.iloc[0] == role_a]
else:
    candidates_b = [p for p in players if p != player_a]

with c2:
    if candidates_b:
        player_b = st.selectbox("Player B", candidates_b, index=0)
        st.caption(f"Showing players in role: {role_a}" if restrict_role else "Showing all other players, any role.")
    else:
        st.selectbox("Player B", ["(no comparable players)"], disabled=True)
        player_b = None

if player_b is None:
    st.info(
        f"No other player shares {player_a}'s role ({role_a}) in this synthetic cohort. "
        "Uncheck the restriction above to compare across roles."
    )
    st.stop()

role_b = df[df.player_id == player_b].role.iloc[0]
if not restrict_role and role_a != role_b:
    st.caption(
        f"{player_a} ({role_a}) and {player_b} ({role_b}) play different broad roles \u2014 percentiles "
        "below are each computed within the player's own role group, so a given percentile does not mean "
        "the same absolute level across the two."
    )

pa = df[df.player_id == player_a].sort_values("appearance_no")
pb = df[df.player_id == player_b].sort_values("appearance_no")

st.write("")
left, right = st.columns(2)
with left:
    st.markdown(f"#### {player_a}")
    components.stat_row([
        ("Squad / Role", f"{pa.squad.iloc[0]} / {pa.role.iloc[0]}"),
        ("Appearances", f"{len(pa)}"),
        ("Stability", f"{pa.stability.iloc[0]:.3f}"),
    ])
with right:
    st.markdown(f"#### {player_b}")
    components.stat_row([
        ("Squad / Role", f"{pb.squad.iloc[0]} / {pb.role.iloc[0]}"),
        ("Appearances", f"{len(pb)}"),
        ("Stability", f"{pb.stability.iloc[0]:.3f}"),
    ])

st.write("")
st.markdown("### Motor profile (percentile vs. reference group)")
radar_a = dp.percentile_profile(df, player_a, T.MOTOR_RADAR_DIMENSIONS, set())
radar_b = dp.percentile_profile(df, player_b, T.MOTOR_RADAR_DIMENSIONS, set())
if not radar_a.empty and not radar_b.empty:
    fig = components.radar_chart(
        radar_a["dimension"].tolist(),
        [(player_a, radar_a["percentile"].tolist()), (player_b, radar_b["percentile"].tolist())],
    )
    st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Each player's percentile is computed within their own squad/role reference group, so this "
    "overlay compares relative standing, not necessarily identical raw values."
)

st.write("")
st.markdown("### GMM composition and dominant state")
g1, g2 = st.columns(2)
with g1:
    st.markdown(f"**{player_a}**")
    components.stat_row([
        ("Mean hybridity", f"{pa.mean_hybridity.iloc[0]:.3f}"),
        ("Dominant-profile prob.", f"{pa.mean_dominant_prob.iloc[0]:.3f}"),
    ])
    dom_a = "Elevated activity" if pa.gmm_higher_prob.mean() > pa.gmm_lower_prob.mean() else "Lower activity"
    st.caption(f"Average dominant GMM profile: {dom_a}")
with g2:
    st.markdown(f"**{player_b}**")
    components.stat_row([
        ("Mean hybridity", f"{pb.mean_hybridity.iloc[0]:.3f}"),
        ("Dominant-profile prob.", f"{pb.mean_dominant_prob.iloc[0]:.3f}"),
    ])
    dom_b = "Elevated activity" if pb.gmm_higher_prob.mean() > pb.gmm_lower_prob.mean() else "Lower activity"
    st.caption(f"Average dominant GMM profile: {dom_b}")

st.write("")
st.markdown("### Trajectory overlay")
metric = st.selectbox("Metric", dp.MOTOR_COLS, format_func=lambda c: dp.MOTOR_LABELS[c])
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=pa.appearance_no, y=pa[metric], mode="lines+markers", name=player_a,
                           line=dict(color=theme.ACCENT)))
fig2.add_trace(go.Scatter(x=pb.appearance_no, y=pb[metric], mode="lines+markers", name=player_b,
                           line=dict(color=theme.THESIS_TONE)))
fig2.update_layout(title=f"{dp.MOTOR_LABELS[metric]} across appearances",
                    xaxis_title="Appearance", yaxis_title=dp.MOTOR_LABELS[metric])
theme.apply_plot_theme(fig2, height=400)
st.plotly_chart(fig2, use_container_width=True)

st.write("")
st.markdown("### Hidden-state transition behaviour")
h1, h2 = st.columns(2)
for col, p, name in [(h1, pa, player_a), (h2, pb, player_b)]:
    with col:
        counts = p.hmm_state_idx.map(lambda i: T.STATE_ORDER[i]).value_counts().reindex(T.STATE_ORDER, fill_value=0)
        fig3 = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[theme.TEXT_MUTED, "#B9A36B", theme.ACCENT]))
        fig3.update_layout(title=f"{name}: appearances per state", yaxis_title="Appearances")
        theme.apply_plot_theme(fig3, height=300)
        st.plotly_chart(fig3, use_container_width=True)

components.note(
    "Comparing two synthetic players shows differences in relative standing, soft-profile "
    "composition, and state occupancy. None of these differences imply one player is "
    "objectively 'better' \u2014 as in the source thesis, they describe profile structure, not "
    "sporting quality."
)
