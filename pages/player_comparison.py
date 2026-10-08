import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import theme, components, data_pipeline as dp, thesis_results as T
from src import comparison_options as comparison
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

# The selection flow goes from analytical context -> valid player pool -> pair.
# Both dropdowns contain only comparable players; no unsupported pairing can
# produce a warning after the user has already made their selections.
roster = comparison.player_roster(df)
roles = comparison.comparable_roles(roster)
if not roles:
    st.info("There are no comparable player pairs in this demo dataset.")
    st.stop()

group_col, scope_col, squad_col = st.columns([1.05, 1.35, 1.05])
with group_col:
    selected_role = st.selectbox(
        "Position group",
        roles,
        format_func=lambda role: comparison.ROLE_LABELS.get(role, role),
    )
with scope_col:
    scope = st.selectbox(
        "Comparison scope",
        ["Same squad & role", "Same role, across squads"],
        help="Same squad & role is the recommended like-for-like comparison.",
    )

same_squad = scope == "Same squad & role"
with squad_col:
    if same_squad:
        squads = comparison.comparable_squads(roster, selected_role)
        selected_squad = st.selectbox("Squad", squads)
    else:
        selected_squad = None
        st.markdown("<div style='padding-top:1.95rem'></div>", unsafe_allow_html=True)
        st.caption("All squads")

eligible = comparison.candidate_players(roster, selected_role, selected_squad)
if len(eligible) < 2:
    st.info("No comparable pair is available for this reference group.")
    st.stop()

roster_labels = {
    row.player_id: row.squad
    for row in roster.itertuples(index=False)
}
player_cols = st.columns(2)
with player_cols[0]:
    player_a = st.selectbox(
        "Player A",
        eligible,
        format_func=lambda player: f"{player}  ·  {roster_labels[player]}",
        key=f"comparison_a_{selected_role}_{selected_squad}",
    )
with player_cols[1]:
    player_b = st.selectbox(
        "Player B",
        comparison.second_player_choices(eligible, player_a),
        format_func=lambda player: f"{player}  ·  {roster_labels[player]}",
        key=f"comparison_b_{selected_role}_{selected_squad}_{player_a}",
    )

if same_squad:
    st.caption(
        f"Like-for-like comparison · {comparison.ROLE_LABELS.get(selected_role, selected_role)} "
        f"· {selected_squad} · both players share the same percentile reference group."
    )
else:
    st.caption(
        f"Same broad role · {comparison.ROLE_LABELS.get(selected_role, selected_role)}. "
        "Percentiles are relative to each player's own squad-and-role reference group; "
        "they do not represent identical absolute performance levels across squads."
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
    "Percentiles describe each player's standing within their own squad-and-role reference group. "
    "Compare absolute output in the trajectory chart below when needed."
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
