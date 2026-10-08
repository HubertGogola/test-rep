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

from src import theme, components, data_pipeline as dp, thesis_results as T
from src.cache import get_bundle
from src.stats_utils import paired_round_comparison


def render():
    bundle = get_bundle()
    df = bundle["df"]

    components.section_header(
        "Dynamic Player DNA",
        "The system's central idea: instead of one static label, every appearance carries soft profile "
        "probabilities, and tracking them match-to-match reveals stability, hybridity, and direction of change.",
    )

    per_player = df.drop_duplicates("player_id")[
        ["player_id", "squad", "role", "n_appearances", "stability", "mean_hybridity", "mean_dominant_prob"]
    ].reset_index(drop=True)

    st.markdown("### Cohort overview")
    components.stat_row([
        ("Players in cohort", f"{len(per_player)}"),
        ("Mean stability", f"{per_player.stability.mean():.3f}"),
        ("Mean hybridity", f"{per_player.mean_hybridity.mean():.3f}"),
        ("Mean dominant-profile probability", f"{per_player.mean_dominant_prob.mean():.3f}"),
    ])

    left, right = st.columns([1.2, 1])
    with left:
        fig = px.scatter(
            per_player, x="stability", y="mean_hybridity", color="role", size="n_appearances",
            hover_data=["player_id", "squad"], title="Stability vs. hybridity, one point per fictional player",
        )
        fig.update_layout(xaxis_title="Stability", yaxis_title="Mean hybridity")
        theme.apply_plot_theme(fig, height=440)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        components.note(
            "High stability + low hybridity: a player whose profile repeats clearly from match to "
            "match.<br><br>Low stability: a player whose balance between the two motor profiles shifts "
            "often.<br><br>High hybridity: appearances that mix both profiles close to evenly, rather "
            "than leaning on one.",
            strong_prefix="Reading this chart",
        )

    st.write("")
    st.markdown("### Most stable / most variable / most hybrid (synthetic demo)")
    components.provenance_note("demo", "Computed live from the synthetic cohort, ranked exactly as in the thesis table.")
    most_stable = per_player.nlargest(3, "stability")
    most_variable = per_player.nsmallest(3, "stability")
    most_hybrid = per_player.nlargest(3, "mean_hybridity")
    t1, t2, t3 = st.columns(3)
    for col, title, table in [(t1, "Most stable", most_stable), (t2, "Most variable", most_variable),
                               (t3, "Most hybrid", most_hybrid)]:
        with col:
            st.markdown(f"**{title}**")
            st.dataframe(
                table[["player_id", "n_appearances", "stability", "mean_hybridity"]].round(3)
                .rename(columns={"player_id": "Player", "n_appearances": "Apps"}),
                use_container_width=True, hide_index=True,
            )

    with st.expander("Thesis reference (aggregate cohort only)"):
        components.provenance_note("thesis", "")
        st.write(
            f"Thesis cohort: {T.DYNAMIC_DNA_COHORT['n_players']} players with \u2265"
            f"{T.DYNAMIC_DNA_COHORT['min_appearances']} appearances. Mean stability = "
            f"{T.DYNAMIC_DNA_COHORT['mean_stability']}, mean hybridity = {T.DYNAMIC_DNA_COHORT['mean_hybridity']}, "
            f"mean dominant-profile probability = {T.DYNAMIC_DNA_COHORT['mean_dominant_probability']}."
        )
        st.caption(
            "The thesis also illustrates its most-stable / most-variable / most-hybrid players with "
            "individual worked examples, identified by pseudonymised player codes. Those individual "
            "results are intentionally not reproduced in this public application, even in aggregate-"
            "looking form \u2014 only cohort-level statistics from the thesis are shown here. The "
            "qualitative pattern is instead demonstrated live, above, on the fictional synthetic cohort."
        )

    st.write("")
    st.markdown("### Player deep-dive")
    player_id = st.selectbox("Fictional player", sorted(df.player_id.unique()), key="dynamic_dna_player")
    p = df[df.player_id == player_id].sort_values("appearance_no")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stability", f"{p.stability.iloc[0]:.3f}")
    c2.metric("Mean hybridity", f"{p.mean_hybridity.iloc[0]:.3f}")
    c3.metric("Dominant-profile probability", f"{p.mean_dominant_prob.iloc[0]:.3f}")
    c4.metric("Appearances", f"{len(p)}")

    st.markdown("##### DNA fingerprint")
    st.caption(
        "Column colour: GMM elevated-activity membership probability. Square marker: fitted HMM state."
    )
    st.plotly_chart(components.dna_fingerprint_figure(p), use_container_width=True)

    with st.expander("Elevated-profile probability as a line (alternate view)"):
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(x=p.appearance_no, y=p.gmm_higher_prob, mode="lines+markers",
                                       name="Elevated-activity membership", line=dict(color=theme.ACCENT)))
        fig_traj.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig_traj.update_layout(
            title="Elevated-activity profile probability across appearances",
            xaxis_title="Appearance", yaxis_title="Probability", yaxis_range=[0, 1],
        )
        theme.apply_plot_theme(fig_traj, height=340)
        st.plotly_chart(fig_traj, use_container_width=True)

    if (p["round"] == "Spring").any() and (p["round"] == "Autumn").any():
        autumn_share = p.loc[p["round"] == "Autumn", "gmm_higher_prob"].mean()
        spring_share = p.loc[p["round"] == "Spring", "gmm_higher_prob"].mean()
        change = spring_share - autumn_share
        st.metric("Autumn \u2192 Spring change in elevated-profile share", f"{change:+.3f}",
                  help="A contextual (within-reference-group) change, not a raw motor-output change.")
        components.note(
            f"This reflects the player's position <i>relative to their reference group</i> in each "
            f"round, not necessarily their raw motor output. {T.DYNAMIC_DNA_ROUND_CHANGE_NOTE}",
        )
    else:
        st.caption("This player only has appearances in one round, so no autumn-to-spring comparison is available.")

    st.write("")
    st.markdown("### Autumn vs. Spring: cohort-level motor comparison")
    components.provenance_note("demo", "Paired Wilcoxon test + bootstrap CI + Benjamini-Hochberg FDR correction, computed live.")
    cmp = paired_round_comparison(df, "player_id", "round", dp.MOTOR_COLS, "Autumn", "Spring", n_boot=1000)
    display_cmp = cmp.copy()
    display_cmp["variable"] = display_cmp["variable"].map(dp.MOTOR_LABELS)
    st.dataframe(
        display_cmp[["variable", "autumn_mean", "spring_mean", "abs_change", "pct_change", "effect_dz", "p_fdr", "significant"]]
        .round({"autumn_mean": 1, "spring_mean": 1, "abs_change": 2, "pct_change": 2, "effect_dz": 2, "p_fdr": 4})
        .rename(columns={"variable": "Variable", "autumn_mean": "Autumn mean", "spring_mean": "Spring mean",
                          "abs_change": "Change", "pct_change": "Change %", "effect_dz": "Effect dz",
                          "p_fdr": "p (FDR)", "significant": "Significant"}),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Based on {cmp.attrs['n_players']} synthetic players observed in both rounds.")

    with st.expander("Thesis reference (Table 17)"):
        components.provenance_note("thesis", "")
        thesis_cmp = pd.DataFrame(
            T.SEASON_COMPARISON["rows"],
            columns=["Variable", "Autumn mean", "Spring mean", "Change", "Change %", "Effect dz", "p (FDR)", "Significant"],
        )
        st.dataframe(thesis_cmp, use_container_width=True, hide_index=True)
        st.caption(f"Based on {T.SEASON_COMPARISON['n_players']} thesis players observed in both rounds.")
