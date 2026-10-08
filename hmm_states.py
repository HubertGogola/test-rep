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


def render():
    bundle = get_bundle()
    df = bundle["df"]
    hmm_out = bundle["hmm"]

    components.section_header(
        "HMM & state transitions",
        "A Hidden Markov Model fitted on chronologically ordered appearances, identifying statistical "
        "activity states and the probability of moving between them from one appearance to the next.",
    )

    components.note(
        "The HMM models sequences of consecutive <b>appearances</b>, not minutes within a single match. "
        "States are named only after estimation, by overall activity level \u2014 never as subjective "
        "'form' (bad / average / good)."
    )


    @st.cache_data(show_spinner="Fitting candidate HMMs for model selection...")
    def _model_selection_table(k_values: tuple, n_init: int = 2):
        rows = []
        for k in k_values:
            res = dp.fit_player_hmm(
                df, ["PC1", "PC2", "PC3"], n_states=k, n_init=n_init, random_state=11,
                intensity_col="PC1",
            )
            rows.append({
                "States": k, "Log-likelihood": res["log_likelihood"], "AIC": res["aic"], "BIC": res["bic"],
                "Mean ARI across inits": res["mean_ari_across_inits"],
            })
        return pd.DataFrame(rows)


    st.markdown("### Model selection")
    sel = _model_selection_table((2, 3, 4))
    left, right = st.columns([1.2, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sel["States"], y=sel["AIC"], mode="lines+markers", name="AIC"))
        fig.add_trace(go.Scatter(x=sel["States"], y=sel["BIC"], mode="lines+markers", name="BIC"))
        fig.update_layout(title="AIC / BIC by number of states (synthetic demo)", xaxis_title="States")
        theme.apply_plot_theme(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.dataframe(sel.round(1), use_container_width=True, hide_index=True)
    st.caption(
        "This application retains 3 states throughout, for interpretability and consistency with the "
        "Lower / Baseline / Elevated framing \u2014 the same principle (not minimising a single metric "
        "in isolation) the thesis applies when choosing its own model sizes."
    )
    with st.expander("Thesis benchmark: HMM model selection (Table 18)"):
        components.provenance_note("thesis", "")
        thesis_sel = pd.DataFrame(T.HMM_MODEL_SELECTION,
                                   columns=["States", "Log-likelihood", "AIC", "BIC", "Smallest state share (%)", "Mean ARI"])
        st.dataframe(thesis_sel, use_container_width=True, hide_index=True)
        st.caption(f"The thesis retained {T.HMM_CHOSEN_STATES} states, each fit {T.HMM_INIT_RUNS} times from different initial values.")

    st.write("")
    st.markdown("### Transition matrix (3 states)")
    transmat = hmm_out["transmat"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Synthetic demo**")
        components.provenance_note("demo", "")
        fig_t = px.imshow(pd.DataFrame(transmat, index=T.STATE_ORDER, columns=T.STATE_ORDER),
                           text_auto=".3f", aspect="auto", color_continuous_scale=theme.PLOT_SEQUENTIAL_GREEN,
                           zmin=0, zmax=1, title="P(next state | current state)")
        theme.apply_plot_theme(fig_t, height=380)
        st.plotly_chart(fig_t, use_container_width=True)
    with c2:
        st.markdown("**Thesis result**")
        components.provenance_note("thesis", "")
        fig_th = px.imshow(pd.DataFrame(T.HMM_TRANSITION_MATRIX, index=T.STATE_ORDER, columns=T.STATE_ORDER),
                            text_auto=".3f", aspect="auto", color_continuous_scale=theme.PLOT_SEQUENTIAL_GREEN,
                            zmin=0, zmax=1, title="P(next state | current state)")
        theme.apply_plot_theme(fig_th, height=380)
        st.plotly_chart(fig_th, use_container_width=True)

    persistence_demo = np.diag(transmat)
    st.caption(
        f"Synthetic demo persistence (probability of remaining in the same state): "
        f"Lower {persistence_demo[0]:.3f}, Baseline {persistence_demo[1]:.3f}, Elevated {persistence_demo[2]:.3f}."
    )
    with st.expander("Thesis persistence note"):
        components.provenance_note("thesis", "")
        st.write(T.HMM_PERSISTENCE_NOTE)

    show_sankey = st.checkbox("Show transition flow as a Sankey diagram (synthetic demo)", key="hmm_show_sankey")
    if show_sankey:
        src, tgt, val = [], [], []
        for i in range(3):
            for j in range(3):
                src.append(i); tgt.append(j + 3); val.append(float(transmat[i, j]))
        labels = T.STATE_ORDER + [s + " (next)" for s in T.STATE_ORDER]
        fig_sankey = go.Figure(go.Sankey(
            node=dict(label=labels, pad=20, thickness=18,
                       color=[theme.TEXT_MUTED, "#B9A36B", theme.ACCENT] * 2),
            link=dict(source=src, target=tgt, value=val),
        ))
        fig_sankey.update_layout(title="State \u2192 next-state transition flow")
        theme.apply_plot_theme(fig_sankey, height=420)
        st.plotly_chart(fig_sankey, use_container_width=True)

    st.write("")
    st.markdown("### State profiles")
    state_profile = df.groupby("hmm_state_idx")[dp.MOTOR_COLS].mean().round(1)
    state_profile.index = [T.STATE_ORDER[i] for i in state_profile.index]
    state_profile.columns = [dp.MOTOR_LABELS[c] for c in state_profile.columns]
    state_profile.insert(0, "Observations", df.hmm_state_idx.value_counts().sort_index().values)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Synthetic demo**")
        components.provenance_note("demo", "")
        st.dataframe(state_profile, use_container_width=True)
    with c2:
        st.markdown("**Thesis result (Table 19)**")
        components.provenance_note("thesis", "")
        thesis_states = pd.DataFrame(T.HMM_STATE_PROFILES).T
        thesis_states.columns = ["Observations", "Distance/90", "HSR/90", "Sprint/90", "ACC/90", "DEC/90", "PII"]
        st.dataframe(thesis_states, use_container_width=True)

    components.note(
        "Note that a state's overall activity ranking is driven mainly by HSR, sprint, accelerations, "
        "decelerations and PII \u2014 not raw total distance, which can move quite differently (see the "
        "table above: distance is not monotonic across the three states the way the other variables "
        "are). This distance-vs-intensity distinction is also present in the thesis, though the exact "
        "strength of the relationship differs between the two independent datasets."
    )

    st.write("")
    st.markdown("### Player state timeline")
    player_id = st.selectbox("Fictional player", sorted(df.player_id.unique()), key="hmm_player")
    p = df[df.player_id == player_id].sort_values("appearance_no")
    state_colors = [theme.TEXT_MUTED, "#B9A36B", theme.ACCENT]
    strip = go.Figure(go.Scatter(
        x=p.appearance_no, y=[0] * len(p), mode="markers",
        marker=dict(size=22, symbol="square", color=p.hmm_state_idx,
                    colorscale=[[0, state_colors[0]], [0.5, state_colors[1]], [1, state_colors[2]]],
                    cmin=0, cmax=2,
                    colorbar=dict(tickvals=[0, 1, 2], ticktext=T.STATE_ORDER, thickness=12, len=0.6)),
        hovertext=[f"Appearance {a}: {T.STATE_ORDER[s]} (posterior {q:.2f})"
                   for a, s, q in zip(p.appearance_no, p.hmm_state_idx, p.hmm_state_posterior_max)],
        hoverinfo="text",
    ))
    strip.update_layout(xaxis_title="Appearance", yaxis=dict(visible=False, range=[-1, 1]), showlegend=False)
    theme.apply_plot_theme(strip, height=150)
    st.plotly_chart(strip, use_container_width=True)

    st.write("")
    st.markdown("### Cohort state occupancy")
    occ = df.hmm_state_idx.value_counts().sort_index()
    occ.index = [T.STATE_ORDER[i] for i in occ.index]
    fig_occ = go.Figure(go.Bar(x=occ.index, y=occ.values, marker_color=state_colors))
    fig_occ.update_layout(title="Appearances per state (synthetic cohort)", yaxis_title="Appearances")
    theme.apply_plot_theme(fig_occ, height=340)
    st.plotly_chart(fig_occ, use_container_width=True)
