import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import theme, components, thesis_results as T
from src.cache import get_bundle
from src.stats_utils import kaplan_meier, bootstrap_km_median_ci


def render():
    bundle = get_bundle()
    events = bundle["survival_events"]

    components.section_header(
        "Survival analysis",
        "Kaplan\u2013Meier estimate of the number of appearances until a player's first direct "
        "transition from the Lower-activity to the Elevated-activity hidden state.",
    )

    components.note(
        f"<b>Event:</b> {T.SURVIVAL_RESULTS['event_definition']}<br><br>"
        f"<b>Censoring:</b> {T.SURVIVAL_RESULTS['censoring_note']}"
    )

    km = kaplan_meier(events["time"], events["observed"])
    ci = bootstrap_km_median_ci(events["time"], events["observed"], n_boot=1000)

    st.write("")
    st.markdown("### Synthetic demo")
    components.provenance_note("demo", "Computed live from the fitted HMM state sequence on the synthetic cohort.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players analysed", f"{km['n']}")
    c2.metric("Observed events", f"{km['n_events']}")
    c3.metric("Right-censored", f"{km['n_censored']}")
    c4.metric("Median appearances to transition", f"{km['median']}" if km["median"] is not None else "not reached")

    fig = go.Figure(go.Scatter(x=km["xs"], y=km["ys"], mode="lines", line_shape="hv",
                                name="Kaplan\u2013Meier estimate", line=dict(color=theme.ACCENT, width=2.5)))
    if km["median"] is not None:
        fig.add_vline(x=km["median"], line_dash="dash", line_color=theme.THESIS_TONE,
                      annotation_text=f"Median = {km['median']:.0f}")
    fig.update_layout(
        title="Kaplan\u2013Meier survival curve \u2014 probability of not yet transitioning",
        xaxis_title="Appearance number", yaxis_title="P(event not yet observed)", yaxis_range=[0, 1.03],
    )
    theme.apply_plot_theme(fig, height=440)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Bootstrap 95% CI for the median: [{ci[0]:.1f}, {ci[1]:.1f}] appearances (1000 resamples).")

    with st.expander("At-risk / events / censored table"):
        st.dataframe(km["table"].round(3), use_container_width=True, hide_index=True)

    st.write("")
    st.markdown("### Thesis result")
    components.provenance_note("thesis", "Reported exactly as in the source thesis.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players analysed", f"{T.SURVIVAL_RESULTS['n_players']}")
    c2.metric("Observed events", f"{T.SURVIVAL_RESULTS['n_events']}")
    c3.metric("Right-censored", f"{T.SURVIVAL_RESULTS['n_censored']}")
    c4.metric("Median appearances to transition", f"{T.SURVIVAL_RESULTS['median_appearances']}")
    lo, hi = T.SURVIVAL_RESULTS["median_ci95"]
    st.caption(f"Bootstrap 95% CI for the median: [{lo}, {hi}] appearances.")

    components.note(
        "The thesis's own curve shape is not reproduced here (only its summary statistics are "
        "published in the source document) \u2014 only the synthetic demo curve above is actually "
        "plotted. Treat the two result blocks as independent: same method, different cohorts."
    )
