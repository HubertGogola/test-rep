"""
Reusable Streamlit UI building blocks shared across every page.

This is the one module in `src/` that is allowed to import streamlit,
besides the thin `src/cache.py` wrapper. Pages should prefer calling
functions here over writing raw HTML/markdown inline, so the visual
language stays consistent without having to repeat it on every page.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import theme
from . import thesis_results as T


DATA_PATH = "data/synthetic_player_data.csv"


def page_header(eyebrow: str, title: str, subtitle: str = "", badges: list[str] | None = None):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-title" style="font-size:2.1rem;">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    if badges:
        st.markdown(theme.badge_row(badges), unsafe_allow_html=True)
    st.write("")


def section_header(title: str, subtitle: str = ""):
    """
    A lighter heading for a sub-section embedded inside a tab (as opposed
    to `page_header`, which is sized for a full top-level page). Used by
    the sections combined under the Modelling and Reference pages, where
    a full giant title repeated on every tab would be redundant.
    """
    st.markdown(f'<div style="font-size:1.25rem;font-weight:700;margin:0 0 0.3rem 0;">{title}</div>',
                unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div style="color:{theme.TEXT_SECONDARY};font-size:0.92rem;line-height:1.55;'
            f'max-width:880px;margin-bottom:0.9rem;">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def provenance_note(kind: str, text: str):
    """A small inline note clarifying whether a figure is a thesis result
    or a synthetic-demo computation. `kind` is 'thesis' or 'demo'."""
    st.markdown(
        f'<div class="badge-row">{theme.badge(kind)}<span style="color:{theme.TEXT_SECONDARY};'
        f'font-size:0.88rem;">{text}</span></div>',
        unsafe_allow_html=True,
    )


def note(text: str, strong_prefix: str = ""):
    prefix = f"<b>{strong_prefix}</b><br>" if strong_prefix else ""
    st.markdown(f'<div class="note">{prefix}{text}</div>', unsafe_allow_html=True)


def stat_block(label: str, value: str):
    st.markdown(
        f'<div class="panel-tight stat-block">'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def stat_row(items: list[tuple[str, str]]):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            stat_block(label, value)


def sidebar_chrome():
    st.sidebar.markdown(
        f'<div style="padding:0.2rem 0 0.6rem 0;">'
        f'<div style="font-weight:750;font-size:1.05rem;letter-spacing:-0.01em;">Dynamic Player DNA</div>'
        f'<div style="color:{theme.TEXT_SECONDARY};font-size:0.78rem;margin-top:2px;">'
        f'Football analytics research product</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()


def sidebar_footer_note():
    st.sidebar.divider()
    st.sidebar.markdown(
        f'<div style="color:{theme.TEXT_MUTED};font-size:0.74rem;line-height:1.5;">'
        f"All player-level data in this application are synthetic. "
        f"Aggregate results labelled \u201cThesis result\u201d are reproduced "
        f"from the source study and are never recomputed from the "
        f"synthetic dataset.</div>",
        unsafe_allow_html=True,
    )


def radar_chart(categories: list[str], series: list[tuple[str, list[float]]], title: str = "",
                range_max: float = 100) -> go.Figure:
    """series: list of (name, values) pairs, values aligned to `categories`."""
    fig = go.Figure()
    for name, values in series:
        vals = list(values) + [values[0]]
        cats = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself", name=name, line=dict(width=2.2)))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, range_max], gridcolor="rgba(255,255,255,0.10)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.10)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=len(series) > 1,
        title=title,
    )
    theme.apply_plot_theme(fig)
    return fig


def dna_fingerprint_figure(p: pd.DataFrame, height: int = 230) -> go.Figure:
    """
    The application's signature visual. A single compact strip where each
    column is one appearance: column colour is a continuous gradient of
    the GMM elevated-activity membership probability (muted grey = fully
    lower-activity profile, green = fully elevated-activity profile), and
    the small square marker above each column is that appearance's fitted
    HMM state. A dotted line marks the Autumn/Spring boundary.

    `p` must be one player's rows, sorted by appearance_no, with columns
    appearance_no, round, gmm_higher_prob, hmm_state_idx.
    """
    state_colors = [theme.TEXT_MUTED, "#B9A36B", theme.ACCENT]
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=p["appearance_no"], y=[1.0] * len(p), width=0.86,
        marker=dict(
            color=p["gmm_higher_prob"], cmin=0, cmax=1,
            colorscale=[[0, theme.TEXT_MUTED], [1, theme.ACCENT]],
            colorbar=dict(title=dict(text="Elevated profile", font=dict(size=10)),
                           thickness=10, len=0.75, tickvals=[0, 1], ticktext=["Lower", "Elevated"]),
            line=dict(width=0),
        ),
        hovertext=[
            f"Appearance {a} ({r})<br>Elevated-profile probability: {g:.2f}<br>State: {T.STATE_ORDER[s]}"
            for a, r, g, s in zip(p["appearance_no"], p["round"], p["gmm_higher_prob"], p["hmm_state_idx"])
        ],
        hoverinfo="text",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=p["appearance_no"], y=[1.24] * len(p), mode="markers",
        marker=dict(
            symbol="square", size=9, color=p["hmm_state_idx"], cmin=0, cmax=2,
            colorscale=[[0, state_colors[0]], [0.5, state_colors[1]], [1, state_colors[2]]],
            line=dict(width=0),
        ),
        hoverinfo="skip", showlegend=False,
    ))

    prev_round = None
    for a, r in zip(p["appearance_no"], p["round"]):
        if prev_round is not None and r != prev_round:
            fig.add_vline(x=a - 0.5, line_dash="dot", line_color="rgba(255,255,255,0.25)")
        prev_round = r

    fig.update_layout(
        xaxis_title="Appearance",
        yaxis=dict(visible=False, range=[0, 1.45]),
        bargap=0.08,
        showlegend=False,
    )
    theme.apply_plot_theme(fig, height=height)
    return fig


def download_csv_button(df: pd.DataFrame, filename: str, label: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")


def reliability_label(n_appearances: int) -> str:
    """Qualitative reliability tag based on sample size, used consistently
    wherever a per-player stability/hybridity estimate is shown."""
    if n_appearances >= 24:
        return "high"
    if n_appearances >= 14:
        return "good"
    if n_appearances >= 8:
        return "moderate"
    return "low (few appearances)"
