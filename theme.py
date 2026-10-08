"""
Visual design tokens and shared chrome for the application.

Design direction: dark, restrained, editorial. No emoji, no decorative
icons, no glassmorphism, no neon gradients. A single muted accent colour
used sparingly. Generous spacing, strong typographic hierarchy, large
legible numbers.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Colour tokens
# ---------------------------------------------------------------------------

BG_PRIMARY = "#0A0F0D"
BG_PANEL = "#101613"
BG_PANEL_RAISED = "#141C18"
BORDER = "rgba(255,255,255,0.08)"
BORDER_STRONG = "rgba(255,255,255,0.14)"

TEXT_PRIMARY = "#EDEFEC"
TEXT_SECONDARY = "#9AA39C"
TEXT_MUTED = "#6E756F"

ACCENT = "#4F9D6E"
ACCENT_SOFT = "rgba(79,157,110,0.14)"
ACCENT_BORDER = "rgba(79,157,110,0.35)"

THESIS_TONE = "#B98A4A"
THESIS_SOFT = "rgba(185,138,74,0.14)"
THESIS_BORDER = "rgba(185,138,74,0.38)"

DEMO_TONE = ACCENT
DEMO_SOFT = ACCENT_SOFT
DEMO_BORDER = ACCENT_BORDER

WARNING_TONE = "#C2694F"

PLOT_PALETTE = ["#4F9D6E", "#B98A4A", "#6E8FB0", "#9AA39C", "#C2694F", "#7C6FA8"]
PLOT_SEQUENTIAL_GREEN = ["#0F1A15", "#1B3327", "#2A5340", "#3C7857", "#4F9D6E", "#7FC79B", "#B6E4C8"]
PLOT_DIVERGING = ["#C2694F", "#D9A98F", "#EDEFEC", "#8FB7A0", "#4F9D6E"]

FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


def inject_css():
    st.markdown(f"""
<style>
.stApp {{
    background: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    font-family: {FONT_STACK};
}}
.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 3.5rem;
    max-width: 1500px;
}}
[data-testid="stSidebar"] {{
    background: {BG_PANEL};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebarNav"] {{
    padding-top: 2.2rem;
}}
[data-testid="stSidebarNav"] ul {{
    padding-top: 0.4rem;
}}
footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}

h1, h2, h3, h4 {{
    font-family: {FONT_STACK};
    letter-spacing: -0.01em;
    color: {TEXT_PRIMARY};
}}
h1 {{ font-weight: 700; }}
h2 {{ font-weight: 650; }}
h3 {{ font-weight: 600; }}

p, label, span {{ color: {TEXT_PRIMARY}; }}

.eyebrow {{
    color: {TEXT_SECONDARY};
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-size: 0.72rem;
}}

.app-title {{
    font-size: 2.6rem;
    font-weight: 750;
    line-height: 1.04;
    margin: 0.35rem 0 0.6rem 0;
    letter-spacing: -0.015em;
}}

.app-subtitle {{
    color: {TEXT_SECONDARY};
    font-size: 1.04rem;
    line-height: 1.6;
    max-width: 880px;
}}

hr, [data-testid="stDivider"] {{
    border-color: {BORDER} !important;
}}

/* --- Result provenance badges --- */
.badge-row {{ margin: 0.1rem 0 0.9rem 0; }}
.badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    border-radius: 4px;
    padding: 3px 9px 3px 8px;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-right: 0.5rem;
}}
.badge-thesis {{
    background: {THESIS_SOFT};
    border-left: 2px solid {THESIS_TONE};
    color: #E3C699;
}}
.badge-demo {{
    background: {DEMO_SOFT};
    border-left: 2px solid {DEMO_TONE};
    color: #B9E3C9;
}}

/* --- Panels / cards --- */
.panel {{
    border: 1px solid {BORDER};
    background: {BG_PANEL_RAISED};
    border-radius: 10px;
    padding: 18px 20px;
}}
.panel-tight {{
    border: 1px solid {BORDER};
    background: {BG_PANEL_RAISED};
    border-radius: 10px;
    padding: 12px 16px;
}}

.stat-block .stat-value {{
    font-size: 1.9rem;
    font-weight: 750;
    color: {TEXT_PRIMARY};
    line-height: 1.1;
}}
.stat-block .stat-label {{
    color: {TEXT_SECONDARY};
    font-size: 0.82rem;
    margin-top: 0.15rem;
}}

.note {{
    border: 1px solid {BORDER};
    border-left: 2px solid {TEXT_SECONDARY};
    border-radius: 6px;
    padding: 12px 15px;
    color: {TEXT_SECONDARY};
    font-size: 0.9rem;
    line-height: 1.55;
}}
.note b {{ color: {TEXT_PRIMARY}; }}

.limitation-item {{
    border-bottom: 1px solid {BORDER};
    padding: 10px 0;
    color: {TEXT_PRIMARY};
    font-size: 0.93rem;
    line-height: 1.55;
}}
.limitation-item:last-child {{ border-bottom: none; }}

/* --- Streamlit metric tweaks --- */
[data-testid="stMetric"] {{
    border: 1px solid {BORDER};
    background: {BG_PANEL_RAISED};
    border-radius: 10px;
    padding: 14px 16px;
}}
[data-testid="stMetricLabel"] {{ color: {TEXT_SECONDARY}; }}
[data-testid="stMetricValue"] {{ color: {TEXT_PRIMARY}; }}

[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{
    color: {TEXT_SECONDARY};
    font-weight: 600;
}}
.stTabs [aria-selected="true"] {{ color: {TEXT_PRIMARY}; }}

a {{ color: {ACCENT}; }}
</style>
""", unsafe_allow_html=True)


def badge(kind: str) -> str:
    """Returns an HTML snippet for a provenance badge: 'thesis' or 'demo'."""
    if kind == "thesis":
        return f'<span class="badge badge-thesis">Thesis result</span>'
    return f'<span class="badge badge-demo">Synthetic demo</span>'


def badge_row(kinds: list[str]) -> str:
    return '<div class="badge-row">' + "".join(badge(k) for k in kinds) + "</div>"


def apply_plot_theme(fig, height: int | None = None):
    """Shared Plotly layout for a consistent, restrained dark theme."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.015)",
        font=dict(color=TEXT_PRIMARY, family=FONT_STACK, size=13),
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SECONDARY)),
        colorway=PLOT_PALETTE,
        hoverlabel=dict(bgcolor=BG_PANEL_RAISED, font=dict(color=TEXT_PRIMARY)),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.10)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.10)")
    if height:
        fig.update_layout(height=height)
    return fig
