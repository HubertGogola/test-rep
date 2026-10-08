import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src import theme, components, thesis_results as T
from src.cache import get_bundle

bundle = get_bundle()
df = bundle["df"]

st.markdown('<div class="eyebrow">Football analytics \u00b7 Machine learning \u00b7 Dynamic profiling</div>',
            unsafe_allow_html=True)
st.markdown('<div class="app-title">Dynamic Player DNA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Dynamic football player profiling through unsupervised learning and '
    'sequence modelling &mdash; from static player profiles to match-to-match behavioural trajectories.</div>',
    unsafe_allow_html=True,
)
st.write("")

components.note(
    "This application is a public, interactive companion to a bachelor's thesis on football player "
    "profiling. Every player, match, and observation visible in this app is fictional and generated "
    "by a documented synthetic process &mdash; no real academy data is published here. Aggregate "
    "research findings from the thesis are shown separately and labelled <b>Thesis result</b>; "
    "everything computed live on the synthetic cohort is labelled <b>Synthetic demo</b>.",
    strong_prefix="Data privacy",
)

st.write("")
st.markdown("### What this is")
left, right = st.columns([1.3, 1])
with left:
    st.markdown(
        "Traditional player profiling asks a single question: **what type of player is this?** "
        "It aggregates a season into one static snapshot and assigns a label.\n\n"
        "Dynamic Player DNA adds a second question: **how does that profile change from match to "
        "match?** Instead of one static label, each appearance is described by soft membership "
        "probabilities across motor-activity profiles. Tracking those probabilities across "
        "consecutive appearances makes it possible to ask whether a player's profile is stable or "
        "variable, whether it mixes two profiles evenly or leans on one, and whether a sequence of "
        "matches shows a detectable shift between statistical activity states."
    )
with right:
    components.note(
        "Traditional clustering: <i>which profile does this observation resemble?</i><br><br>"
        "Dynamic Player DNA: <i>how stable is that profile, how hybrid is it, and how does it "
        "evolve across appearances?</i>",
        strong_prefix="The core idea",
    )

st.write("")
st.markdown("### Analytical pipeline")
pipeline = [
    ("01", "Data preparation", "Cleaning, contextual standardisation, per-90 conversion"),
    ("02", "PCA", "Reduce correlated motor variables to interpretable components"),
    ("03", "K-means", "Hard clustering baseline on the retained components"),
    ("04", "GMM", "Soft profile membership probabilities per appearance"),
    ("05", "Dynamic Player DNA", "Match-to-match stability and hybridity"),
    ("06", "HMM", "Hidden activity states and transition probabilities"),
    ("07", "Kaplan\u2013Meier", "Time to first transition between states"),
]
def _render_pipeline_row(items):
    cols = st.columns(len(items))
    for col, (n, label, desc) in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="panel" style="min-height:140px;">'
                f'<div style="color:{theme.ACCENT};font-weight:800;font-size:1.3rem;">{n}</div>'
                f'<div style="font-weight:700;margin:6px 0 6px 0;">{label}</div>'
                f'<div style="color:{theme.TEXT_SECONDARY};font-size:0.85rem;line-height:1.45;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# Two rows (4 + 3) rather than all 7 in one row -- keeps each card legible
# on a 1366px laptop instead of cramming seven narrow columns side by side.
_render_pipeline_row(pipeline[:4])
st.write("")
_render_pipeline_row(pipeline[4:])

st.write("")
st.markdown("### Research headline numbers")
components.provenance_note("thesis", "Reproduced exactly as reported in the source thesis.")
r1 = st.columns(5)
with r1[0]:
    components.stat_block("Motor observations (thesis)", f"{T.ANALYTICAL_DATASETS['motor']['observations']}")
with r1[1]:
    components.stat_block("Players (thesis)", f"{T.ANALYTICAL_DATASETS['motor']['players']}")
with r1[2]:
    components.stat_block("PC1\u2013PC3 variance", f"{T.PCA_MOTOR_CUMULATIVE_3:.2f}%")
with r1[3]:
    components.stat_block("GMM profiles (BIC-selected)", f"{T.GMM_CHOSEN_K}")
with r1[4]:
    components.stat_block("HMM states (BIC-selected)", f"{T.HMM_CHOSEN_STATES}")

r2 = st.columns(5)
with r2[0]:
    components.stat_block("K-means silhouette (k=2)", f"{T.KMEANS_QUALITY_BY_K[0][1]:.3f}")
with r2[1]:
    components.stat_block("Mean Dynamic DNA stability", f"{T.DYNAMIC_DNA_COHORT['mean_stability']:.3f}")
with r2[2]:
    components.stat_block("Mean hybridity", f"{T.DYNAMIC_DNA_COHORT['mean_hybridity']:.3f}")
with r2[3]:
    components.stat_block("Baseline-state persistence", "88.7%")
with r2[4]:
    components.stat_block("Median appearances to transition", f"{T.SURVIVAL_RESULTS['median_appearances']}")

st.write("")
st.markdown("### Synthetic demo cohort (this application)")
components.provenance_note("demo", "Computed live from the fictional dataset bundled with this app.")
s1 = st.columns(5)
with s1[0]:
    components.stat_block("Synthetic observations", f"{len(df)}")
with s1[1]:
    components.stat_block("Fictional players", f"{df.player_id.nunique()}")
with s1[2]:
    components.stat_block("Squads", ", ".join(sorted(df.squad.unique())))
with s1[3]:
    components.stat_block("Rounds covered", ", ".join(sorted(df['round'].unique())))
with s1[4]:
    components.stat_block("Median appearances / player", f"{int(df.groupby('player_id').size().median())}")

st.write("")
st.markdown("### Methods used in this pipeline")
pills = ["PCA", "K-means", "Gaussian Mixture Model", "Stability & hybridity",
         "Hidden Markov Model", "Kaplan\u2013Meier survival analysis"]
st.markdown(
    " &nbsp; ".join(
        f'<span style="border:1px solid {theme.BORDER_STRONG};border-radius:6px;padding:4px 10px;'
        f'font-size:0.8rem;color:{theme.TEXT_SECONDARY};">{p}</span>' for p in pills
    ),
    unsafe_allow_html=True,
)

st.write("")
st.markdown("### Where to go next")
nav_cols = st.columns(3)
with nav_cols[0]:
    st.markdown(
        '<div class="panel"><b>Start with a player</b><br><span style="font-size:0.88rem;color:'
        f'{theme.TEXT_SECONDARY};">Open <i>Player DNA</i> to see a full fictional profile, or '
        '<i>Player Comparison</i> to set two players side by side.</span></div>',
        unsafe_allow_html=True,
    )
with nav_cols[1]:
    st.markdown(
        '<div class="panel"><b>Explore the models</b><br><span style="font-size:0.88rem;color:'
        f'{theme.TEXT_SECONDARY};"><i>PCA Explorer</i>, <i>Clustering Lab</i> and <i>HMM & State '
        'Transitions</i> let you refit each model and compare it with the thesis benchmark.</span></div>',
        unsafe_allow_html=True,
    )
with nav_cols[2]:
    st.markdown(
        '<div class="panel"><b>Read the method</b><br><span style="font-size:0.88rem;color:'
        f'{theme.TEXT_SECONDARY};"><i>Methodology</i> explains every method in plain terms; '
        '<i>About & Research</i> covers the thesis context and privacy approach.</span></div>',
        unsafe_allow_html=True,
    )
