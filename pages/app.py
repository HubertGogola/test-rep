"""
Dynamic Player DNA -- entry point and navigation router.

This file intentionally contains no page content of its own (besides
global chrome). Each page lives in `pages/` as a plain script; this file
only sets global configuration and defines navigation order and grouping.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src import theme, components

st.set_page_config(
    page_title="Dynamic Player DNA",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject_css()
components.sidebar_chrome()

pages = {
    "Start": [
        st.Page("pages/overview.py", title="Overview", default=True),
    ],
    "Player analysis": [
        st.Page("pages/player_dna.py", title="Player DNA"),
        st.Page("pages/player_comparison.py", title="Player comparison"),
        st.Page("pages/cohort_analysis.py", title="Cohort analysis"),
    ],
    "Modelling & reference": [
        st.Page("pages/modelling.py", title="Modelling"),
        st.Page("pages/reference.py", title="Reference"),
    ],
}

nav = st.navigation(pages)
nav.run()

components.sidebar_footer_note()
