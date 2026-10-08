import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src import components
from sections import methodology, about_research

components.page_header(
    "Reference",
    "Reference",
    "What each method does and why, the thesis's own limitations, and the research context behind "
    "this application.",
)

tab_method, tab_about = st.tabs(["Methodology", "About & Research"])

with tab_method:
    methodology.render()
with tab_about:
    about_research.render()
