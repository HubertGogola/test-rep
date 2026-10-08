import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src import components
from sections import pca_explorer, clustering_lab, dynamic_dna, hmm_states, survival_analysis

components.page_header(
    "Modelling pipeline",
    "Modelling",
    "PCA, clustering, the Dynamic Player DNA trajectory, hidden Markov states, and survival "
    "analysis \u2014 the full analytical pipeline described in the thesis, each recomputed live "
    "on the synthetic cohort.",
    badges=["demo"],
)

tab_pca, tab_cluster, tab_dna, tab_hmm, tab_survival = st.tabs([
    "PCA Explorer", "Clustering Lab", "Dynamic Player DNA", "HMM & States", "Survival Analysis",
])

with tab_pca:
    pca_explorer.render()
with tab_cluster:
    clustering_lab.render()
with tab_dna:
    dynamic_dna.render()
with tab_hmm:
    hmm_states.render()
with tab_survival:
    survival_analysis.render()
