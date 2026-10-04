"""
Thin Streamlit caching layer around `data_pipeline`.

Kept separate from `data_pipeline.py` so that module stays importable and
testable without a Streamlit runtime. This file should contain no
analytical logic of its own -- only cache decorators.
"""

from __future__ import annotations

import streamlit as st

from . import data_pipeline as dp

DATA_PATH = "data/synthetic_player_data.csv"


@st.cache_resource(show_spinner="Fitting PCA, clustering, GMM and HMM on the synthetic cohort...")
def get_bundle() -> dict:
    """
    The full model bundle is computed once per app process and shared
    across all pages and users. Pages must treat the returned dataframe
    as read-only (filter with boolean masks / .copy(), never mutate
    columns on the shared object in place) since st.cache_resource does
    not copy its return value on each access the way st.cache_data does.
    """
    return dp.build_full_bundle(DATA_PATH)


@st.cache_data(show_spinner=False)
def bootstrap_player_ci(player_id: str, n_boot: int = 300) -> dict:
    """
    Cached per-player bootstrap CI for stability/hybridity. Cached on
    `player_id` and `n_boot` alone -- the dataset itself is immutable for
    the lifetime of the process (it comes from the cache_resource bundle
    above), so it does not need to be part of the cache key.
    """
    bundle = get_bundle()
    return dp.bootstrap_stability_hybridity_ci(bundle["df"], player_id, n_boot=n_boot)
