import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import theme, components, thesis_results as T
from src.cache import get_bundle


def render():
    bundle = get_bundle()
    df = bundle["df"]

    components.section_header(
        "About & research",
        "Thesis context, research questions, data provenance, and how to reach the underlying material.",
    )

    st.markdown("### Thesis context")
    st.markdown(
        "**Application of Selected Machine Learning Methods in Football Data Analysis** "
        "(Polish title: *Wykorzystanie wybranych metod uczenia maszynowego w analizie danych "
        "pi\u0142ki no\u017cnej*)\n\n"
        "Bachelor's thesis, AGH University of Krakow, Faculty of Management, 2026. "
        "Field of study: Computer Science and Econometrics. Supervisor: Dr Beata Basiura."
    )

    with st.expander("Research questions addressed by the thesis"):
        for q in T.RESEARCH_QUESTIONS:
            st.markdown(f"- {q}")

    with st.expander("Thesis study scope"):
        st.write(
            f"Squads: {', '.join(T.STUDY_SCOPE['squads'])}. Season: {T.STUDY_SCOPE['season']} "
            f"({' and '.join(T.STUDY_SCOPE['rounds'])} rounds). {T.STUDY_SCOPE['notes']}"
        )
        st.dataframe(
            pd.DataFrame(T.DATA_FUNNEL, columns=["Preparation stage", "Rows / observations"]),
            use_container_width=True, hide_index=True,
        )

    st.write("")
    st.markdown("### This application")
    st.write(
        "This application is an independent, public companion to the thesis above. It reimplements the "
        "same analytical pipeline (PCA, K-means, GMM, Dynamic Player DNA, HMM, Kaplan\u2013Meier) end to "
        "end on a fully synthetic dataset, so the methods can be explored interactively without "
        "publishing any real academy data."
    )

    with st.expander("How the synthetic dataset was built"):
        st.write(
            "A deterministic, seeded generator assigns each fictional player a role, squad category, "
            "and individual baseline, then simulates a hidden 3-state activity process across a "
            "chronological sequence of appearances. Raw GPS totals are drawn conditional on role, "
            "hidden state, round and the player's baseline; per-90 values and the Player Intensity "
            "Index are then derived from those raw totals using the same formulas described in the "
            "thesis, rather than sampled directly, so the dataset is internally consistent. Technical/"
            "tactical actions are generated from separate per-player latent skills, with a deliberately "
            "small coupling to the hidden motor state. A data-completeness pattern (technical detail "
            "more often missing in autumn than spring) is injected to mirror the real completeness "
            "issue documented in the thesis. The generator script is included in the project source "
            "(`src/generate_data.py`) so the whole process is auditable and reproducible."
        )
        components.download_csv_button(df.drop(columns=[c for c in df.columns if c.endswith("_baseline")]),
                                        "dynamic_player_dna_synthetic_dataset.csv",
                                        "Download the full synthetic dataset (CSV)")

    st.write("")
    st.markdown("### About the author")
    st.markdown(
        f'<div class="panel">'
        f'<b>Hubert Gogola</b><br>'
        f'<span style="color:{theme.TEXT_SECONDARY};">'
        f"Bachelor's degree \u2014 Computer Science and Econometrics, AGH University of Krakow<br>"
        f"MSc student \u2014 Computer Science and Econometrics, AGH University of Krakow "
        f"(in progress)<br><br>"
        f"Professional focus: football analytics, data analysis, machine learning, player profiling."
        f'</span></div>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("### Citation")
    st.code(
        "Gogola, H. (2026). Application of Selected Machine Learning Methods in Football Data Analysis "
        "[Bachelor's thesis]. AGH University of Krakow, Faculty of Management.",
        language="text",
    )

    st.write("")
    components.note(
        "No real player data, academy records, or anonymised transformations of real observations "
        "appear anywhere in this application. Aggregate thesis findings are reproduced exactly as "
        "published and are always labelled \u201cThesis result\u201d.",
        strong_prefix="Privacy",
    )
