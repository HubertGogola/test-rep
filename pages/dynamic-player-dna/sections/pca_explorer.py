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
from src.stats_utils import cramers_v


def render():
    bundle = get_bundle()
    df = bundle["df"]
    pca_main = bundle["pca_main"]
    pca_combined = bundle["pca_combined"]
    pca_baseline = bundle["pca_baseline"]

    components.section_header(
        "PCA explorer",
        "Three PCA variants, mirroring the thesis structure: a main motor variant on contextually "
        "standardized features, a combined technical+motor variant, and a baseline comparison showing "
        "why contextual standardisation matters.",
    )

    tab_main, tab_combined, tab_context = st.tabs(["Motor PCA (main)", "Combined PCA", "Why context standardisation"])

    # ---------------------------------------------------------------------------
    with tab_main:
        evr = pca_main.explained_variance_ratio_
        cum = np.cumsum(evr)
        st.markdown("#### Explained variance")
        components.provenance_note("demo", "Recomputed live on the synthetic cohort.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PC1", f"{evr[0]*100:.1f}%")
        c2.metric("PC2", f"{evr[1]*100:.1f}%")
        c3.metric("PC3", f"{evr[2]*100:.1f}%")
        c4.metric("PC1\u2013PC3 cumulative", f"{cum[2]*100:.1f}%")
        with st.expander("Thesis benchmark (main motor PCA)"):
            components.provenance_note("thesis", "Reported exactly as in the thesis; not recomputed.")
            st.write(
                f"PC1 = {T.PCA_MOTOR_VARIANCE['PC1']}%, PC2 = {T.PCA_MOTOR_VARIANCE['PC2']}%, "
                f"PC3 = {T.PCA_MOTOR_VARIANCE['PC3']}%, cumulative = {T.PCA_MOTOR_CUMULATIVE_3}%."
            )

        left, right = st.columns([1, 1])
        with left:
            fig_scree = go.Figure()
            labels = [f"PC{i}" for i in range(1, len(evr) + 1)]
            fig_scree.add_trace(go.Bar(x=labels, y=evr * 100, name="Individual", marker_color=theme.ACCENT))
            fig_scree.add_trace(go.Scatter(x=labels, y=cum * 100, name="Cumulative", mode="lines+markers",
                                            yaxis="y2", line=dict(color=theme.THESIS_TONE)))
            fig_scree.update_layout(
                title="Scree plot", yaxis=dict(title="Individual variance (%)"),
                yaxis2=dict(title="Cumulative (%)", overlaying="y", side="right", range=[0, 105]),
            )
            theme.apply_plot_theme(fig_scree, height=380)
            st.plotly_chart(fig_scree, use_container_width=True)
        with right:
            loadings = pd.DataFrame(
                pca_main.components_[:3].T,
                index=[dp.MOTOR_LABELS[c.replace("_z", "")] for c in bundle["zcols"]],
                columns=["PC1", "PC2", "PC3"],
            )
            fig_load = px.imshow(loadings, text_auto=".2f", aspect="auto", color_continuous_scale=theme.PLOT_DIVERGING,
                                  zmin=-0.8, zmax=0.8, title="Loadings (synthetic demo)")
            theme.apply_plot_theme(fig_load, height=380)
            st.plotly_chart(fig_load, use_container_width=True)

        st.markdown("##### Score plot")
        color_by = st.selectbox("Colour by", ["squad", "role", "round", "kmeans_cluster"], key="pca_color")
        dim_mode = st.radio("Dimensions", ["2D", "3D"], horizontal=True, key="pca_dim")
        hover_cols = ["player_id", "appearance_no"]
        if dim_mode == "2D":
            fig_score = px.scatter(df, x="PC1", y="PC2", color=df[color_by].astype(str), hover_data=hover_cols,
                                    title="PC1 vs PC2")
        else:
            fig_score = px.scatter_3d(df, x="PC1", y="PC2", z="PC3", color=df[color_by].astype(str),
                                       hover_data=hover_cols, title="PC1 / PC2 / PC3")
        theme.apply_plot_theme(fig_score, height=520)
        st.plotly_chart(fig_score, use_container_width=True)

        st.markdown("##### Interpretation (synthetic demo components)")
        for i, col in enumerate(["PC1", "PC2", "PC3"]):
            top_idx = np.argsort(-np.abs(pca_main.components_[i]))[:3]
            zlabels = [dp.MOTOR_LABELS[bundle["zcols"][j].replace("_z", "")] for j in top_idx]
            signs = ["+" if pca_main.components_[i][j] > 0 else "\u2212" for j in top_idx]
            st.write(f"**{col}**: strongest loadings on " + ", ".join(f"{s}{l}" for s, l in zip(signs, zlabels)))
        with st.expander("Thesis interpretation of PC1\u2013PC3 (main motor PCA)"):
            components.provenance_note("thesis", "")
            for k, v in T.PCA_MOTOR_INTERPRETATION.items():
                st.write(f"**{k}**: {v}")

    # ---------------------------------------------------------------------------
    with tab_combined:
        st.markdown(
            "Adds the two aggregated technical indicators (offensive / defensive points) to the six "
            "motor variables, following the thesis's combined-data PCA variant."
        )
        evrc = pca_combined.explained_variance_ratio_
        cumc = np.cumsum(evrc)
        c1, c2 = st.columns(2)
        c1.metric("PC1\u2013PC4 cumulative (demo)", f"{cumc[3]*100:.1f}%")
        c2.metric("PC1\u2013PC5 cumulative (demo)", f"{cumc[4]*100:.1f}%")
        with st.expander("Thesis benchmark (combined PCA)"):
            components.provenance_note("thesis", "")
            st.write(f"PC1\u2013PC4 = {T.PCA_COMBINED_VARIANCE_4}%, PC1\u2013PC5 = {T.PCA_COMBINED_VARIANCE_5}%.")
            for k, v in T.PCA_COMBINED_INTERPRETATION.items():
                st.write(f"**{k}**: {v}")

        loadings_c = pd.DataFrame(
            pca_combined.components_[:5].T,
            index=[c.replace("_z", "").replace("_", " ") for c in bundle["combined_cols"]],
            columns=[f"PC{i}" for i in range(1, 6)],
        )
        fig_lc = px.imshow(loadings_c, text_auto=".2f", aspect="auto", color_continuous_scale=theme.PLOT_DIVERGING,
                            zmin=-0.8, zmax=0.8, title="Combined-PCA loadings (synthetic demo, first 5 components)")
        theme.apply_plot_theme(fig_lc, height=420)
        st.plotly_chart(fig_lc, use_container_width=True)
        st.caption(
            "offensive_index / defensive_index are synthetic, internally-consistent analogues of the "
            "thesis's two aggregated technical-tactical point indicators."
        )

    # ---------------------------------------------------------------------------
    with tab_context:
        st.markdown(
            "The thesis found that, without controlling for context, clustering mainly reproduced "
            "broad playing role; after standardising within round \u00d7 squad \u00d7 role, that "
            "association dropped sharply. The same comparison, recomputed on the synthetic cohort:"
        )
        X_main = df[["PC1", "PC2", "PC3"]].to_numpy()
        X_base = df[["PC1_baseline", "PC2_baseline", "PC3_baseline"]].to_numpy()
        _, labels_main = dp.fit_kmeans(X_main, k=2)
        _, labels_base = dp.fit_kmeans(X_base, k=4)
        v_main = cramers_v(df.role, pd.Series(labels_main, index=df.index))
        v_base = cramers_v(df.role, pd.Series(labels_base, index=df.index))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Main variant (context-standardized, k=2)**")
            st.metric("Cram\u00e9r's V vs. broad role", f"{v_main:.3f}")
            fig_m = px.scatter(df, x="PC1", y="PC2", color=df.role, title="Context-standardized score space")
            theme.apply_plot_theme(fig_m, height=380)
            st.plotly_chart(fig_m, use_container_width=True)
        with c2:
            st.markdown("**Baseline (no context control, k=4)**")
            st.metric("Cram\u00e9r's V vs. broad role", f"{v_base:.3f}")
            fig_b = px.scatter(df, x="PC1_baseline", y="PC2_baseline", color=df.role, title="Raw-standardized score space")
            theme.apply_plot_theme(fig_b, height=380)
            st.plotly_chart(fig_b, use_container_width=True)

        with st.expander("Thesis benchmark for this comparison"):
            components.provenance_note("thesis", "")
            st.write(
                f"Without context control, the best-separated baseline solution (k={T.KMEANS_BASELINE_NO_CONTEXT['best_k_by_silhouette']}) "
                f"had silhouette {T.KMEANS_BASELINE_NO_CONTEXT['best_silhouette']} and Cram\u00e9r's V "
                f"vs. role of {T.KMEANS_BASELINE_NO_CONTEXT['cramers_v_vs_role']}. "
                f"{T.PCA_MOTOR_CONTEXT_CRAMERS_V_NOTE}"
            )
        components.note(
            "Lower Cram\u00e9r's V in the main variant means cluster membership is less explained by "
            "broad playing role \u2014 i.e. the model is picking up individual variation rather than "
            "simply re-discovering defenders vs. midfielders vs. attackers.",
        )
