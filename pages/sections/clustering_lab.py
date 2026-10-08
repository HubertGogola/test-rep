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
from src.stats_utils import kmeans_quality_curve, bootstrap_kmeans_stability, cramers_v


def render():
    bundle = get_bundle()
    df = bundle["df"]

    components.section_header(
        "Clustering lab",
        "Compare hard clustering (K-means) against soft clustering (a Gaussian Mixture Model), and see "
        "how contextual standardisation changes what the clusters actually represent.",
    )

    components.note(
        "<b>Hard clustering (K-means)</b> assigns every observation to exactly one group. "
        "<b>Soft clustering (GMM)</b> assigns membership probabilities across groups that sum to 1 for "
        "each observation. Silhouette, Calinski-Harabasz and Davies-Bouldin evaluate how well-separated "
        "a hard partition is \u2014 they do not create the assignment, and a higher silhouette does not "
        "automatically mean a 'better' number of groups; interpretability and stability matter too."
    )

    variant = st.radio("Feature space", ["Context-standardized (main)", "Raw baseline (no context control)"],
                        horizontal=True, key="clustering_variant")
    model_choice = st.radio("Model", ["K-means (hard)", "Gaussian Mixture Model (soft)"], horizontal=True,
                             key="clustering_model_choice")

    if variant.startswith("Context"):
        X = df[["PC1", "PC2", "PC3"]].to_numpy()
    else:
        X = df[["PC1_baseline", "PC2_baseline", "PC3_baseline"]].to_numpy()

    st.write("")

    if model_choice.startswith("K-means"):
        st.markdown("### K-means quality across k")
        quality = kmeans_quality_curve(X, range(2, 7))
        left, right = st.columns([1.3, 1])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=quality.k, y=quality.silhouette, mode="lines+markers", name="Silhouette"))
            fig.update_layout(title="Silhouette by k (synthetic demo)", xaxis_title="k", yaxis_title="Silhouette")
            theme.apply_plot_theme(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.dataframe(quality.round(3), use_container_width=True, hide_index=True)
        with st.expander("Thesis benchmark: K-means quality by k"):
            components.provenance_note("thesis", "")
            thesis_tbl = pd.DataFrame(T.KMEANS_QUALITY_BY_K,
                                       columns=["k", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin",
                                                "Smallest cluster share (%)"])
            st.dataframe(thesis_tbl, use_container_width=True, hide_index=True)
            st.caption(f"The thesis retained k={T.KMEANS_CHOSEN_K} for the main variant.")

        k = st.slider("Number of clusters (k)", 2, 6, 2, key="clustering_kmeans_k")
        km, labels = dp.fit_kmeans(X, k=k)
        df_view = df.copy()
        df_view["cluster"] = [f"Cluster {c+1}" for c in labels]

        st.write("")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Silhouette (k={})".format(k), f"{quality[quality.k==k].silhouette.iloc[0]:.3f}" if k in quality.k.values else "n/a")
        with c2:
            v = cramers_v(df.role, pd.Series(labels, index=df.index))
            st.metric("Cram\u00e9r's V vs. broad role", f"{v:.3f}")
        with c3:
            boot = bootstrap_kmeans_stability(X, k=k, n_boot=100)
            st.metric("Bootstrap mean ARI", f"{boot['mean_ari']:.3f}")

        left2, right2 = st.columns([1.2, 1])
        with left2:
            fig_sc = px.scatter(df_view, x="PC1" if variant.startswith("Context") else "PC1_baseline",
                                 y="PC2" if variant.startswith("Context") else "PC2_baseline",
                                 color="cluster", hover_data=["player_id", "role"], title=f"K-means, k={k}")
            theme.apply_plot_theme(fig_sc, height=420)
            st.plotly_chart(fig_sc, use_container_width=True)
        with right2:
            profile = df_view.groupby("cluster")[dp.MOTOR_COLS].mean().round(1)
            profile.columns = [dp.MOTOR_LABELS[c] for c in profile.columns]
            sizes = df_view["cluster"].value_counts()
            profile.insert(0, "n", sizes)
            st.dataframe(profile, use_container_width=True)

        if k == 2:
            with st.expander("Thesis benchmark: K-means cluster profiles (k=2, main variant)"):
                components.provenance_note("thesis", "")
                thesis_profiles = pd.DataFrame(T.KMEANS_CLUSTER_PROFILES).T
                thesis_profiles = thesis_profiles[["label", "n_profiles", "distance_per90", "hsr_per90",
                                                    "sprint_per90", "acc_per90", "dec_per90", "pii"]]
                thesis_profiles.columns = ["Label", "n", "Distance/90", "HSR/90", "Sprint/90",
                                           "ACC/90", "DEC/90", "PII"]
                st.dataframe(thesis_profiles, use_container_width=True)
                st.caption(
                    "Cluster numbering is arbitrary and independent between this synthetic demo and the "
                    "thesis \u2014 compare the two tables by their motor-variable profile, not by matching "
                    "cluster numbers."
                )

    else:
        st.markdown("### GMM model selection across number of profiles")
        rows = []
        for kk in range(2, 6):
            gmm_k, _, _ = dp.fit_gmm(X, k=kk)
            rows.append({"profiles": kk, "AIC": gmm_k.aic(X), "BIC": gmm_k.bic(X)})
        sel = pd.DataFrame(rows)
        left, right = st.columns([1.3, 1])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sel.profiles, y=sel.AIC, mode="lines+markers", name="AIC"))
            fig.add_trace(go.Scatter(x=sel.profiles, y=sel.BIC, mode="lines+markers", name="BIC"))
            fig.update_layout(title="AIC / BIC by number of profiles (synthetic demo)", xaxis_title="Profiles")
            theme.apply_plot_theme(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.dataframe(sel.round(1), use_container_width=True, hide_index=True)
        with st.expander("Thesis benchmark: GMM model selection"):
            components.provenance_note("thesis", "")
            thesis_gmm = pd.DataFrame(T.GMM_MODEL_SELECTION, columns=["Profiles", "AIC", "BIC", "Stability (ARI)",
                                                                        "Smallest profile share (%)"])
            st.dataframe(thesis_gmm, use_container_width=True, hide_index=True)
            st.caption(f"BIC selected {T.GMM_CHOSEN_K} profiles in the thesis.")

        k = st.slider("Number of profiles", 2, 5, 2, key="gmm_k")
        gmm, labels, probs = dp.fit_gmm(X, k=k)
        confidence = probs.max(axis=1)
        df_view = df.copy()
        df_view["profile"] = [f"Profile {c+1}" for c in labels]
        df_view["confidence"] = confidence

        c1, c2 = st.columns(2)
        c1.metric("AIC", f"{gmm.aic(X):,.0f}")
        c2.metric("Mean membership confidence", f"{confidence.mean():.3f}")

        left2, right2 = st.columns([1.2, 1])
        with left2:
            fig_sc = px.scatter(df_view, x="PC1" if variant.startswith("Context") else "PC1_baseline",
                                 y="PC2" if variant.startswith("Context") else "PC2_baseline",
                                 color="profile", size="confidence", hover_data=["player_id", "role"],
                                 title=f"GMM, {k} profiles")
            theme.apply_plot_theme(fig_sc, height=420)
            st.plotly_chart(fig_sc, use_container_width=True)
        with right2:
            fig_hist = px.histogram(df_view, x="confidence", nbins=25, title="Membership confidence distribution")
            theme.apply_plot_theme(fig_hist, height=420)
            st.plotly_chart(fig_hist, use_container_width=True)

    st.write("")
    st.caption(
        "Note: silhouette, AIC and BIC never assign a cluster or profile by themselves \u2014 they only "
        "score a partition that an algorithm already produced. Neither metric implies ground truth, and "
        "unsupervised results here are not evaluated with accuracy, precision or recall, since no "
        "reference labels exist."
    )
