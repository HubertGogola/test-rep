import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src import theme, components, thesis_results as T


def render():
    components.section_header(
        "Methodology",
        "What each method does, what it needs as input, what it produces, how to interpret it, and "
        "where it breaks down. Written to be precise enough for a data analyst and legible to someone "
        "working in football.",
    )

    METHODS = [
        {
            "name": "PCA (Principal Component Analysis)",
            "purpose": "Reduce a set of correlated motor/technical variables to a small number of "
                       "uncorrelated components that capture most of the variation in the data.",
            "input": "Standardized numeric variables (here: contextually standardized per-90 motor "
                     "variables, or a combined technical+motor set).",
            "output": "A set of components (PC1, PC2, ...), each a linear combination of the input "
                      "variables, with an explained-variance share and a loading vector.",
            "interpretation": "A component is read from the variables with the largest absolute "
                               "loadings on it. PCA does not label players or assign quality \u2014 it "
                               "only re-expresses the same information in fewer, uncorrelated dimensions.",
            "limitations": "Components can mix several concepts together; interpretation always "
                           "requires inspecting the loadings, not just the component number. Explained "
                           "variance depends heavily on which variables are included.",
        },
        {
            "name": "K-means (hard clustering)",
            "purpose": "Partition observations into k groups that are compact and separated from each "
                       "other, as a simple baseline grouping.",
            "input": "Numeric features (here: the retained PCA components).",
            "output": "One cluster label per observation; cluster centroids.",
            "interpretation": "Clusters are described after the fact, from the average characteristics "
                               "of their members. Cluster membership describes profile similarity, not "
                               "sporting quality or ranking.",
            "limitations": "Every observation is forced into exactly one group, even when it genuinely "
                           "sits between two profiles. The number of clusters (k) must be chosen, not "
                           "discovered automatically.",
        },
        {
            "name": "Silhouette / Calinski-Harabasz / Davies-Bouldin",
            "purpose": "Quantify how well-separated a hard clustering solution is, to help compare "
                       "different values of k.",
            "input": "A fitted clustering solution and the feature space it was fit on.",
            "output": "A single score per candidate k (higher silhouette/Calinski-Harabasz is better "
                      "separation; lower Davies-Bouldin is better separation).",
            "interpretation": "These metrics score separation quality of a partition that already "
                               "exists \u2014 they do not create or validate the assignment itself, and "
                               "the single best score is not automatically the right choice: "
                               "interpretability and stability matter too.",
            "limitations": "Moderate values (e.g. silhouette around 0.3) are common and expected for "
                           "sporting data, where group boundaries are rarely sharp.",
        },
        {
            "name": "Adjusted Rand Index (ARI)",
            "purpose": "Measure agreement between two partitions of the same data (e.g. the same "
                       "clustering refit on a bootstrap resample, or two model initializations).",
            "input": "Two label assignments over the same (or resampled) observations.",
            "output": "A score where 1.0 means identical partitions and 0 means no better than chance "
                      "agreement.",
            "interpretation": "ARI is a repeatability / stability measure. ARI = 1.0 means two runs "
                               "agreed completely \u2014 it does not mean the partition reflects any "
                               "ground truth, since none exists for unsupervised clustering.",
            "limitations": "High ARI across *initializations* does not guard against the model being "
                           "wrong for the data \u2014 only against being unstable given the data.",
        },
        {
            "name": "Gaussian Mixture Model (GMM, soft clustering)",
            "purpose": "Describe each observation as a probabilistic mixture of several underlying "
                       "profiles, rather than assigning it to exactly one.",
            "input": "Numeric features (here: the retained PCA components).",
            "output": "Membership probabilities per observation (summing to 1 across profiles); "
                      "profile means and covariances.",
            "interpretation": "A single appearance can be, for example, 70% one profile and 30% "
                               "another. The profile definitions come from the data, not from "
                               "predetermined categories.",
            "limitations": "Profile count is chosen via AIC/BIC plus stability and interpretability, "
                           "not derived from a single automatic rule. Profiles describe motor-activity "
                           "level here, not a complete playing style.",
        },
        {
            "name": "Stability (Dynamic Player DNA)",
            "purpose": "Describe how repeatable a player's soft profile is from one appearance to the "
                       "next.",
            "input": "A chronological sequence of a player's GMM membership probabilities.",
            "output": "A single number in [0, 1] per player; higher means more repeatable.",
            "interpretation": "High stability means the profile looks similar match after match. It is "
                               "not a measure of player quality.",
            "limitations": (
                "The thesis describes stability conceptually (match-to-match profile similarity) but "
                "does not publish a closed-form equation. This application's documented "
                "implementation: stability = 1 \u2212 mean<sub>t</sub>( \u2016p<sub>t</sub> "
                "\u2212 p<sub>t-1</sub>\u2016\u2082 / \u221a2 ), where p<sub>t</sub> is the "
                "two-profile membership vector at appearance t. This is this application's own "
                "operationalisation for the synthetic demo, not a reproduction of an unpublished "
                "thesis formula."
            ),
        },
        {
            "name": "Hybridity (Dynamic Player DNA)",
            "purpose": "Describe how evenly a single appearance mixes the two motor-activity profiles.",
            "input": "GMM membership probabilities for one observation.",
            "output": "A single number in [0, 1]; 1 means an exact 50/50 split, 0 means fully one profile.",
            "interpretation": "High hybridity does not mean general footballing versatility \u2014 it is "
                               "specific to the two motor-activity profiles used here.",
            "limitations": "With exactly two profiles, hybridity and dominant-profile probability carry "
                           "closely related (though not identical) information.",
        },
        {
            "name": "Hidden Markov Model (HMM)",
            "purpose": "Identify a small number of statistical activity states from a chronological "
                       "sequence of appearances, and estimate the probability of moving between them.",
            "input": "A player's ordered sequence of PCA-reduced motor observations (PC1-PC3, the same "
                     "reduced space used for K-means and GMM) -- appearance order, not minutes within a match.",
            "output": "A most-likely state per appearance (Viterbi decoding), a transition matrix, and "
                      "state-conditional means.",
            "interpretation": "States are named only after fitting, by overall activity level (Lower / "
                               "Baseline / Elevated) \u2014 never as subjective form.",
            "limitations": "Treated as experimental in the thesis given limited sequence lengths per "
                           "player; a transition is only identifiable as having occurred somewhere "
                           "between two consecutive appearances, not at an exact moment in time.",
        },
        {
            "name": "Kaplan\u2013Meier survival analysis",
            "purpose": "Estimate the distribution of appearances until a specific event, while "
                       "correctly handling players for whom the event had not yet happened by the end "
                       "of observation.",
            "input": "A time (here: appearance number) and an event/censoring indicator per player.",
            "output": "A survival curve, a median time-to-event, and a confidence interval.",
            "interpretation": "The median describes the group, not any individual player \u2014 it is "
                               "not true that every player transitions at exactly the median appearance "
                               "number.",
            "limitations": "Right-censored players (the event had not yet occurred) are retained in the "
                           "analysis rather than dropped or treated as missing data; dropping them would "
                           "bias the estimate.",
        },
    ]

    for m in METHODS:
        with st.expander(m["name"]):
            st.markdown(f"**Purpose.** {m['purpose']}")
            st.markdown(f"**Input.** {m['input']}")
            st.markdown(f"**Output.** {m['output']}")
            st.markdown(f"**Interpretation.** {m['interpretation']}")
            st.markdown(f"**Limitations.** {m['limitations']}", unsafe_allow_html=True)

    st.write("")
    st.markdown("### Player Intensity Index (PII)")
    st.code(T.PII_DEFINITION["formula"], language="text")
    components.note(T.PII_DEFINITION["note"], strong_prefix="Important")
    components.note(
        "The formula combines raw (not per-90) match totals in the numerator with an already-"
        "per-minute pace in the denominator. A consequence: for two appearances with <i>identical</i> "
        "per-90 motor rates, the one with fewer minutes played will show a <i>lower</i> PII under this "
        "formula \u2014 PII is not purely rate-invariant. This is a property of the formula as "
        "published, not an artefact of this application's synthetic data generator, which reproduces "
        "the formula exactly from each appearance's raw totals and minutes played. Separately, because "
        "total distance sits in the denominator, PII has a built-in tendency toward a negative "
        "association with raw distance; whether that shows up as a strong or a near-zero empirical "
        "correlation in a given dataset depends on how correlated distance happens to be with the "
        "formula's numerator terms (HSR, sprint, accelerations, decelerations) in that dataset. The "
        "thesis's cohort and this application's synthetic cohort are independent datasets and are not "
        "expected to show the same correlation strength.",
        strong_prefix="A non-obvious property worth knowing",
    )

    st.write("")
    st.markdown("### Thesis scope")
    st.write(
        "Dynamic Player DNA is an analytical pipeline combining existing, established methods "
        "(PCA, K-means, GMM, HMM, Kaplan\u2013Meier). It is not presented as a new machine-learning "
        "algorithm, and it is not intended to automatically rank talent or replace coaching judgement."
    )

    st.write("")
    st.markdown("### Thesis limitations")
    for item in T.LIMITATIONS:
        st.markdown(f'<div class="limitation-item">{item}</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("### Further work proposed in the thesis")
    for item in T.FURTHER_WORK:
        st.markdown(f"- {item}")

    st.write("")
    st.markdown("### Software")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Used in the original thesis**")
        for s in T.THESIS_SOFTWARE:
            st.markdown(f"- {s}")
    with c2:
        st.markdown("**Used in this application**")
        for s in ["Python", "pandas and NumPy", "SciPy (statistical tests)", "scikit-learn (PCA, K-means, GMM)",
                  "A from-scratch Gaussian HMM (Baum-Welch EM, Viterbi decoding, implemented for this app)",
                  "Plotly (interactive visualisation)", "Streamlit (application framework)"]:
            st.markdown(f"- {s}")

    st.write("")
    st.markdown("### Data privacy")
    st.write(
        "Every player-level observation in this application is synthetic, generated by a documented, "
        "seeded process (see the README for details). No real academy data, real player identities, or "
        "anonymised transformations of real observations are published here. Aggregate findings "
        "reproduced from the thesis are always labelled \u201cThesis result\u201d and are never blended "
        "with synthetic-demo computations."
    )
