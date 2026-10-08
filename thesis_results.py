"""
Frozen results from the source thesis.

Hubert Gogola, "Wykorzystanie wybranych metod uczenia maszynowego w analizie
danych pi\u0142ki no\u017cnej" / "Application of Selected Machine Learning
Methods in Football Data Analysis", AGH University of Krakow, 2026.

Every number in this module is transcribed directly from the thesis text,
tables, and figures. Nothing here is recomputed, estimated, or adjusted.

This module must never import streamlit, numpy, or any synthetic-data
pipeline. It exists so that a thesis figure can be displayed with total
confidence that it has not been accidentally blended with a live
recomputation on the synthetic demo dataset.

If a number is not explicitly present in the thesis, it is not included
here -- the application must not invent or back-fill thesis results.
"""

from dataclasses import dataclass, field


SOURCE_CITATION = (
    "Hubert Gogola, \u201cApplication of Selected Machine Learning Methods in "
    "Football Data Analysis\u201d, bachelor's thesis, AGH University of Krakow, "
    "Faculty of Management, 2026. Supervisor: Dr Beata Basiura."
)

# ---------------------------------------------------------------------------
# 3.1 -- Study scope
# ---------------------------------------------------------------------------

STUDY_SCOPE = {
    "squads": ["CII", "U17", "U19"],
    "season": "2025/2026",
    "rounds": ["Autumn", "Spring"],
    "notes": (
        "Internal data of one football academy. Goalkeepers were excluded "
        "from the main analysis. Detailed positions were grouped into three "
        "broad roles: defence, midfield, attack."
    ),
}

# ---------------------------------------------------------------------------
# 3.2 -- Data preparation funnel (Table 8)
# ---------------------------------------------------------------------------

DATA_FUNNEL = [
    ("Source rows -- technical/tactical", 1412),
    ("Valid appearances -- technical/tactical", 730),
    ("Source rows -- GPS", 1690),
    ("Valid observations -- GPS", 840),
    ("Strict technical/GPS matches", 551),
    ("Observations in the motor dataset", 799),
    ("Observations in the combined dataset", 523),
    ("Observations in the detailed technical dataset", 260),
    ("Matches excluded (team-label mismatch)", 23),
]

ANALYTICAL_DATASETS = {
    "motor": {
        "observations": 799,
        "players": 37,
        "use": "Motor profiling, GMM, Dynamic Player DNA and HMM (primary dataset).",
    },
    "combined": {
        "observations": 523,
        "players": 37,
        "use": "Integration of technical indicators with GPS variables.",
    },
    "technical_detailed": {
        "observations": 260,
        "players": 33,
        "use": "Exploratory analysis of technical/tactical actions.",
    },
}

MEDIAN_APPEARANCES_PER_PLAYER = 24
MINUTES_THRESHOLD = 20

# ---------------------------------------------------------------------------
# 3.3 -- Correlation structure of motor variables
# ---------------------------------------------------------------------------

MOTOR_CORRELATIONS = [
    ("HSR/90", "Sprint distance/90", 0.736),
    ("Accelerations/90", "Decelerations/90", 0.699),
    ("HSR/90", "Decelerations/90", 0.535),
    ("Total distance/90", "Decelerations/90", 0.522),
    ("PII", "Total distance/90", -0.071),
]

# ---------------------------------------------------------------------------
# 3.4 -- PCA, motor dataset (main / context-standardized variant)
# ---------------------------------------------------------------------------

PCA_MOTOR_VARIANCE = {
    "PC1": 50.94,
    "PC2": 19.07,
    "PC3": 13.79,
    "PC4": 8.33,
    "PC5": 4.83,
    "PC6": 3.04,
}
PCA_MOTOR_CUMULATIVE_3 = 83.80

PCA_MOTOR_LOADINGS = {
    # variable: (PC1, PC2, PC3)
    "Distance/90": (0.349, 0.498, -0.385),
    "HSR/90": (0.499, -0.175, -0.327),
    "Sprint/90": (0.426, -0.488, -0.240),
    "Accelerations/90": (0.333, 0.530, 0.525),
    "Decelerations/90": (0.487, 0.162, 0.046),
    "PII": (0.317, -0.420, 0.639),
}

PCA_MOTOR_INTERPRETATION = {
    "PC1": "General high-intensity motor-activity dimension (highest positive loadings on HSR, decelerations and sprint).",
    "PC2": "Contrast between work volume / frequent tempo changes (positive: accelerations, distance) and a more explosive profile (negative: sprint, PII).",
    "PC3": "Relative intensity independent of raw volume (positive: PII, accelerations; negative: distance, HSR).",
}

PCA_MOTOR_NO_CONTEXT_CUMULATIVE_2 = 82.87
PCA_MOTOR_NO_CONTEXT_CRAMERS_V = 0.608
PCA_MOTOR_CONTEXT_CRAMERS_V_NOTE = (
    "After contextual standardisation (round, squad category, broad role) "
    "the association between cluster membership and broad role fell to "
    "practically zero."
)

PCA_COMBINED_VARIANCE_4 = 79.27
PCA_COMBINED_VARIANCE_5 = 87.81
PCA_COMBINED_INTERPRETATION = {
    "PC1": "Motor activity (highest loadings: decelerations 0.491, HSR 0.446, sprint 0.426).",
    "PC2": "Defensive technical points (0.684) together with distance (0.443) and accelerations (0.370), contrasted with sprint (-0.414).",
    "PC3": "Dominated by offensive technical points (0.900) -- an offensive-activity dimension.",
    "PC4": "Contrast between tempo/offensive action (accelerations 0.622, offensive points 0.405) and HSR / defensive points (both negative).",
    "PC5": "Almost entirely dominated by PII (0.903) -- a standalone relative-intensity dimension.",
}

PCA_TECHNICAL_CUMULATIVE_6 = 85.86
PCA_TECHNICAL_NOTE = (
    "Based on only 260 observations with a more complex loading structure; "
    "treated in the thesis as exploratory rather than a confirmatory model."
)
PCA_TECHNICAL_INTERPRETATION = {
    "PC1": "Individual play and creation (individual actions 0.498, key passes 0.463, assists 0.404).",
    "PC2": "Finishing vs. duel-based contrast (goals 0.534, finishing actions 0.413 positive; box duels won -0.444, 1v1 defensive duels lost -0.425 negative).",
    "PC3": "Chance creation (chance assists 0.655 positive; box duels lost -0.397, interventions -0.347 negative).",
    "PC4": "Contrast between box duels lost (0.739) and finishing (-0.590).",
    "PC5": "Mixed structure (box duels won 0.494 positive; interventions -0.399, 1v1 defensive duels lost -0.372 negative) -- less clear-cut sporting interpretation.",
    "PC6": "Creative vs. defensive contrast (assists 0.714 positive; interventions -0.557 negative).",
}

# ---------------------------------------------------------------------------
# 3.4 -- K-means (motor dataset, context-standardized, main variant)
# ---------------------------------------------------------------------------

KMEANS_QUALITY_BY_K = [
    # k, silhouette, calinski_harabasz, davies_bouldin, smallest_cluster_share_pct
    (2, 0.335, 47.88, 1.122, 43.1),
    (3, 0.269, 39.92, 1.122, 26.4),
    (4, 0.294, 38.76, 1.132, 13.9),
    (5, 0.316, 38.24, 0.974, 8.3),
    (6, 0.272, 35.87, 1.053, 9.7),
]
KMEANS_CHOSEN_K = 2

KMEANS_BASELINE_NO_CONTEXT = {
    "best_k_by_silhouette": 4,
    "best_silhouette": 0.461,
    "cramers_v_vs_role": 0.608,
    "note": (
        "Without contextual standardisation, the best-separated solution "
        "(k=4) largely reproduced broad playing role rather than individual "
        "variation."
    ),
}

KMEANS_BOOTSTRAP = {
    "resamples": 150,
    "mean_ari": 0.755,
    "median_ari": 0.763,
    "ari_ci95": (0.426, 0.960),
    "jaccard_cluster_1": 0.886,
    "jaccard_cluster_2": 0.867,
}

KMEANS_CLUSTER_PROFILES = {
    1: {
        "n_profiles": 41,
        "distance_per90": 10635,
        "hsr_per90": 594,
        "sprint_per90": 111,
        "acc_per90": 116,
        "dec_per90": 110,
        "pii": 36.96,
        "label": "Lower motor-activity level",
    },
    2: {
        "n_profiles": 31,
        "distance_per90": 11099,
        "hsr_per90": 776,
        "sprint_per90": 168,
        "acc_per90": 127,
        "dec_per90": 126,
        "pii": 44.98,
        "label": "Higher motor-activity level",
    },
}

# ---------------------------------------------------------------------------
# 3.4 -- Similarity model, aggregate characterisation only.
#
# The thesis's Table 13 lists individual example pairs identified by their
# pseudonymised player codes (e.g. "P003", "P009") together with their
# individual similarity scores. Those codes still identify specific real
# (pseudonymised) individuals from the source academy dataset, so they are
# intentionally NOT reproduced here, even in aggregate-looking form -- this
# module only carries genuinely aggregate, non-individual statistics.
# ---------------------------------------------------------------------------

SIMILARITY_RESULT_RANGE = {
    "cosine_similarity_min_observed": 0.614,
    "cosine_similarity_max_observed": 0.904,
    "note": (
        "The thesis illustrates its similarity model with individual example pairs "
        "(identified by pseudonymised player codes) across several round/category/role "
        "groups, with cosine similarity ranging from about 0.61 to 0.90 among the examples "
        "shown. Those individual pairings are not reproduced here to avoid publishing any "
        "individual-level (even pseudonymised) result from the source dataset."
    ),
}

# ---------------------------------------------------------------------------
# 3.4 -- GMM model selection (Table 14) and profiles (Table 15)
# ---------------------------------------------------------------------------

GMM_MODEL_SELECTION = [
    # n_profiles, AIC, BIC, stability_ari, smallest_profile_share_pct
    (2, 7547.30, 7636.28, 1.000, 24.9),
    (3, 7547.06, 7682.88, 0.967, 23.7),
    (4, 7548.14, 7730.79, 0.861, 12.8),
    (5, 7549.75, 7779.24, 0.679, 8.1),
]
GMM_CHOSEN_K = 2
GMM_OBSERVATIONS = 799

GMM_PROFILES = {
    "lower_activity": {
        "label": "Lower motor activity",
        "effective_n": 553,
        "distance_per90": 10670,
        "hsr_per90": 592,
        "sprint_per90": 107,
        "acc_per90": 117,
        "dec_per90": 112,
        "pii": 34.35,
    },
    "elevated_activity": {
        "label": "Elevated activity and high intensity",
        "effective_n": 246,
        "distance_per90": 11306,
        "hsr_per90": 840,
        "sprint_per90": 198,
        "acc_per90": 134,
        "dec_per90": 131,
        "pii": 44.44,
    },
}

# ---------------------------------------------------------------------------
# 3.4 -- Dynamic Player DNA (Table 16 + narrative examples)
# ---------------------------------------------------------------------------

DYNAMIC_DNA_COHORT = {
    "n_players": 36,
    "min_appearances": 8,
    "mean_stability": 0.703,
    "mean_hybridity": 0.500,
    "mean_dominant_probability": 0.85,
}

# As with the similarity examples above, the thesis's worked per-player
# illustrations (e.g. its most-stable / most-variable / most-hybrid examples,
# and its single worked case study) are identified by individual
# pseudonymised player codes and individual result values. They are
# intentionally not reproduced here -- only the genuinely aggregate cohort
# statistics in DYNAMIC_DNA_COHORT above are. The qualitative pattern (that
# stability and hybridity vary meaningfully across players, with some
# players much more repeatable than others) is instead demonstrated live on
# the synthetic cohort, where every player is fictional.

DYNAMIC_DNA_ROUND_CHANGE_NOTE = (
    "The thesis illustrates, with one individual worked example, that a player's "
    "autumn-to-spring change in contextual elevated-activity share can be negative "
    "even without that implying a decline in ability -- the contextual profile "
    "reflects standing relative to a player's reference group in a given round, "
    "which can move for reasons unrelated to individual development. The specific "
    "individual figures behind that illustration are not reproduced here."
)

# ---------------------------------------------------------------------------
# 3.4 -- Autumn vs. Spring motor comparison (Table 17)
# ---------------------------------------------------------------------------

SEASON_COMPARISON = {
    "n_players": 31,
    "test": "Paired Wilcoxon signed-rank test, bootstrap CIs, Benjamini-Hochberg FDR correction across 6 variables.",
    "rows": [
        # variable, autumn_mean, spring_mean, abs_change, pct_change, effect_dz, p_fdr, significant
        ("Distance/90", 10638.13, 10965.90, 327.77, 3.1, 0.55, 0.0028, True),
        ("HSR/90", 632.17, 728.15, 95.98, 15.2, 0.79, 0.0003, True),
        ("Sprint/90", 124.66, 153.36, 28.70, 23.0, 0.72, 0.0015, True),
        ("Accelerations/90", 118.89, 124.95, 6.06, 5.1, 0.38, 0.0329, True),
        ("Decelerations/90", 115.60, 119.32, 3.72, 3.2, 0.28, 0.2556, False),
        ("PII", 38.11, 43.94, 5.82, 15.3, 0.72, 0.0006, True),
    ],
}

# ---------------------------------------------------------------------------
# 3.4 -- HMM (Table 18, 19, 20)
# ---------------------------------------------------------------------------

HMM_MODEL_SELECTION = [
    # n_states, log_likelihood, aic, bic, smallest_state_share_pct, mean_ari
    (2, -3731.87, 7493.74, 7563.95, 27.0, 1.000),
    (3, -3684.41, 7420.81, 7542.51, 22.3, 1.000),
]
HMM_CHOSEN_STATES = 3
HMM_INIT_RUNS = 4

STATE_ORDER = ["Lower activity", "Baseline activity", "Elevated activity"]

HMM_STATE_PROFILES = {
    "Lower activity": {
        "n_observations": 267,
        "distance_per90": 10535,
        "hsr_per90": 586,
        "sprint_per90": 115,
        "acc_per90": 107,
        "dec_per90": 101,
        "pii": 31.83,
    },
    "Baseline activity": {
        "n_observations": 352,
        "distance_per90": 10870,
        "hsr_per90": 610,
        "sprint_per90": 104,
        "acc_per90": 127,
        "dec_per90": 120,
        "pii": 37.50,
    },
    "Elevated activity": {
        "n_observations": 178,
        "distance_per90": 11341,
        "hsr_per90": 902,
        "sprint_per90": 225,
        "acc_per90": 135,
        "dec_per90": 136,
        "pii": 45.69,
    },
}

# Row = from-state, column = to-state, order matches STATE_ORDER
HMM_TRANSITION_MATRIX = [
    [0.740, 0.113, 0.147],
    [0.066, 0.887, 0.047],
    [0.189, 0.089, 0.722],
]

HMM_PERSISTENCE_NOTE = (
    "The baseline state showed the highest persistence (0.887 probability "
    "of remaining in the same state at the next appearance), followed by "
    "lower activity (0.740) and elevated activity (0.722). The direct "
    "lower-to-elevated transition probability was 0.147; the direct "
    "baseline-to-elevated transition probability was 0.047."
)

# ---------------------------------------------------------------------------
# 3.4 -- Kaplan-Meier (first Lower -> Elevated transition)
# ---------------------------------------------------------------------------

SURVIVAL_RESULTS = {
    "n_players": 31,
    "n_events": 24,
    "n_censored": 7,
    "median_appearances": 8,
    "median_ci95": (4, 16),
    "event_definition": (
        "First direct transition from the Lower-activity state to the "
        "Elevated-activity state in a player's chronologically ordered "
        "sequence of qualifying appearances."
    ),
    "censoring_note": (
        "Players whose observed sequence ended before this transition "
        "occurred are right-censored -- the event may still occur after "
        "the observation window, not that data is missing."
    ),
}

# ---------------------------------------------------------------------------
# 3.5 -- Limitations (verbatim scope, paraphrased into English)
# ---------------------------------------------------------------------------

LIMITATIONS = [
    "Technical/tactical data completeness varied across matches and rounds; "
    "the main dynamic analysis was therefore built on the more complete GPS "
    "(motor) dataset.",
    "The detailed technical-tactical PCA is exploratory: it is based on "
    "fewer observations and a more complex loading structure, and should "
    "not be generalised beyond the three squads studied without "
    "re-validation on new data.",
    "Players differ substantially in the number of qualifying appearances. "
    "A minimum of eight appearances and bootstrap confidence intervals were "
    "used for the main Dynamic Player DNA table, but some estimates -- "
    "particularly for players with fewer matches -- still carry wide "
    "intervals.",
    "Per-90-minute conversion improves comparability but does not fully "
    "remove instability in short appearances; the 20-minute threshold "
    "reduces, but does not eliminate, the effect of substitute entries, "
    "match phase, and scoreline.",
    "Information on opponent strength, match type, final result, exact "
    "in-match position, and tactical assignment was incomplete. Contextual "
    "standardisation (round, squad category, broad role) does not control "
    "for these factors, so profile changes should not automatically be "
    "read as improvement or decline.",
    "GMM profiles and HMM states are statistical structures that depend on "
    "the chosen variable set, standardisation method, and analysed sample. "
    "The two GMM profiles describe motor-activity level, not a complete "
    "playing style.",
    "The study covers three squads (CII, U17, U19) of one academy over one "
    "season (2025/2026, autumn and spring rounds). Results describe this "
    "group and are not universal norms for other teams or academies.",
    "No completed expert (coaching staff) validation of the profiles "
    "against pitch-side observation was carried out at the time of "
    "writing; this is proposed as further work.",
    "Goalkeepers were excluded from the main analysis, which requires a "
    "distinct set of diagnostic variables.",
]

FURTHER_WORK = [
    "Collect additional seasons and a larger player sample.",
    "Standardise the collection of technical/tactical statistics across "
    "matches and rounds.",
    "Incorporate opponent strength, match result, exact playing position, "
    "and tactical assignment.",
    "Build a dedicated profiling model for goalkeepers.",
    "Carry out a formal validation of profiles and states against coaching "
    "staff observation.",
]

# ---------------------------------------------------------------------------
# Player Intensity Index -- definition (section 2.2)
# ---------------------------------------------------------------------------

PII_DEFINITION = {
    "formula": "PII = (HSR + 1.5 x Sprint + 2 x ACC + 2 x DEC) / (TD / MP)",
    "weights": {"HSR": 1.0, "Sprint": 1.5, "Accelerations": 2.0, "Decelerations": 2.0},
    "denominator": "Total distance divided by minutes played (pace of total distance covered).",
    "note": (
        "The weights are constructional choices made to combine variables "
        "expressed in different units into a single index. They are NOT "
        "empirically calibrated estimates of physiological cost. PII should "
        "be read as a synthetic analytical index describing the share of "
        "intense, dynamic actions in a player's motor activity -- it has no "
        "physical unit of its own."
    ),
}

# ---------------------------------------------------------------------------
# Radar dimension construction (Tables 5 and 6)
# ---------------------------------------------------------------------------

TECHNICAL_RADAR_DIMENSIONS = {
    "Finishing": ["goals_per90", "finishing_actions_per90"],
    "Creativity": ["key_passes_per90", "chance_assists_per90", "assists_per90"],
    "Individual play": ["off_duels_won_per90", "key_individual_actions_per90"],
    "Defence & recovery": ["recoveries_per90", "interventions_per90", "def_duels_won_per90"],
    "Box defence": ["blocks_per90", "box_actions_per90"],
    "Safety": ["key_losses_per90", "errors_per90"],  # percentile reversed: fewer is better
}
TECHNICAL_RADAR_REVERSED = {"Safety"}

MOTOR_RADAR_DIMENSIONS = {
    "Work volume": ["distance_per90"],
    "High intensity": ["hsr_per90", "sprint_per90"],
    "Tempo changes": ["acc_per90", "dec_per90"],
    "Relative intensity": ["pii"],
}

COMBINED_RADAR_DIMENSIONS = {
    "Finishing": ["goals_per90", "finishing_actions_per90"],
    "Creativity": ["key_passes_per90", "chance_assists_per90", "assists_per90"],
    "Individual play": ["off_duels_won_per90", "key_individual_actions_per90"],
    "Defensive influence": ["recoveries_per90", "interventions_per90", "def_duels_won_per90"],
    "Safety": ["key_losses_per90", "errors_per90"],
    "Work volume": ["distance_per90"],
    "High intensity": ["hsr_per90", "sprint_per90"],
    "Movement dynamics": ["acc_per90", "dec_per90", "pii"],
}
COMBINED_RADAR_REVERSED = {"Safety"}

# ---------------------------------------------------------------------------
# Software used in the original study (for the Methodology / About pages)
# ---------------------------------------------------------------------------

THESIS_SOFTWARE = [
    "Python",
    "pandas and NumPy (data preparation)",
    "SciPy (statistical computation)",
    "scikit-learn (standardisation, PCA, K-means, Gaussian Mixture Models)",
    "Matplotlib (static visualisation)",
]

RESEARCH_QUESTIONS = [
    "Does combining technical/tactical and motor data produce a clear and "
    "interpretable player profile?",
    "Does soft membership to motor-activity profiles describe the "
    "complexity of a player's appearance better than a hard cluster "
    "assignment?",
    "How do player profiles change between consecutive appearances and "
    "between the autumn and spring rounds?",
    "Which players show a stable profile, and which show greater "
    "variability or hybridity?",
    "Can transition points between motor-activity states be identified "
    "from a sequence of consecutive matches?",
    "What is the median number of appearances to a first transition from "
    "a lower-activity state to an elevated-activity state?",
]
