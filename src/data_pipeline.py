"""
Core analytical pipeline for the synthetic demo dataset.

Every function here operates only on whatever dataframe/array is passed
in -- at runtime that is always `data/synthetic_player_data.csv` or a
filtered view of it. Nothing in this module reads from, or writes into,
`thesis_results.py`. Keeping the two completely separate is what makes it
possible to guarantee thesis figures and live synthetic-demo computations
are never silently blended.

This module intentionally does not import streamlit, so it can be
exercised directly with plain Python for testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from . import hmm_model as hmm

MOTOR_COLS = ["distance_per90", "hsr_per90", "sprint_per90", "acc_per90", "dec_per90", "pii"]
MOTOR_LABELS = {
    "distance_per90": "Distance / 90",
    "hsr_per90": "HSR / 90",
    "sprint_per90": "Sprint / 90",
    "acc_per90": "Accelerations / 90",
    "dec_per90": "Decelerations / 90",
    "pii": "PII",
}
#: The thesis's main PCA / clustering / GMM / HMM variant standardises
#: motor variables with respect to round, squad category, and broad role
#: together -- not role alone -- so that a player's contextual profile
#: reflects their position relative to the group they were actually
#: compared against in a given round.
CONTEXT_GROUP_COLS = ["round", "squad", "role"]

#: The thesis describes the percentile radar's reference group as squad
#: category + broad role (it does not condition the radar on round), so
#: this is kept as a separate constant rather than reusing CONTEXT_GROUP_COLS.
RADAR_GROUP_COLS = ["squad", "role"]


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(["player_id", "round", "appearance_no"]).reset_index(drop=True)
    return df


def add_context_standardized_columns(df: pd.DataFrame, cols: list[str],
                                      group_cols: list[str] = CONTEXT_GROUP_COLS,
                                      suffix: str = "_z") -> pd.DataFrame:
    """
    Z-score each column within its (squad, role) reference group. This is
    the "contextual standardisation" step described in the thesis: it
    removes the part of the variation that is simply due to broad playing
    role or squad category, before any unsupervised model sees the data.
    """
    out = df.copy()
    for col in cols:
        out[col + suffix] = out.groupby(group_cols)[col].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 1e-9 else 1.0)
        )
    return out


def fit_pca(df: pd.DataFrame, feature_cols: list[str], n_components: int | None = None,
            standardize: bool = False):
    """
    Fit PCA. If `standardize` is True, the raw columns are z-scored across
    the whole sample first (the "no context control" baseline variant);
    if False, the columns are assumed to already be the contextually
    standardized (_z) features (the main variant).
    """
    X = df[feature_cols].to_numpy()
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None
    n_components = n_components or len(feature_cols)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X)
    scores_df = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, n_components + 1)], index=df.index)
    return scaler, pca, scores_df


def fit_kmeans(X: np.ndarray, k: int, random_state: int = 42):
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels


def fit_gmm(X: np.ndarray, k: int, random_state: int = 42):
    gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=random_state)
    labels = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)
    return gmm, labels, probs


def order_gmm_components_by_intensity(gmm: GaussianMixture, X: np.ndarray,
                                       intensity_values: np.ndarray) -> dict:
    """
    Map raw GMM component index -> rank (0 = lowest activity level), based
    on each component's responsibility-weighted mean of `intensity_values`
    (e.g. the PII column, or PC1 scores -- whatever single series best
    reflects overall activity level in the feature space `X` was fit on).

    `X` must have the same number of columns the GMM was actually fit on.
    """
    responsibilities = gmm.predict_proba(X)
    weighted_mean = (responsibilities * intensity_values[:, None]).sum(axis=0) / \
        np.maximum(responsibilities.sum(axis=0), 1e-9)
    order = np.argsort(weighted_mean)
    return {int(c): int(rank) for rank, c in enumerate(order)}


def compute_gmm_dynamic_metrics(df: pd.DataFrame, lower_prob: np.ndarray, higher_prob: np.ndarray,
                                 player_col: str = "player_id",
                                 order_cols: list[str] = ("round", "appearance_no")) -> pd.DataFrame:
    """
    Adds, per observation: hybridity (how evenly the two GMM profile
    probabilities are split) and, per player: stability (match-to-match
    repeatability of the soft profile) and mean dominant-profile
    probability.

    Implementation note: the thesis describes stability conceptually as
    "similarity of profile between consecutive appearances" but does not
    publish a closed-form equation. This application operationalises it as

        stability = 1 - mean_t( || p_t - p_(t-1) ||_2 / sqrt(2) )

    where p_t = (lower_prob, higher_prob) at appearance t. The sqrt(2)
    normalisation is the maximum possible distance between two points on
    the probability simplex {(a, 1-a)}, so stability is bounded in [0, 1]
    without needing to clip. This is this application's own, documented
    implementation for the synthetic demo -- not a reproduction of an
    unpublished thesis formula.
    """
    out = df.copy()
    out["gmm_lower_prob"] = lower_prob
    out["gmm_higher_prob"] = higher_prob
    out["hybridity"] = 1 - np.abs(out["gmm_higher_prob"] - out["gmm_lower_prob"])
    out["dominant_prob_obs"] = np.maximum(out["gmm_lower_prob"], out["gmm_higher_prob"])

    out = out.sort_values([player_col, *order_cols])
    stability_map, hybridity_map, dominant_map, n_app_map = {}, {}, {}, {}
    for player, g in out.groupby(player_col):
        probs = g[["gmm_lower_prob", "gmm_higher_prob"]].to_numpy()
        if len(probs) > 1:
            deltas = np.linalg.norm(np.diff(probs, axis=0), axis=1) / np.sqrt(2)
            stability_map[player] = float(np.clip(1 - deltas.mean(), 0, 1))
        else:
            stability_map[player] = float("nan")
        hybridity_map[player] = float(g["hybridity"].mean())
        dominant_map[player] = float(g["dominant_prob_obs"].mean())
        n_app_map[player] = int(len(g))

    out["stability"] = out[player_col].map(stability_map)
    out["mean_hybridity"] = out[player_col].map(hybridity_map)
    out["mean_dominant_prob"] = out[player_col].map(dominant_map)
    out["n_appearances"] = out[player_col].map(n_app_map)
    return out


def bootstrap_stability_hybridity_ci(df: pd.DataFrame, player_id: str, n_boot: int = 300,
                                      random_state: int = 42, block: int = 4) -> dict:
    """
    Bootstrap confidence intervals for one player's stability and
    hybridity, resampling consecutive blocks of appearances (a moving
    block bootstrap) so that the autocorrelation structure of a
    chronological sequence is partly preserved, rather than treating
    appearances as fully exchangeable.
    """
    rng = np.random.default_rng(random_state)
    g = df[df["player_id"] == player_id].sort_values(["round", "appearance_no"])
    probs = g[["gmm_lower_prob", "gmm_higher_prob"]].to_numpy()
    n = len(probs)
    if n < 3:
        return {"stability_ci": (float("nan"), float("nan")), "hybridity_ci": (float("nan"), float("nan"))}

    stabilities, hybridities = [], []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=max(n // block, 1))
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
        sample = probs[idx]
        if len(sample) > 1:
            deltas = np.linalg.norm(np.diff(sample, axis=0), axis=1) / np.sqrt(2)
            stabilities.append(float(np.clip(1 - deltas.mean(), 0, 1)))
        hybridities.append(float(1 - np.abs(sample[:, 1] - sample[:, 0]).mean()))

    def ci(values):
        if not values:
            return (float("nan"), float("nan"))
        return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))

    return {"stability_ci": ci(stabilities), "hybridity_ci": ci(hybridities)}


def fit_player_hmm(df: pd.DataFrame, feature_cols: list[str], n_states: int = 3,
                    player_col: str = "player_id", n_init: int = 4, random_state: int = 7,
                    intensity_col: str | None = None, intensity_projection: np.ndarray | None = None):
    """
    Fit the custom Gaussian HMM on every player's chronological sequence
    of (contextually standardized) motor observations concatenated
    together, then relabel states by intensity so state 0 is always the
    lowest-activity state regardless of the arbitrary order EM converges
    to.

    `intensity_col` must be one of `feature_cols` and is used to rank the
    fitted states (e.g. "pii_z"). If not given, the function looks for a
    column ending in "pii" (covering both "pii" and "pii_z"), and only
    falls back to the first feature column if no such column exists.
    """
    df_sorted = df.sort_values([player_col, "round", "appearance_no"])
    lengths_series = df_sorted.groupby(player_col, sort=False).size()
    ordered_players = df_sorted[player_col].drop_duplicates().tolist()
    lengths = [int(lengths_series[p]) for p in ordered_players]

    X = df_sorted[feature_cols].to_numpy()
    fit_out = hmm.fit_best_of(X, lengths, n_states=n_states, n_init=n_init, random_state=random_state)
    model = fit_out["model"]

    raw_states = model.predict(X, lengths)

    if intensity_projection is not None:
        rank_map = hmm.order_states_by_projection(model, intensity_projection)
    else:
        if intensity_col is None:
            pii_matches = [c for c in feature_cols if c.endswith("pii") or c.endswith("pii_z")]
            intensity_col = pii_matches[0] if pii_matches else feature_cols[0]
        pii_idx = feature_cols.index(intensity_col)
        rank_map = hmm.order_states_by_intensity(model, intensity_col_idx=pii_idx)

    ranked_states = np.array([rank_map[s] for s in raw_states])

    posterior = model.posterior_proba(X, lengths)
    # Re-order posterior columns to match the intensity ranking too.
    col_order = [k for k, _ in sorted(rank_map.items(), key=lambda kv: kv[1])]
    posterior = posterior[:, col_order]

    df_sorted = df_sorted.copy()
    df_sorted["hmm_state_idx"] = ranked_states
    df_sorted["hmm_state_posterior_max"] = posterior.max(axis=1)

    # Re-order the transition matrix and means to the same ranking.
    ranked_transmat = model.result.transmat[np.ix_(col_order, col_order)]
    ranked_means = model.result.means[col_order]

    return {
        "df": df_sorted,
        "model": model,
        "mean_ari_across_inits": fit_out["mean_ari_across_inits"],
        "transmat": ranked_transmat,
        "means": pd.DataFrame(ranked_means, columns=feature_cols),
        "aic": model.result.aic,
        "bic": model.result.bic,
        "log_likelihood": model.result.log_likelihood,
        "n_params": model.result.n_params,
        "posterior": posterior,
    }


def extract_first_transition_events(df: pd.DataFrame, state_col: str = "hmm_state_idx",
                                     lower_state: int = 0, target_state: int = 2,
                                     player_col: str = "player_id",
                                     order_cols: list[str] = ("round", "appearance_no")) -> pd.DataFrame:
    """
    For each player, find the appearance number of the first DIRECT
    transition from `lower_state` to `target_state` between two
    consecutive qualifying appearances. If no such direct transition is
    observed, the player is right-censored at their last appearance.
    """
    rows = []
    for player, g in df.sort_values([player_col, *order_cols]).groupby(player_col):
        states = g[state_col].to_numpy()
        hit = None
        for i in range(len(states) - 1):
            if states[i] == lower_state and states[i + 1] == target_state:
                hit = i + 2  # appearance number of the transition (1-indexed)
                break
        if hit is not None:
            rows.append({"player_id": player, "time": hit, "observed": True})
        else:
            rows.append({"player_id": player, "time": len(states), "observed": False})
    return pd.DataFrame(rows)


def _similarity_pool(df: pd.DataFrame, reference_player: str, feature_cols: list[str],
                      same_round: bool = False, same_squad: bool = False,
                      player_col: str = "player_id") -> dict | None:
    """
    Builds the player-level comparison pool and its standardized feature
    matrix for similarity comparisons -- restricted to the same broad role
    (always), and optionally also the same round and/or squad category,
    following the thesis's own approach of comparing players only within
    comparable groups so that similarity is not driven mainly by
    positional or age-category differences.

    This is the single shared implementation behind both
    `role_restricted_similarity` (the ranking) and
    `similarity_feature_breakdown` (the per-feature "why similar"
    explanation), so both always operate on the exact same standardized
    space -- rather than one computing similarity on standardized features
    while the other explains it using raw-unit differences.

    Returns None if the reference player is not in the resulting pool, or
    the pool has fewer than 2 players (nothing to compare against).
    """
    ref_rows = df[df[player_col] == reference_player]
    if ref_rows.empty:
        return None
    ref_role = ref_rows["role"].iloc[0]

    pool = df[df["role"] == ref_role]
    if same_round:
        ref_round = ref_rows["round"].iloc[0]
        pool = pool[pool["round"] == ref_round]
    if same_squad:
        ref_squad = ref_rows["squad"].iloc[0]
        pool = pool[pool["squad"] == ref_squad]

    agg = pool.groupby(player_col)[feature_cols].mean()
    meta = pool.groupby(player_col)[["squad", "role"]].first()
    if reference_player not in agg.index or len(agg) < 2:
        return None

    X = StandardScaler().fit_transform(agg.to_numpy())
    ids = agg.index.to_numpy()
    ref_idx = int(np.where(ids == reference_player)[0][0])
    return {"agg": agg, "meta": meta, "X": X, "ids": ids, "ref_idx": ref_idx}


def role_restricted_similarity(df: pd.DataFrame, reference_player: str, feature_cols: list[str],
                                same_round: bool = False, same_squad: bool = False,
                                player_col: str = "player_id") -> pd.DataFrame:
    """Ranks every other player in the comparison pool by similarity to `reference_player`."""
    pool_info = _similarity_pool(df, reference_player, feature_cols, same_round, same_squad, player_col)
    if pool_info is None:
        return pd.DataFrame()

    X, ids, meta, ref_idx = pool_info["X"], pool_info["ids"], pool_info["meta"], pool_info["ref_idx"]
    cos = cosine_similarity(X)[ref_idx]
    euc = euclidean_distances(X)[ref_idx]

    out = pd.DataFrame({
        player_col: ids, "squad": meta["squad"].to_numpy(), "role": meta["role"].to_numpy(),
        "cosine_similarity": cos, "euclidean_distance": euc,
    })
    out = out[out[player_col] != reference_player].sort_values("cosine_similarity", ascending=False)
    return out.reset_index(drop=True)


def similarity_feature_breakdown(df: pd.DataFrame, reference_player: str, comparison_player: str,
                                  feature_cols: list[str], same_round: bool = False, same_squad: bool = False,
                                  player_col: str = "player_id") -> pd.DataFrame:
    """
    Per-feature absolute standardized difference between two players,
    computed in the exact same standardized comparison-pool space used by
    `role_restricted_similarity` -- so "why these players are similar" is
    answered in the same space the similarity score itself was computed
    in, rather than on raw per-90 units (whose absolute scales differ
    wildly across features, e.g. distance vs. PII).
    """
    pool_info = _similarity_pool(df, reference_player, feature_cols, same_round, same_squad, player_col)
    if pool_info is None or comparison_player not in pool_info["ids"]:
        return pd.DataFrame()

    X, ids, ref_idx = pool_info["X"], pool_info["ids"], pool_info["ref_idx"]
    comp_idx = int(np.where(ids == comparison_player)[0][0])
    diff = np.abs(X[ref_idx] - X[comp_idx])
    return pd.DataFrame({"feature": feature_cols, "abs_standardized_diff": diff}).sort_values("abs_standardized_diff")


def percentile_profile(df: pd.DataFrame, player_id: str, dimension_map: dict, reversed_dims: set,
                        group_cols: list[str] = RADAR_GROUP_COLS,
                        player_col: str = "player_id") -> pd.DataFrame:
    """
    Builds a percentile-based radar profile for one player, following the
    thesis's construction: each source variable is converted to a
    percentile within the player's reference group (here: squad x role,
    averaged across the player's own observations), negative-direction
    variables are reversed, and each radar dimension is the mean of its
    component percentiles.
    """
    ref_rows = df[df[player_col] == player_id]
    if ref_rows.empty:
        return pd.DataFrame()
    key_values = {c: ref_rows[c].iloc[0] for c in group_cols}
    mask = pd.Series(True, index=df.index)
    for c, v in key_values.items():
        mask &= df[c] == v
    group = df[mask]

    player_means = ref_rows[list({c for cols in dimension_map.values() for c in cols})].mean()
    group_means = group.groupby(player_col)[list({c for cols in dimension_map.values() for c in cols})].mean()

    rows = []
    for dim, cols in dimension_map.items():
        dim_percentiles = []
        for col in cols:
            if col not in group_means.columns or group_means[col].dropna().empty:
                continue
            if pd.isna(player_means.get(col, np.nan)):
                continue  # the player has no valid observations for this column
            pct = float((group_means[col].dropna() <= player_means[col]).mean() * 100)
            if dim in reversed_dims:
                pct = 100 - pct
            dim_percentiles.append(pct)
        if dim_percentiles:
            rows.append({"dimension": dim, "percentile": float(np.mean(dim_percentiles))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full pipeline bundle -- the one-time, expensive computation that every
# page reads from. Pure function (no streamlit) so it can be unit-tested;
# the thin `src/cache.py` wrapper adds Streamlit caching around this.
# ---------------------------------------------------------------------------

def build_full_bundle(data_path: str, hmm_states: int = 3, hmm_n_init: int = 4,
                       random_state: int = 42) -> dict:
    df = load_raw_data(data_path)
    df = add_context_standardized_columns(df, MOTOR_COLS)
    zcols = [c + "_z" for c in MOTOR_COLS]

    # Main PCA variant: contextually standardized features, no further scaling.
    _, pca_main, scores_main = fit_pca(df, zcols, n_components=6, standardize=False)
    df = df.join(scores_main)
    # PC1 here loads positively and broadly on HSR, sprint, accelerations,
    # decelerations and PII (and ~0 on raw distance) -- i.e. it behaves as
    # a general "high-intensity activity" composite, which is a more
    # faithful single signal for ranking Lower/Baseline/Elevated HMM
    # states than any one raw variable alone.

    # Baseline PCA variant: raw per-90 values, globally standardized (no
    # context control) -- used only to illustrate the confounding effect
    # that contextual standardization corrects for.
    _, pca_baseline, scores_baseline = fit_pca(df, MOTOR_COLS, n_components=6, standardize=True)
    scores_baseline = scores_baseline.add_suffix("_baseline")
    df = df.join(scores_baseline)

    # Combined PCA variant (motor + two aggregated technical indicators),
    # mirroring the thesis's second PCA variant.
    combined_cols = ["offensive_index", "defensive_index"] + zcols
    _, pca_combined, scores_combined = fit_pca(df, combined_cols, n_components=len(combined_cols), standardize=False)

    # Main clustering (K-means, k=2) and GMM (k=2) on PC1-3 of the main variant.
    X3 = df[["PC1", "PC2", "PC3"]].to_numpy()
    kmeans_main, kmeans_labels = fit_kmeans(X3, k=2, random_state=random_state)
    df["kmeans_cluster"] = kmeans_labels

    gmm_main, gmm_labels, gmm_probs = fit_gmm(X3, k=2, random_state=random_state)
    intensity = df["pii_z"].to_numpy()
    order_map = order_gmm_components_by_intensity(gmm_main, X3, intensity)
    lower_col = [c for c, r in order_map.items() if r == 0][0]
    higher_col = [c for c, r in order_map.items() if r == 1][0]
    df = compute_gmm_dynamic_metrics(df, gmm_probs[:, lower_col], gmm_probs[:, higher_col])

    # HMM. Fit on PC1-3 of the main variant -- the same reduced space used
    # for K-means and GMM above -- rather than the full 6-variable motor
    # space. This keeps the pipeline internally consistent (data prep ->
    # PCA -> clustering / GMM / HMM, all on the same 3-component
    # representation) and matches what the thesis's own reported numbers
    # imply: for its chosen 3-state model, AIC=7420.81 and
    # log-likelihood=-3684.41 solve to essentially exactly 26 free
    # parameters, which is exactly what a diagonal-covariance 3-state
    # Gaussian HMM has with 3 input dimensions (2*3*3 + 3*2 + 2 = 26) --
    # not 6 (which would give 44). States are ranked by their mean PC1
    # value, since PC1 behaves as a broad "general activity intensity"
    # composite here (see note above).
    hmm_out = fit_player_hmm(df, ["PC1", "PC2", "PC3"], n_states=hmm_states, n_init=hmm_n_init,
                              random_state=random_state, intensity_col="PC1")
    hmm_df = hmm_out["df"][["player_id", "round", "appearance_no", "hmm_state_idx", "hmm_state_posterior_max"]]
    df = df.merge(hmm_df, on=["player_id", "round", "appearance_no"], how="left")

    survival_events = extract_first_transition_events(df, state_col="hmm_state_idx")

    return {
        "df": df,
        "zcols": zcols,
        "pca_main": pca_main,
        "pca_baseline": pca_baseline,
        "pca_combined": pca_combined,
        "combined_cols": combined_cols,
        "kmeans_main": kmeans_main,
        "gmm_main": gmm_main,
        "hmm": hmm_out,
        "survival_events": survival_events,
    }
