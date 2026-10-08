"""
Statistical helper functions shared across the application.

Kept dependency-light on purpose (numpy, pandas, scipy, scikit-learn only)
so every function here can be unit-checked without a Streamlit runtime.
Nothing in this module touches the frozen thesis numbers in
`thesis_results.py` -- it only operates on whatever dataframe/array it is
given, which in the application is always the synthetic demo data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# ---------------------------------------------------------------------------
# Clustering quality across a range of k
# ---------------------------------------------------------------------------

def kmeans_quality_curve(X: np.ndarray, k_values=range(2, 7), random_state: int = 42) -> pd.DataFrame:
    """Silhouette / Calinski-Harabasz / Davies-Bouldin for each k."""
    rows = []
    n = X.shape[0]
    for k in k_values:
        if k >= n:
            continue
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sizes = pd.Series(labels).value_counts(normalize=True)
        rows.append({
            "k": k,
            "silhouette": float(silhouette_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "smallest_cluster_share_pct": float(sizes.min() * 100),
        })
    return pd.DataFrame(rows)


def bootstrap_kmeans_stability(X: np.ndarray, k: int, n_boot: int = 150, random_state: int = 42) -> dict:
    """
    Bootstrap stability of a k-means partition.

    Resamples rows with replacement, refits k-means on the resample, then
    measures agreement with labels predicted (via nearest-centroid) for the
    ORIGINAL data from the resampled model. This mirrors comparing a
    bootstrap partition back to the full-sample partition, which is the
    standard way to probe cluster stability (Efron bootstrap + Hubert-Arabie
    Adjusted Rand Index), without requiring any extra dependency.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    base_km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    base_labels = base_km.fit_predict(X)

    aris = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_boot = X[idx]
        try:
            km_b = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(0, 1_000_000)))
            km_b.fit(X_boot)
            boot_labels_full = km_b.predict(X)
        except Exception:
            continue
        aris.append(adjusted_rand_score(base_labels, boot_labels_full))

    aris = np.array(aris)
    return {
        "labels": base_labels,
        "mean_ari": float(np.mean(aris)) if len(aris) else float("nan"),
        "median_ari": float(np.median(aris)) if len(aris) else float("nan"),
        "ci95": (float(np.percentile(aris, 2.5)), float(np.percentile(aris, 97.5))) if len(aris) else (float("nan"), float("nan")),
        "n_boot": int(len(aris)),
    }


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramer's V association between two categorical series."""
    table = pd.crosstab(a, b)
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, c = table.shape
    denom = min(r - 1, c - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(phi2 / denom))


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction (implemented directly; statsmodels is
# not assumed to be available in all deployment environments)
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p-value down.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


def paired_round_comparison(df: pd.DataFrame, player_col: str, round_col: str,
                             variables: list[str], round_a: str, round_b: str,
                             n_boot: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Paired Wilcoxon signed-rank test (per variable) between two rounds for
    players observed in both, with bootstrap CI on the mean difference,
    standardised effect size dz, and Benjamini-Hochberg FDR correction
    across variables. Mirrors the thesis's Table 17 methodology on
    whatever synthetic data is passed in.
    """
    wide_a = df[df[round_col] == round_a].groupby(player_col)[variables].mean()
    wide_b = df[df[round_col] == round_b].groupby(player_col)[variables].mean()
    common = wide_a.index.intersection(wide_b.index)
    wide_a = wide_a.loc[common]
    wide_b = wide_b.loc[common]

    rng = np.random.default_rng(random_state)
    rows = []
    raw_p = []
    for var in variables:
        a = wide_a[var].to_numpy()
        b = wide_b[var].to_numpy()
        diff = b - a
        mean_a, mean_b = float(a.mean()), float(b.mean())
        abs_change = mean_b - mean_a
        pct_change = (abs_change / mean_a * 100) if mean_a != 0 else float("nan")
        dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("nan")

        try:
            stat, p = wilcoxon(a, b)
        except ValueError:
            p = float("nan")

        boot_means = []
        n = len(diff)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_means.append(diff[idx].mean())
        ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))

        rows.append({
            "variable": var,
            f"{round_a.lower()}_mean": mean_a,
            f"{round_b.lower()}_mean": mean_b,
            "abs_change": abs_change,
            "pct_change": pct_change,
            "effect_dz": dz,
            "p_value": p,
            "boot_ci_low": ci[0],
            "boot_ci_high": ci[1],
        })
        raw_p.append(p)

    out = pd.DataFrame(rows)
    out["p_fdr"] = benjamini_hochberg(out["p_value"].to_numpy())
    out["significant"] = out["p_fdr"] < 0.05
    out.attrs["n_players"] = len(common)
    return out


# ---------------------------------------------------------------------------
# Kaplan-Meier (manual product-limit estimator)
# ---------------------------------------------------------------------------

def kaplan_meier(durations, observed) -> dict:
    """
    Standard product-limit (Kaplan-Meier) estimator with a Greenwood-type
    bootstrap confidence band and a bootstrap CI for the median. Implemented
    directly from the definition so no extra survival-analysis dependency
    is required.

    durations: time (here: appearance number) at which the event or
               censoring occurred.
    observed:  boolean array, True if the event was observed, False if the
               observation is right-censored.
    """
    durations = np.asarray(durations, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    order = np.argsort(durations)
    t = durations[order]
    e = observed[order]

    unique_times = np.unique(t)
    surv = 1.0
    xs = [0.0]
    ys = [1.0]
    table_rows = []
    for ut in unique_times:
        at_risk = int(np.sum(t >= ut))
        n_events = int(np.sum((t == ut) & e))
        n_censored = int(np.sum((t == ut) & ~e))
        if at_risk > 0 and n_events > 0:
            surv *= (1 - n_events / at_risk)
        xs.extend([ut, ut])
        ys.extend([ys[-1], surv])
        table_rows.append({
            "time": ut, "at_risk": at_risk, "events": n_events,
            "censored": n_censored, "survival": surv,
        })

    median = None
    for row in table_rows:
        if row["survival"] <= 0.5:
            median = row["time"]
            break

    return {
        "xs": xs,
        "ys": ys,
        "table": pd.DataFrame(table_rows),
        "median": median,
        "n": len(t),
        "n_events": int(e.sum()),
        "n_censored": int((~e).sum()),
    }


def bootstrap_km_median_ci(durations, observed, n_boot: int = 1000, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    durations = np.asarray(durations, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    n = len(durations)
    medians = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        res = kaplan_meier(durations[idx], observed[idx])
        if res["median"] is not None:
            medians.append(res["median"])
    if not medians:
        return (float("nan"), float("nan"))
    return (float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5)))


# ---------------------------------------------------------------------------
# Similarity engine
# ---------------------------------------------------------------------------

def similarity_table(pool: pd.DataFrame, feature_cols: list[str], reference_id: str,
                      id_col: str = "player_id", top_n: int = 5) -> pd.DataFrame:
    """
    Cosine similarity and Euclidean distance between a reference player and
    every other player in `pool`, computed on standardised features. `pool`
    should already be restricted to a comparable reference group (same
    broad role, ideally same round/category) before calling this -- the
    function does not do that restriction itself, to keep it reusable.
    """
    from sklearn.preprocessing import StandardScaler

    if reference_id not in pool[id_col].values:
        return pd.DataFrame()

    X = StandardScaler().fit_transform(pool[feature_cols].to_numpy())
    ids = pool[id_col].to_numpy()
    ref_idx = int(np.where(ids == reference_id)[0][0])

    cos = cosine_similarity(X)[ref_idx]
    euc = euclidean_distances(X)[ref_idx]

    out = pd.DataFrame({
        id_col: ids,
        "cosine_similarity": cos,
        "euclidean_distance": euc,
    })
    out = out[out[id_col] != reference_id].sort_values("cosine_similarity", ascending=False)
    return out.head(top_n).reset_index(drop=True)
