"""
Sanity checks that are easy to silently get wrong: GMM soft-membership
probabilities must sum to 1 per observation, and HMM transition matrix
rows must sum to 1 (each is a proper conditional probability
distribution over "next state").
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src import data_pipeline as dp
from src import hmm_model as hmm


def test_gmm_probabilities_sum_to_one():
    bundle = dp.build_full_bundle("data/synthetic_player_data.csv")
    df = bundle["df"]
    totals = df["gmm_lower_prob"] + df["gmm_higher_prob"]
    assert np.allclose(totals, 1.0, atol=1e-6), "GMM membership probabilities do not sum to 1"
    assert (df["gmm_lower_prob"] >= -1e-9).all() and (df["gmm_lower_prob"] <= 1 + 1e-9).all()
    assert (df["gmm_higher_prob"] >= -1e-9).all() and (df["gmm_higher_prob"] <= 1 + 1e-9).all()


def test_hmm_transition_matrix_rows_sum_to_one():
    bundle = dp.build_full_bundle("data/synthetic_player_data.csv")
    transmat = bundle["hmm"]["transmat"]
    row_sums = transmat.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"Transition matrix rows do not sum to 1: {row_sums}"
    assert (transmat >= -1e-9).all(), "Transition matrix contains negative probabilities"


def test_hmm_posterior_probabilities_sum_to_one():
    """Direct check on the underlying HMM implementation, independent of the full pipeline."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    lengths = [20, 20, 20]
    model = hmm.GaussianHMM(n_states=2, n_iter=30, random_state=1)
    model.fit(X, lengths)
    posterior = model.posterior_proba(X, lengths)
    sums = posterior.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6), "HMM posterior state probabilities do not sum to 1"


def test_hmm_fit_best_of_n_params_matches_expected_formula():
    """
    Regression guard matching the parameter-count argument that caught the
    HMM feature-space bug: a diagonal-covariance Gaussian HMM with K states
    and D input dimensions must have exactly K*(K-1) + (K-1) + 2*K*D free
    parameters.
    """
    rng = np.random.default_rng(0)
    for K, D in [(2, 3), (3, 3), (3, 6)]:
        X = rng.normal(size=(90, D))
        lengths = [30, 30, 30]
        model = hmm.GaussianHMM(n_states=K, n_iter=10, random_state=2)
        model.fit(X, lengths)
        expected = K * (K - 1) + (K - 1) + 2 * K * D
        assert model.result.n_params == expected, (K, D, model.result.n_params, expected)


if __name__ == "__main__":
    test_gmm_probabilities_sum_to_one()
    test_hmm_transition_matrix_rows_sum_to_one()
    test_hmm_posterior_probabilities_sum_to_one()
    test_hmm_fit_best_of_n_params_matches_expected_formula()
    print("test_gmm_and_hmm.py: all tests passed")
