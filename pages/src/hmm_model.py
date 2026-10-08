"""
A compact, dependency-light Gaussian Hidden Markov Model.

This is implemented directly on top of numpy/scipy (Baum-Welch EM in log
space, Viterbi decoding) rather than importing a third-party HMM package.
The model supports multiple independent sequences of different lengths
(one per player), diagonal covariance emissions, multiple random
restarts, and standard AIC/BIC model-selection statistics -- everything
the application's HMM page needs, with no dependency whose behaviour
cannot be verified in this environment.

The algorithm is the standard textbook Baum-Welch procedure for a Gaussian
HMM (Rabiner, 1989), implemented in log-space with `scipy.special.logsumexp`
for numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

_LOG_ZERO = -1e10


def _log_gaussian_pdf(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Log density of each row of X under each of K diagonal Gaussians.

    X: (T, D), means: (K, D), variances: (K, D) -> returns (T, K)
    """
    T, D = X.shape
    K = means.shape[0]
    out = np.empty((T, K))
    for k in range(K):
        var_k = np.maximum(variances[k], 1e-6)
        diff = X - means[k]
        out[:, k] = -0.5 * (
            D * np.log(2 * np.pi) + np.sum(np.log(var_k)) + np.sum(diff * diff / var_k, axis=1)
        )
    return out


@dataclass
class HMMFitResult:
    n_states: int
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    log_likelihood: float
    n_iter_run: int
    converged: bool
    n_params: int
    n_obs: int
    aic: float
    bic: float


class GaussianHMM:
    """Gaussian HMM with diagonal covariance, fit via Baum-Welch EM."""

    def __init__(self, n_states: int, n_iter: int = 150, tol: float = 1e-4,
                 random_state: int = 0, var_floor: float = 1e-3):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.var_floor = var_floor
        self.result: HMMFitResult | None = None

    # -- internal: single EM run -------------------------------------------------
    def _init_params(self, X: np.ndarray, rng: np.random.Generator):
        K = self.n_states
        D = X.shape[1]
        km = KMeans(n_clusters=K, n_init=4, random_state=int(rng.integers(0, 1_000_000)))
        labels = km.fit_predict(X)
        means = km.cluster_centers_.copy()
        variances = np.empty((K, D))
        global_var = np.var(X, axis=0) + 1e-6
        for k in range(K):
            mask = labels == k
            if mask.sum() > 1:
                variances[k] = np.var(X[mask], axis=0) + 1e-6
            else:
                variances[k] = global_var
        transmat = rng.dirichlet(np.ones(K) * 5 + np.eye(K)[0] * 0, size=K)
        # Bias toward persistence so the chain does not start degenerate.
        transmat = 0.6 * np.eye(K) + 0.4 * transmat
        transmat = transmat / transmat.sum(axis=1, keepdims=True)
        startprob = rng.dirichlet(np.ones(K))
        return startprob, transmat, means, variances

    def _forward_backward(self, log_b: np.ndarray, log_pi: np.ndarray, log_A: np.ndarray):
        T, K = log_b.shape
        log_alpha = np.empty((T, K))
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_b[t]

        log_beta = np.empty((T, K))
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(log_A + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

        log_seq_prob = logsumexp(log_alpha[-1])
        log_gamma = log_alpha + log_beta - log_seq_prob

        log_xi = np.full((T - 1, K, K), _LOG_ZERO)
        for t in range(T - 1):
            log_xi[t] = (
                log_alpha[t][:, None] + log_A + log_b[t + 1][None, :] + log_beta[t + 1][None, :] - log_seq_prob
            )
        return log_gamma, log_xi, log_seq_prob

    def _single_fit(self, X: np.ndarray, lengths: list[int], rng: np.random.Generator) -> HMMFitResult:
        K = self.n_states
        D = X.shape[1]
        startprob, transmat, means, variances = self._init_params(X, rng)

        splits = np.cumsum(lengths)[:-1]
        sequences = np.split(X, splits)

        prev_ll = -np.inf
        converged = False
        n_iter_run = 0

        for it in range(self.n_iter):
            n_iter_run = it + 1
            log_A = np.log(np.clip(transmat, 1e-12, 1))
            log_pi = np.log(np.clip(startprob, 1e-12, 1))

            gamma_list, xi_list, ll_total = [], [], 0.0
            for seq in sequences:
                log_b = _log_gaussian_pdf(seq, means, variances)
                log_gamma, log_xi, log_seq_prob = self._forward_backward(log_b, log_pi, log_A)
                gamma_list.append(np.exp(log_gamma))
                xi_list.append(np.exp(np.clip(log_xi, -700, 0)))
                ll_total += log_seq_prob

            # M-step
            new_startprob = np.mean([g[0] for g in gamma_list], axis=0)
            xi_sum = sum(xi.sum(axis=0) for xi in xi_list)
            gamma_sum_excl_last = sum(g[:-1].sum(axis=0) for g in gamma_list)
            new_transmat = xi_sum / np.maximum(gamma_sum_excl_last[:, None], 1e-12)
            new_transmat = new_transmat / new_transmat.sum(axis=1, keepdims=True)

            gamma_all = np.concatenate(gamma_list, axis=0)
            weight_sum = gamma_all.sum(axis=0)
            new_means = (gamma_all.T @ X) / np.maximum(weight_sum[:, None], 1e-12)
            new_variances = np.empty((K, D))
            for k in range(K):
                diff = X - new_means[k]
                new_variances[k] = (gamma_all[:, k][:, None] * diff * diff).sum(axis=0) / max(weight_sum[k], 1e-12)
            new_variances = np.maximum(new_variances, self.var_floor)

            startprob, transmat, means, variances = new_startprob, new_transmat, new_means, new_variances

            if abs(ll_total - prev_ll) < self.tol * (1 + abs(prev_ll)):
                converged = True
                prev_ll = ll_total
                break
            prev_ll = ll_total

        n_params = K * (K - 1) + (K - 1) + K * D + K * D
        n_obs = X.shape[0]
        aic = -2 * prev_ll + 2 * n_params
        bic = -2 * prev_ll + n_params * np.log(n_obs)

        return HMMFitResult(
            n_states=K, startprob=startprob, transmat=transmat, means=means,
            variances=variances, log_likelihood=float(prev_ll), n_iter_run=n_iter_run,
            converged=converged, n_params=n_params, n_obs=n_obs, aic=float(aic), bic=float(bic),
        )

    def fit(self, X: np.ndarray, lengths: list[int]):
        rng = np.random.default_rng(self.random_state)
        self.result = self._single_fit(np.asarray(X, dtype=float), list(lengths), rng)
        return self

    def predict(self, X: np.ndarray, lengths: list[int]) -> np.ndarray:
        """Viterbi (most likely state sequence) decoding."""
        assert self.result is not None, "call fit() first"
        r = self.result
        log_A = np.log(np.clip(r.transmat, 1e-12, 1))
        log_pi = np.log(np.clip(r.startprob, 1e-12, 1))

        splits = np.cumsum(lengths)[:-1]
        sequences = np.split(np.asarray(X, dtype=float), splits)
        all_states = []
        for seq in sequences:
            T, K = len(seq), r.n_states
            log_b = _log_gaussian_pdf(seq, r.means, r.variances)
            delta = np.empty((T, K))
            psi = np.zeros((T, K), dtype=int)
            delta[0] = log_pi + log_b[0]
            for t in range(1, T):
                scores = delta[t - 1][:, None] + log_A
                psi[t] = np.argmax(scores, axis=0)
                delta[t] = scores[psi[t], np.arange(K)] + log_b[t]
            states = np.empty(T, dtype=int)
            states[-1] = int(np.argmax(delta[-1]))
            for t in range(T - 2, -1, -1):
                states[t] = psi[t + 1, states[t + 1]]
            all_states.append(states)
        return np.concatenate(all_states)

    def posterior_proba(self, X: np.ndarray, lengths: list[int]) -> np.ndarray:
        """Per-timestep posterior state probabilities (gamma), concatenated."""
        assert self.result is not None, "call fit() first"
        r = self.result
        log_A = np.log(np.clip(r.transmat, 1e-12, 1))
        log_pi = np.log(np.clip(r.startprob, 1e-12, 1))
        splits = np.cumsum(lengths)[:-1]
        sequences = np.split(np.asarray(X, dtype=float), splits)
        out = []
        for seq in sequences:
            log_b = _log_gaussian_pdf(seq, r.means, r.variances)
            log_gamma, _, _ = self._forward_backward(log_b, log_pi, log_A)
            out.append(np.exp(log_gamma))
        return np.concatenate(out, axis=0)


def fit_best_of(X: np.ndarray, lengths: list[int], n_states: int, n_init: int = 4,
                 n_iter: int = 150, random_state: int = 42):
    """
    Fit n_init random restarts and keep the highest-log-likelihood run,
    mirroring the thesis's practice of fitting each candidate model several
    times from different initial values. Also reports the mean pairwise
    Adjusted Rand Index between the state labels of the different restarts
    as a stability diagnostic, in the same spirit as the thesis's
    cross-initialisation ARI check.
    """
    X = np.asarray(X, dtype=float)
    runs = []
    label_sets = []
    for i in range(n_init):
        model = GaussianHMM(n_states=n_states, n_iter=n_iter, random_state=random_state + i * 97)
        model.fit(X, lengths)
        runs.append(model)
        label_sets.append(model.predict(X, lengths))

    best_idx = int(np.argmax([m.result.log_likelihood for m in runs]))
    best_model = runs[best_idx]

    aris = []
    for i in range(n_init):
        for j in range(i + 1, n_init):
            aris.append(adjusted_rand_score(label_sets[i], label_sets[j]))
    mean_ari = float(np.mean(aris)) if aris else float("nan")

    return {
        "model": best_model,
        "mean_ari_across_inits": mean_ari,
        "n_init": n_init,
        "all_log_likelihoods": [m.result.log_likelihood for m in runs],
    }


def order_states_by_intensity(model: GaussianHMM, intensity_col_idx: int) -> dict:
    """
    Returns a mapping {raw_state_index -> rank (0=lowest intensity, ... )}
    based on the mean of one chosen emission dimension (e.g. PII or HSR),
    so fitted states can be consistently labelled Lower / Baseline /
    Elevated regardless of the arbitrary state order EM converges to.
    """
    means = model.result.means[:, intensity_col_idx]
    order = np.argsort(means)  # ascending
    rank_of_state = {int(state): int(rank) for rank, state in enumerate(order)}
    return rank_of_state


def order_states_by_projection(model: GaussianHMM, weight_vector: np.ndarray) -> dict:
    """
    Same as `order_states_by_intensity`, but ranks states by the mean
    emission vector projected onto an arbitrary direction (e.g. a PCA
    loading vector) rather than a single raw column. Useful when the best
    single "activity level" signal is a composite of several emission
    dimensions rather than any one of them alone.
    """
    projected = model.result.means @ np.asarray(weight_vector, dtype=float)
    order = np.argsort(projected)  # ascending
    return {int(state): int(rank) for rank, state in enumerate(order)}
