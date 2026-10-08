# Dynamic Player DNA

An interactive football analytics research product: dynamic player profiling through
unsupervised learning and sequence modelling, built as a public companion to a bachelor's
thesis on the same subject.

## Overview

Traditional player profiling asks one question: *what type of player is this?* It aggregates
a season into a single static label. Dynamic Player DNA adds a second question: *how does
that profile change from match to match?* Instead of one static label, each appearance is
described by soft membership probabilities across motor-activity profiles. Tracking those
probabilities across consecutive appearances makes it possible to describe a player's
profile as stable or variable, as leaning on one activity profile or mixing two evenly, and
to detect statistically distinct activity states and the transitions between them.

This repository contains a full, working Streamlit application implementing that pipeline
end to end on a synthetic dataset, plus the generator that produced the dataset.

## Research context

- **Thesis**: *Application of Selected Machine Learning Methods in Football Data Analysis*
  (Polish: *Wykorzystanie wybranych metod uczenia maszynowego w analizie danych piłki
  nożnej*).
- **Author**: Hubert Gogola.
- **Institution**: AGH University of Krakow, Faculty of Management, field of study:
  Computer Science and Econometrics, 2026.
- **Supervisor**: Dr Beata Basiura.

The thesis studied internal technical/tactical and GPS motor data from three squads (CII,
U17, U19) of one football academy across the autumn and spring rounds of the 2025/2026
season, and developed the Dynamic Player DNA concept: PCA for dimensionality reduction,
K-means as a hard-clustering baseline, a Gaussian Mixture Model for soft profile membership,
match-to-match stability/hybridity metrics, an experimental Hidden Markov Model for activity
states, and Kaplan-Meier survival analysis for time-to-transition between states.

Research questions addressed by the thesis, and the pipeline stage that answers each one,
are listed in the application's *About & Research* page.

## Methodology / pipeline

1. **Data preparation** -- cleaning, per-90 conversion, contextual standardisation (round x
   squad category x broad role).
2. **PCA** -- dimensionality reduction on the standardized motor variables (plus a combined
   technical+motor variant).
3. **K-means** -- hard-clustering baseline on the retained components.
4. **Gaussian Mixture Model** -- soft profile membership probabilities per appearance.
5. **Dynamic Player DNA** -- stability and hybridity computed from the GMM membership
   trajectory.
6. **Hidden Markov Model** -- a from-scratch Gaussian HMM (Baum-Welch EM, Viterbi decoding;
   see `src/hmm_model.py`) fit on the same PC1-PC3 representation used for K-means and GMM,
   identifying Lower / Baseline / Elevated activity states and transition probabilities from
   chronological appearance sequences. (Fitting on the 3 retained components, rather than
   the full 6-variable motor space, also matches what the thesis's own reported AIC and
   log-likelihood for its 3-state model imply about its parameter count.)
7. **Kaplan-Meier** -- time (in appearances) to a player's first direct transition from the
   Lower-activity to the Elevated-activity state, with right-censoring handled correctly.

Every page in the application clearly labels each figure as either a **Thesis result**
(reproduced exactly from the source document, frozen in `src/thesis_results.py`, never
recomputed) or a **Synthetic demo** (computed live, every time the app runs, on the
fictional dataset). The two are never visually or numerically blended.

### Selected thesis findings (reproduced in the app, see the *Methodology* and page-level
benchmark panels for full detail and citations)

- Main motor PCA: PC1-PC3 explained 83.80% of variance.
- K-means (context-standardized, k=2): silhouette = 0.335; bootstrap mean ARI = 0.755.
- GMM (BIC-selected): 2 profiles -- "Lower motor activity" and "Elevated activity and high
  intensity".
- Dynamic Player DNA (36 players, >=8 appearances): mean stability = 0.703, mean
  hybridity = 0.500.
- HMM (BIC-selected): 3 states; baseline-state persistence = 0.887.
- Kaplan-Meier: median = 8 appearances to first Lower -> Elevated transition (bootstrap 95%
  CI: 4-16; n=31, 24 events, 7 right-censored).

## Synthetic data

**No real academy data, real player identities, or anonymised transformations of real
observations are published in this repository.** Every player, match and observation in
`data/synthetic_player_data.csv` is fictional, produced by the deterministic, seeded
generator in `src/generate_data.py`.

The generator simulates a hidden 3-state activity process per fictional player (so the
application's own HMM has a real, recoverable structure to find rather than labels being
hard-coded), draws raw GPS totals conditional on role / hidden state / round / an individual
baseline, and then *derives* per-90 values and the Player Intensity Index from those raw
totals using the formula described in the thesis -- it does not sample PII directly. A
data-completeness pattern (technical detail more often missing in autumn than spring) is
injected to mirror the real completeness issue documented in the thesis. Regenerate it with:

```bash
python3 -m src.generate_data
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.10+ (the codebase uses `X | None` type hints).

## Tests

```bash
python3 tests/run_tests.py        # works with no extra dependency
# or, if pytest is installed:
python3 -m pytest tests/
```

Covers: the PII formula against an independent reconstruction from raw totals, contextual
z-score standardisation (group mean/std, and that it weakens the role confound relative to
the no-context baseline), GMM membership probabilities summing to 1, HMM transition-matrix
rows summing to 1 (plus a parameter-count regression guard), Kaplan-Meier against a
hand-worked example, and -- most importantly -- that no individual (even pseudonymised)
player-level result from the source thesis appears anywhere in the shipped dataset or
constants.

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app pointing at `app.py` on the default
   branch.
3. No secrets or external services are required -- the app is fully self-contained and reads
   only the bundled synthetic CSV.

## Project structure

```
app.py                     Entry point: page config, global CSS, navigation router (6 top-level pages)
pages/
  overview.py               Landing page
  player_dna.py             Individual fictional player profile (the "DNA fingerprint" centerpiece)
  player_comparison.py      Player A vs. Player B, side by side
  cohort_analysis.py        EDA: distributions, correlations, data completeness
  modelling.py              Tabs: PCA Explorer | Clustering Lab | Dynamic Player DNA | HMM & States | Survival
  reference.py              Tabs: Methodology | About & Research
sections/                   The content behind each tab in modelling.py / reference.py, each a
                             plain `render()` function (kept separate from pages/ so each tab's
                             widget state and local variables stay fully independent)
src/
  thesis_results.py         Frozen thesis constants (never recomputed, aggregate-only -- see Privacy)
  generate_data.py          Deterministic synthetic data generator
  data_pipeline.py          Pure analytical pipeline (PCA, K-means, GMM, HMM, similarity...)
  hmm_model.py               From-scratch Gaussian HMM (Baum-Welch EM, Viterbi)
  stats_utils.py             Bootstrap ARI, Benjamini-Hochberg FDR, Kaplan-Meier, Cramer's V
  cache.py                    Thin Streamlit caching layer around data_pipeline
  theme.py                    Design tokens, CSS injection, Plotly theme
  components.py                Reusable Streamlit UI building blocks (incl. the DNA fingerprint chart)
data/synthetic_player_data.csv  Generated synthetic dataset
.streamlit/config.toml           Theme configuration
tests/                            PII, standardisation, GMM/HMM, survival and privacy tests
```

`data_pipeline.py`, `generate_data.py`, `hmm_model.py` and `stats_utils.py` contain no
Streamlit imports, so the entire analytical core can be exercised and unit-tested with plain
Python.

## Limitations

See the *Methodology* page for the full list of limitations carried over from the thesis
(uneven technical-data completeness, variable sequence lengths, incomplete match context,
single-academy / single-season scope, no completed expert validation at time of writing, and
others), and this application's own additional note that its stability/hybridity formula is
a documented operationalisation of a concept the thesis describes qualitatively but does not
publish as a closed-form equation.

## Author

Hubert Gogola -- Bachelor's degree, Computer Science and Econometrics, AGH University of
Krakow (MSc studies in progress, same field). Professional focus: football analytics, data
analysis, machine learning, player profiling.
