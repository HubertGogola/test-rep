"""
Synthetic data generator for the Dynamic Player DNA public demo.

Every player, match, and observation produced by this module is entirely
fictional. Nothing here is derived from, copied from, or anonymised from
the original academy dataset used in the source thesis. The generator
exists to demonstrate the analytical pipeline (PCA, K-means, GMM, Dynamic
Player DNA, HMM, Kaplan-Meier) on data with a known internal structure, so
that the methods can be shown recovering a pattern rather than operating on
numbers chosen to look good.

Design summary
---------------
* Each fictional player is assigned a role (DEF / MID / ATT), a squad
  category (CII / U17 / U19) and an individual baseline offset.
* A hidden 3-state Markov chain ("Lower" / "Baseline" / "Elevated"
  activity) is simulated per player across their chronological sequence
  of appearances. This chain is NOT exposed in the shipped dataset -- it
  exists only so that the HMM fitted later in the application has a real,
  recoverable structure to find, rather than labels being hard-coded.
* Raw GPS totals (distance, HSR, sprint distance, accelerations,
  decelerations) are drawn conditional on role, hidden state, round
  (autumn/spring), the player's individual baseline, and minutes played;
  per-90 values and the Player Intensity Index are then DERIVED from those
  raw totals using the same formulas described in the thesis, rather than
  sampled directly, which keeps the dataset internally consistent.
* Technical/tactical action counts are generated from per-player latent
  skill parameters conditioned on role, independent of the hidden motor
  state (with only a small deliberate coupling), mirroring the thesis's
  own observation that technical and motor data carry partially
  independent information.
* A data-completeness pattern is injected (technical detail is more often
  missing in autumn than in spring) to mirror the real completeness issue
  documented in the thesis, without copying any of its actual values.

Run this file directly to (re)generate data/synthetic_player_data.csv:

    python3 -m src.generate_data

The random seed is fixed, so the output is byte-for-byte reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 713_205

SQUADS = ["CII", "U17", "U19"]
ROLES = ["DEF", "MID", "ATT"]
ROUNDS = ["Autumn", "Spring"]

STATE_NAMES = ["Lower activity", "Baseline activity", "Elevated activity"]

# Ground-truth hidden-state transition matrix used only to SIMULATE data.
# Deliberately different in value from anything reported in the thesis --
# the application later fits its own HMM to recover a transition structure
# from the observable motor variables, and that recovered (synthetic-demo)
# matrix is compared, clearly labelled, against the thesis's own result.
TRUE_TRANSITION = np.array([
    [0.72, 0.20, 0.08],
    [0.10, 0.80, 0.10],
    [0.12, 0.18, 0.70],
])
TRUE_INITIAL = np.array([0.32, 0.43, 0.25])

# Reference per-90 motor rates (league-plausible magnitudes; not sourced
# from any dataset) and per-state / per-role multipliers.
BASE_RATE = {"distance": 10800.0, "hsr": 650.0, "sprint": 140.0, "acc": 120.0, "dec": 115.0}

STATE_MULT = {
    0: {"distance": 0.98, "hsr": 0.87, "sprint": 0.85, "acc": 0.92, "dec": 0.91},
    1: {"distance": 1.00, "hsr": 1.00, "sprint": 1.00, "acc": 1.00, "dec": 1.00},
    2: {"distance": 1.03, "hsr": 1.20, "sprint": 1.28, "acc": 1.11, "dec": 1.12},
}

ROLE_MULT = {
    "DEF": {"distance": 1.00, "hsr": 0.85, "sprint": 0.80, "acc": 0.95, "dec": 1.05},
    "MID": {"distance": 1.05, "hsr": 1.05, "sprint": 0.95, "acc": 1.05, "dec": 1.00},
    "ATT": {"distance": 0.95, "hsr": 1.15, "sprint": 1.25, "acc": 1.05, "dec": 0.95},
}

# Small, deliberately modest seasonal drift (autumn -> spring). Magnitudes
# are independent design choices, not copies of the thesis's Table 17.
SPRING_BUMP = {"distance": 1.02, "hsr": 1.08, "sprint": 1.12, "acc": 1.04, "dec": 1.01}

ROLE_SKILL_PRIOR = {
    # role: (finishing, creativity, individual, defence, box_defence, discipline)
    "ATT": (0.75, 0.55, 0.65, 0.25, 0.20, 0.55),
    "MID": (0.40, 0.75, 0.55, 0.45, 0.30, 0.58),
    "DEF": (0.15, 0.35, 0.35, 0.75, 0.70, 0.62),
}

SQUAD_AGE_RANGE = {"U17": (15.2, 17.4), "U19": (17.1, 19.4), "CII": (18.3, 21.6)}
SQUAD_SIZE = {"CII": 12, "U17": 14, "U19": 14}


def _clip01(x):
    return np.clip(x, 0.03, 0.97)


def _simulate_state_path(rng: np.random.Generator, n: int) -> np.ndarray:
    states = np.empty(n, dtype=int)
    states[0] = rng.choice(3, p=TRUE_INITIAL)
    for t in range(1, n):
        states[t] = rng.choice(3, p=TRUE_TRANSITION[states[t - 1]])
    return states


def _build_player_roster(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pid = 1
    for squad in SQUADS:
        n_players = SQUAD_SIZE[squad]
        age_lo, age_hi = SQUAD_AGE_RANGE[squad]
        role_choices = rng.choice(ROLES, size=n_players, p=[0.38, 0.38, 0.24])
        for i in range(n_players):
            role = role_choices[i]
            age = round(float(rng.uniform(age_lo, age_hi)), 1)
            rows.append({
                "player_id": f"FPL{pid:02d}",
                "squad": squad,
                "role": role,
                "age": age,
            })
            pid += 1
    return pd.DataFrame(rows)


def generate_dataset(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    roster = _build_player_roster(rng)

    all_rows = []

    for _, player in roster.iterrows():
        role = player["role"]
        pid = player["player_id"]

        # Individual baseline multiplicative offsets (per-player heterogeneity).
        motor_keys = ["distance", "hsr", "sprint", "acc", "dec"]
        player_baseline = {k: float(np.exp(rng.normal(0, 0.08))) for k in motor_keys}

        fin0, cre0, ind0, dfn0, box0, dis0 = ROLE_SKILL_PRIOR[role]
        skill = {
            "finishing": _clip01(rng.normal(fin0, 0.15)),
            "creativity": _clip01(rng.normal(cre0, 0.15)),
            "individual": _clip01(rng.normal(ind0, 0.15)),
            "defence": _clip01(rng.normal(dfn0, 0.15)),
            "box_defence": _clip01(rng.normal(box0, 0.15)),
            "discipline": _clip01(rng.normal(dis0, 0.15)),
        }

        has_autumn = True
        has_spring = rng.random() < 0.87
        n_autumn = int(rng.integers(5, 19)) if has_autumn else 0
        n_spring = int(rng.integers(5, 19)) if has_spring else 0
        n_total = n_autumn + n_spring
        if n_total == 0:
            n_autumn = 6
            n_total = 6

        round_labels = ["Autumn"] * n_autumn + ["Spring"] * n_spring
        states = _simulate_state_path(rng, n_total)

        for t in range(n_total):
            rnd = round_labels[t]
            state = states[t]

            # Minutes played: mixture of full/near-full appearances and
            # shorter substitute entries, including some sub-20-minute
            # cameos that will later be filtered out exactly as the
            # thesis describes doing with its own threshold.
            if rng.random() < 0.68:
                minutes = float(rng.uniform(64, 96))
            else:
                minutes = float(rng.uniform(8, 64))

            bump = SPRING_BUMP if rnd == "Spring" else {k: 1.0 for k in motor_keys}

            raw = {}
            for k in motor_keys:
                target_per90 = (
                    BASE_RATE[k]
                    * ROLE_MULT[role][k]
                    * STATE_MULT[state][k]
                    * player_baseline[k]
                    * bump[k]
                )
                noise = float(np.exp(rng.normal(0, 0.12)))
                raw_total = target_per90 * (minutes / 90.0) * noise
                raw[k] = max(raw_total, 0.0)

            td, hsr, sprint, acc, dec = (raw["distance"], raw["hsr"], raw["sprint"], raw["acc"], raw["dec"])

            # Per-90 conversion, exactly as defined in the thesis: X90 = X/M * 90
            td90 = float(np.clip(td / minutes * 90, 7200, 13600))
            hsr90 = float(np.clip(hsr / minutes * 90, 280, 1150))
            sprint90 = float(np.clip(sprint / minutes * 90, 45, 290))
            acc90 = float(np.clip(acc / minutes * 90, 65, 185))
            dec90 = float(np.clip(dec / minutes * 90, 65, 180))

            # Player Intensity Index, derived (not sampled) from the
            # per-90 motor values, following the thesis formula exactly:
            # PII = (HSR + 1.5*Sprint + 2*ACC + 2*DEC) / (TD / M), where
            # HSR/Sprint/ACC/DEC/TD are RAW match totals and M is the raw
            # minutes played. Substituting X_raw = X90 * M / 90 for every
            # raw quantity and simplifying gives the algebraically
            # equivalent form below, expressed directly in terms of the
            # per-90 columns actually shipped in the dataset:
            #
            #   PII = M * (HSR90 + 1.5*Sprint90 + 2*ACC90 + 2*DEC90) / TD90
            #
            # Note this is genuinely NOT rate-invariant: two appearances
            # with identical per-90 motor rates but different minutes
            # played (M) will get different PII values under the
            # published formula, because the numerator is built from raw
            # (not per-90) totals while the denominator is already a rate.
            # That is a property of the formula as published, not an
            # artefact of this implementation -- see the Methodology page.
            numerator_90 = hsr90 + 1.5 * sprint90 + 2 * acc90 + 2 * dec90
            pii = minutes * numerator_90 / td90 if td90 > 0 else np.nan

            # Technical/tactical actions -- latent skill driven, with a
            # small, deliberate coupling to the hidden motor state so the
            # two data domains are not fully independent (consistent with
            # the thesis noting partial, not total, independence).
            state_tech_boost = 1.0 + 0.05 * (state - 1)
            def _rate(mean, sd, floor=0.0):
                return max(float(rng.normal(mean, sd)) * state_tech_boost, floor)

            goals = _rate(0.55 * skill["finishing"], 0.10)
            finishing_actions = _rate(2.6 * skill["finishing"], 0.4)
            key_passes = _rate(3.4 * skill["creativity"], 0.5)
            chance_assists = _rate(1.0 * skill["creativity"], 0.2)
            assists = _rate(0.42 * skill["creativity"], 0.09)
            off_duels_won = _rate(5.2 * skill["individual"], 0.8)
            key_individual_actions = _rate(2.6 * skill["individual"], 0.5)
            recoveries = _rate(7.8 * skill["defence"], 1.1)
            interventions = _rate(5.0 * skill["defence"], 0.9)
            def_duels_won = _rate(5.0 * skill["defence"], 0.9)
            blocks = _rate(2.2 * skill["box_defence"], 0.5)
            box_actions = _rate(2.2 * skill["box_defence"], 0.5)
            key_losses = _rate(2.6 * (1 - skill["discipline"]), 0.5)
            errors = _rate(1.1 * (1 - skill["discipline"]), 0.3)

            offensive_index = round(max(
                3.0 * goals + 2.0 * assists + 1.0 * key_passes + 2.0 * chance_assists
                + 1.0 * finishing_actions + 0.6 * off_duels_won + 1.0 * key_individual_actions,
                0.0,
            ), 1)
            defensive_index = round(max(
                1.0 * recoveries + 1.4 * interventions + 1.2 * def_duels_won
                + 1.6 * blocks + 1.3 * box_actions - 0.8 * key_losses - 1.0 * errors,
                0.0,
            ), 1)

            # Technical-detail completeness: lower in autumn than spring,
            # mirroring the uneven completeness documented in the thesis.
            detail_prob = 0.55 if rnd == "Autumn" else 0.82
            has_detail = rng.random() < detail_prob

            all_rows.append({
                "player_id": pid,
                "squad": player["squad"],
                "role": role,
                "age": player["age"],
                "round": rnd,
                "appearance_no": t + 1,
                "minutes": round(minutes, 1),
                "distance_per90": round(td90, 1),
                "hsr_per90": round(hsr90, 1),
                "sprint_per90": round(sprint90, 1),
                "acc_per90": round(acc90, 1),
                "dec_per90": round(dec90, 1),
                "pii": round(float(pii), 2) if np.isfinite(pii) else np.nan,
                "offensive_index": offensive_index,
                "defensive_index": defensive_index,
                "technical_detail_available": bool(has_detail),
                "goals_per90": round(goals, 2) if has_detail else np.nan,
                "finishing_actions_per90": round(finishing_actions, 2) if has_detail else np.nan,
                "key_passes_per90": round(key_passes, 2) if has_detail else np.nan,
                "chance_assists_per90": round(chance_assists, 2) if has_detail else np.nan,
                "assists_per90": round(assists, 2) if has_detail else np.nan,
                "off_duels_won_per90": round(off_duels_won, 2) if has_detail else np.nan,
                "key_individual_actions_per90": round(key_individual_actions, 2) if has_detail else np.nan,
                "recoveries_per90": round(recoveries, 2) if has_detail else np.nan,
                "interventions_per90": round(interventions, 2) if has_detail else np.nan,
                "def_duels_won_per90": round(def_duels_won, 2) if has_detail else np.nan,
                "blocks_per90": round(blocks, 2) if has_detail else np.nan,
                "box_actions_per90": round(box_actions, 2) if has_detail else np.nan,
                "key_losses_per90": round(key_losses, 2) if has_detail else np.nan,
                "errors_per90": round(errors, 2) if has_detail else np.nan,
            })

    data = pd.DataFrame(all_rows)

    # Apply the same minimum-minutes qualifying threshold used in the
    # thesis (>= 20 minutes) before the dataset is used anywhere downstream.
    n_before = len(data)
    data = data[data["minutes"] >= 20.0].copy()
    n_after = len(data)
    data.attrs["n_before_minutes_filter"] = n_before
    data.attrs["n_after_minutes_filter"] = n_after

    # Re-sequence appearance_no after filtering so it is contiguous again.
    data = data.sort_values(["player_id", "round", "appearance_no"], kind="stable")
    data["appearance_no"] = data.groupby("player_id").cumcount() + 1
    data = data.reset_index(drop=True)

    return data


def main():
    df = generate_dataset()
    out_path = "data/synthetic_player_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows, {df.player_id.nunique()} players -> {out_path}")
    print(
        "Minutes-threshold filter removed",
        df.attrs.get("n_before_minutes_filter", "?"),
        "->",
        df.attrs.get("n_after_minutes_filter", "?"),
    )


if __name__ == "__main__":
    main()
