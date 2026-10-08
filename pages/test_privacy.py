"""
Regression guard for the most important requirement in this project: no
individual-level data from the real (pseudonymised) thesis dataset is
published anywhere in this public application.

The thesis identifies individual players with codes of the exact form
"P" + three digits (e.g. "P001", "P023"). This application's fictional
players use a deliberately different scheme ("FPL" + two digits) so the
two can never be confused, and this test enforces that distinction
structurally rather than relying on anyone remembering to keep it that way:

  1. No player_id in the shipped synthetic dataset matches the thesis's
     pseudonym pattern.
  2. No string value anywhere in `thesis_results.py` (which is allowed to
     contain real thesis findings, but only in aggregate form) matches
     that pattern either -- i.e. nobody has reintroduced an individual
     thesis example (a specific player's stability, hybridity, or
     similarity score) into the frozen constants module.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src import thesis_results as T

THESIS_PSEUDONYM_PATTERN = re.compile(r"^P\d{3}$")


def _iter_strings(obj):
    """Recursively yield every string found inside a (possibly nested) object."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_strings(k)
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from _iter_strings(item)


def test_no_thesis_pseudonyms_in_synthetic_dataset():
    df = pd.read_csv("data/synthetic_player_data.csv")
    offenders = [pid for pid in df["player_id"].unique() if THESIS_PSEUDONYM_PATTERN.match(str(pid))]
    assert not offenders, f"Synthetic dataset contains IDs matching the thesis's pseudonym scheme: {offenders}"
    # Also confirm the dataset uses the expected, visibly distinct scheme.
    assert all(str(pid).startswith("FPL") for pid in df["player_id"].unique())


def test_no_thesis_pseudonyms_in_frozen_constants():
    public_names = [name for name in dir(T) if not name.startswith("_")]
    offenders = []
    for name in public_names:
        value = getattr(T, name)
        for s in _iter_strings(value):
            if THESIS_PSEUDONYM_PATTERN.match(s):
                offenders.append((name, s))
    assert not offenders, (
        f"thesis_results.py contains individual thesis pseudonyms, which should never be "
        f"published even in an otherwise-aggregate module: {offenders}"
    )


def test_dynamic_dna_cohort_constant_is_aggregate_only():
    """
    Specifically guard the constant that replaced the removed individual
    per-player examples: it must only contain cohort-level summary
    numbers, never a per-player breakdown.
    """
    cohort = T.DYNAMIC_DNA_COHORT
    expected_keys = {"n_players", "min_appearances", "mean_stability", "mean_hybridity", "mean_dominant_probability"}
    assert set(cohort.keys()) == expected_keys, (
        f"DYNAMIC_DNA_COHORT has unexpected keys: {set(cohort.keys())} -- if someone added a "
        f"per-player field here, that would reintroduce individual-level thesis data."
    )
    assert not hasattr(T, "DYNAMIC_DNA_EXAMPLES"), "Individual-level DYNAMIC_DNA_EXAMPLES must not exist."
    assert not hasattr(T, "DYNAMIC_DNA_WORKED_EXAMPLE"), "Individual-level DYNAMIC_DNA_WORKED_EXAMPLE must not exist."
    assert not hasattr(T, "SIMILARITY_EXAMPLES"), "Individual-level SIMILARITY_EXAMPLES must not exist."


if __name__ == "__main__":
    test_no_thesis_pseudonyms_in_synthetic_dataset()
    test_no_thesis_pseudonyms_in_frozen_constants()
    test_dynamic_dna_cohort_constant_is_aggregate_only()
    print("test_privacy.py: all tests passed")
