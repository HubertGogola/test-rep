"""
Verifies the Kaplan-Meier implementation against a small example worked
out by hand, and checks basic correctness properties (monotonically
non-increasing survival curve, correct event/censoring counts).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.stats_utils import kaplan_meier


def test_kaplan_meier_hand_worked_example():
    # 10 subjects. Hand computation:
    # t=3: at_risk=10, events=1 -> S=0.9000
    # t=5: at_risk=9,  events=1 (1 censored same time) -> S=0.9*(1-1/9)=0.8000
    # t=8: at_risk=7,  events=3 -> S=0.8*(1-3/7)=0.457142857...
    # t=12: at_risk=4, events=0 (1 censored) -> S unchanged = 0.457142857
    # t=15: at_risk=3, events=1 -> S=0.457142857*(2/3)=0.304761905
    # t=20: at_risk=2, events=1 (1 censored same time) -> S=0.304761905*0.5=0.152380952
    durations = [3, 5, 5, 8, 8, 8, 12, 15, 20, 20]
    observed = [True, True, False, True, True, True, False, True, False, True]

    result = kaplan_meier(durations, observed)

    assert result["n"] == 10
    assert result["n_events"] == 7
    assert result["n_censored"] == 3
    assert result["median"] == 8  # first time survival <= 0.5

    table = result["table"].set_index("time")
    assert abs(table.loc[3.0, "survival"] - 0.9) < 1e-9
    assert abs(table.loc[5.0, "survival"] - 0.8) < 1e-9
    assert abs(table.loc[8.0, "survival"] - (0.8 * (1 - 3 / 7))) < 1e-9
    assert abs(table.loc[15.0, "survival"] - (0.8 * (1 - 3 / 7) * (2 / 3))) < 1e-9
    assert abs(table.loc[20.0, "survival"] - (0.8 * (1 - 3 / 7) * (2 / 3) * 0.5)) < 1e-9


def test_kaplan_meier_survival_curve_is_non_increasing():
    rng = np.random.default_rng(0)
    durations = rng.integers(1, 30, size=50)
    observed = rng.random(50) < 0.7
    result = kaplan_meier(durations, observed)
    ys = np.array(result["ys"])
    assert np.all(np.diff(ys) <= 1e-12), "Survival curve must be non-increasing"
    assert ys[0] == 1.0


def test_kaplan_meier_no_events_gives_no_median():
    durations = [5, 10, 15]
    observed = [False, False, False]
    result = kaplan_meier(durations, observed)
    assert result["median"] is None
    assert result["n_events"] == 0
    assert result["n_censored"] == 3


if __name__ == "__main__":
    test_kaplan_meier_hand_worked_example()
    test_kaplan_meier_survival_curve_is_non_increasing()
    test_kaplan_meier_no_events_gives_no_median()
    print("test_survival_analysis.py: all tests passed")
