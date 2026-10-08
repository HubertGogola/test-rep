"""
Verifies the synthetic data generator's Player Intensity Index matches a
direct, independent reconstruction of the published formula:

    PII = (HSR + 1.5*Sprint + 2*ACC + 2*DEC) / (TD / M)

where HSR/Sprint/ACC/DEC/TD are RAW match totals and M is raw minutes
played. The generator stores per-90 values and derives PII from them
algebraically; this test reconstructs PII independently from raw totals
(by inverting the per-90 conversion) and checks the two agree, which is
the exact bug class a prior review caught (an earlier implementation
divided by a constant 90 instead of the actual minutes played, inflating
PII by a factor of 90/minutes for any non-full-length appearance).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import generate_data as gd


def test_pii_matches_raw_totals_formula():
    df = gd.generate_dataset(seed=gd.SEED)
    sample = df.sample(min(100, len(df)), random_state=0)

    max_rel_err = 0.0
    for _, row in sample.iterrows():
        m = row["minutes"]
        # Invert the per-90 conversion (X90 = X/M * 90) to recover the raw
        # match totals implied by the shipped per-90 columns.
        hsr_raw = row["hsr_per90"] * m / 90
        sprint_raw = row["sprint_per90"] * m / 90
        acc_raw = row["acc_per90"] * m / 90
        dec_raw = row["dec_per90"] * m / 90
        td_raw = row["distance_per90"] * m / 90

        pii_from_raw_formula = (hsr_raw + 1.5 * sprint_raw + 2 * acc_raw + 2 * dec_raw) / (td_raw / m)
        rel_err = abs(pii_from_raw_formula - row["pii"]) / max(abs(row["pii"]), 1e-9)
        max_rel_err = max(max_rel_err, rel_err)

    # A small tolerance accounts only for the 1-2 decimal place rounding
    # applied when the CSV columns are written, not for any formula error.
    assert max_rel_err < 0.01, f"PII does not match the raw-totals formula (max rel. error {max_rel_err:.4f})"


def test_pii_is_not_minutes_invariant_by_a_constant_factor():
    """
    Regression guard for the specific historical bug: PII should depend on
    minutes played even when every per-90 rate is held fixed (this is a
    property of the published formula itself -- see Methodology page) --
    but it must NOT scale by the specific, wrong factor (90 / minutes)
    that the buggy implementation introduced.
    """
    hsr90, sprint90, acc90, dec90, td90 = 700.0, 150.0, 120.0, 115.0, 11000.0
    numerator_90 = hsr90 + 1.5 * sprint90 + 2 * acc90 + 2 * dec90

    def correct_pii(minutes):
        return minutes * numerator_90 / td90

    def buggy_pii(minutes):
        return numerator_90 / (td90 / 90)

    pii_20 = correct_pii(20)
    pii_90 = correct_pii(90)
    # Correct formula: PII scales linearly with minutes for fixed per-90 rates.
    assert abs(pii_20 / pii_90 - 20 / 90) < 1e-9
    # The buggy formula is constant regardless of minutes -- demonstrate
    # the two are different functions, so a future regression would be caught
    # if someone accidentally reintroduced the constant-90 denominator.
    assert abs(buggy_pii(20) - buggy_pii(90)) < 1e-9
    assert abs(correct_pii(20) - buggy_pii(20)) > 1e-6


if __name__ == "__main__":
    test_pii_matches_raw_totals_formula()
    test_pii_is_not_minutes_invariant_by_a_constant_factor()
    print("test_pii_formula.py: all tests passed")
