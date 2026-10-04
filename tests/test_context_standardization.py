"""
Verifies that contextual standardisation (z-scoring within round x squad x
role) actually produces, within each group, a mean of ~0 and a standard
deviation of ~1 -- and that it is computed with the grouping the thesis
actually describes (round, squad category, and broad role together, not
role alone).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src import data_pipeline as dp


def test_context_group_cols_include_round():
    assert set(dp.CONTEXT_GROUP_COLS) == {"round", "squad", "role"}, (
        "Main-variant standardisation must group by round + squad + role together, "
        "matching the thesis's own description of its main motor-variant standardisation."
    )


def test_group_mean_and_std_after_standardization():
    df = dp.load_raw_data("data/synthetic_player_data.csv")
    df = dp.add_context_standardized_columns(df, dp.MOTOR_COLS)

    for col in dp.MOTOR_COLS:
        zcol = col + "_z"
        grouped = df.groupby(dp.CONTEXT_GROUP_COLS)[zcol]
        means = grouped.mean()
        stds = grouped.std(ddof=0)
        assert np.allclose(means, 0, atol=1e-8), f"{zcol}: group means are not ~0"
        # Groups with only one member have an undefined std (division
        # guarded to 1.0 in the implementation); only check groups with
        # more than one observation.
        sizes = grouped.size()
        multi_member_groups = sizes[sizes > 1].index
        checkable_stds = stds.loc[stds.index.isin(multi_member_groups)]
        assert np.allclose(checkable_stds, 1, atol=1e-6), f"{zcol}: group stds are not ~1"


def test_standardization_removes_role_confound_more_than_baseline():
    """
    The thesis's key finding: without context control, cluster structure
    mainly reproduces broad playing role; after contextual standardisation,
    that association drops sharply. This test checks the same qualitative
    direction holds on the synthetic pipeline (not the same magnitude --
    a different, independent dataset).
    """
    from src.stats_utils import cramers_v

    bundle = dp.build_full_bundle("data/synthetic_player_data.csv")
    df = bundle["df"]

    X_main = df[["PC1", "PC2", "PC3"]].to_numpy()
    X_base = df[["PC1_baseline", "PC2_baseline", "PC3_baseline"]].to_numpy()
    _, labels_main = dp.fit_kmeans(X_main, k=2)
    _, labels_base = dp.fit_kmeans(X_base, k=4)

    import pandas as pd
    v_main = cramers_v(df.role, pd.Series(labels_main, index=df.index))
    v_base = cramers_v(df.role, pd.Series(labels_base, index=df.index))

    assert v_main < v_base, (
        f"Expected the context-standardized variant (Cramer's V={v_main:.3f}) to show a weaker "
        f"association with role than the no-context baseline (Cramer's V={v_base:.3f})"
    )


if __name__ == "__main__":
    test_context_group_cols_include_round()
    test_group_mean_and_std_after_standardization()
    test_standardization_removes_role_confound_more_than_baseline()
    print("test_context_standardization.py: all tests passed")
