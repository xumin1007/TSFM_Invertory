import numpy as np

from f2d.run_known_truth_censoring import closed_loop_table, static_table


def test_known_truth_censoring_bias_and_ranking_reversal():
    static = static_table()
    assert np.allclose(
        static.differential_bias, static.closed_form_wedge, atol=1e-10)
    assert (static.loc[static.logging_cap >= 15, "differential_bias"] == 0).all()
    assert static.ranking_reversal.any()

    dynamic = closed_loop_table(n_paths=300)
    assert np.allclose(
        dynamic.loc[dynamic.logging_target >= 20, "differential_bias"], 0)
    assert dynamic.loc[dynamic.logging_target == 17,
                       "ranking_reversal"].item()
