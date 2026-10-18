import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from skewguard.stats import (
    TopN,
    compare_numeric,
    js_divergence,
    linf_bounds,
    new_category_mass,
    null_rate_pvalue,
    psi,
    rebin,
)

# ---------- rebin ----------


def test_rebin_same_edges_returns_same_counts():
    edges = [0.0, 1.0, 2.0, 3.0]
    r = rebin(edges, [5, 10, 5], edges)
    assert r.counts == pytest.approx([5, 10, 5])
    assert r.out_of_range == pytest.approx(0)


def test_rebin_splits_mass_uniformly_within_a_bin():
    r = rebin([0.0, 2.0], [10], [0.0, 1.0, 2.0])
    assert r.counts == pytest.approx([5, 5])


def test_rebin_conserves_total_mass_when_ranges_match():
    src = [0.0, 0.3, 0.9, 1.0]
    dst = [0.0, 0.25, 0.5, 0.75, 1.0]
    r = rebin(src, [3, 6, 1], dst)
    assert sum(r.counts) + r.out_of_range == pytest.approx(10)


def test_rebin_reports_mass_outside_target_range():
    # baseline covers [0,1]; current window was multiplied by 100 (unit bug)
    r = rebin([0.0, 50.0, 100.0], [4, 6], [0.0, 0.5, 1.0])
    assert r.out_of_range == pytest.approx(10 - 4 * (1 / 50))
    assert sum(r.counts) == pytest.approx(4 * (1 / 50))


def test_rebin_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        rebin([0.0, 1.0], [1, 2], [0.0, 1.0])


def test_rebin_splits_out_of_range_into_below_and_above():
    r = rebin([-1.0, 3.0], [8], [0.0, 1.0, 2.0])
    assert r.below == pytest.approx(2)
    assert r.above == pytest.approx(2)
    assert r.counts == pytest.approx([2, 2])


def test_rebin_zero_width_source_bin_is_a_point_mass():
    # DuckDB/Postgres store a constant column as {"bins": [v, v], "counts": [n]}
    r = rebin([0.5, 0.5], [7], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert r.counts == pytest.approx([0, 0, 7, 0])


def test_rebin_point_mass_on_last_edge_lands_in_last_bin():
    r = rebin([1.0, 1.0], [3], [0.0, 0.5, 1.0])
    assert r.counts == pytest.approx([0, 3])
    assert r.out_of_range == 0


def test_rebin_onto_zero_width_target_keeps_matching_point_mass():
    r = rebin([2.0, 2.0], [5], [2.0, 2.0])
    assert r.counts == pytest.approx([5])


def test_rebin_spread_onto_zero_width_target_is_all_out_of_range():
    # baseline was constant at 2.0; the window now varies around it
    r = rebin([1.0, 3.0], [10], [2.0, 2.0])
    assert r.counts == pytest.approx([0])
    assert (r.below, r.above) == pytest.approx((5, 5))


def test_rebin_rejects_decreasing_edges():
    with pytest.raises(ValueError):
        rebin([0.0, 2.0, 1.0], [1, 1], [0.0, 1.0])


edges_strategy = st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=2, max_size=12).map(
    sorted
)


@given(
    src=edges_strategy,
    dst=edges_strategy,
    data=st.data(),
)
def test_rebin_always_conserves_mass(src, dst, data):
    counts = data.draw(st.lists(st.integers(0, 1000), min_size=len(src) - 1, max_size=len(src) - 1))
    r = rebin(src, counts, dst)
    assert r.total == pytest.approx(sum(counts), rel=1e-9, abs=1e-6)
    assert min(r.counts) >= 0 and r.below >= 0 and r.above >= 0


# ---------- psi ----------


def test_psi_is_zero_for_identical_distributions():
    assert psi([10, 20, 30], [10, 20, 30]) == pytest.approx(0)


def test_psi_ignores_sample_size_scaling():
    assert psi([10, 20, 30], [100, 200, 300]) == pytest.approx(0)


def test_psi_is_symmetric():
    a, b = [10, 20, 30], [30, 20, 10]
    assert psi(a, b) == pytest.approx(psi(b, a))


def test_psi_large_shift_exceeds_common_alert_level():
    assert psi([90, 10, 0, 0], [0, 0, 10, 90]) > 0.25


def test_psi_is_finite_with_empty_bins():
    assert math.isfinite(psi([10, 0, 10], [0, 20, 0]))


def test_psi_rejects_empty_distribution():
    with pytest.raises(ValueError):
        psi([0, 0], [1, 1])


def test_psi_matches_hand_computation():
    # (0.5-0.25)ln(0.5/0.25) + (0.5-0.75)ln(0.5/0.75)
    expected = 0.25 * math.log(2) - 0.25 * math.log(2 / 3)
    assert psi([1, 3], [2, 2]) == pytest.approx(expected)


counts_strategy = st.lists(st.integers(0, 500), min_size=2, max_size=25).filter(
    lambda c: sum(c) > 0
)


@given(st.data())
def test_psi_is_non_negative_and_symmetric(data):
    a = data.draw(counts_strategy)
    b = data.draw(st.lists(st.integers(0, 500), min_size=len(a), max_size=len(a)).filter(sum))
    assert psi(a, b) >= -1e-12
    assert psi(a, b) == pytest.approx(psi(b, a))


# ---------- js ----------


def test_js_is_zero_for_identical_and_one_for_disjoint():
    assert js_divergence([1, 2, 3], [2, 4, 6]) == pytest.approx(0)
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1)


@given(st.data())
def test_js_is_bounded_and_symmetric(data):
    a = data.draw(counts_strategy)
    b = data.draw(st.lists(st.integers(0, 500), min_size=len(a), max_size=len(a)).filter(sum))
    assert 0 <= js_divergence(a, b) <= 1
    assert js_divergence(a, b) == pytest.approx(js_divergence(b, a))


# ---------- compare_numeric ----------


def test_compare_numeric_is_zero_when_window_equals_baseline():
    edges, counts = [0.0, 1.0, 2.0], [4, 6]
    p, j, r = compare_numeric(edges, counts, edges, counts)
    assert p == pytest.approx(0, abs=1e-12)
    assert j == pytest.approx(0, abs=1e-12)
    assert r.out_of_range == 0


def test_compare_numeric_counts_out_of_range_mass_as_drift():
    base = ([0.0, 0.5, 1.0], [50, 50])
    shifted = ([50.0, 75.0, 100.0], [50, 50])  # x100 unit change
    p, j, r = compare_numeric(*base, *shifted)
    assert r.above == pytest.approx(100)
    assert p > 5
    assert j == pytest.approx(1)


# ---------- categorical ----------


def test_linf_exact_when_both_sides_fully_enumerated():
    base = TopN({"a": 50, "b": 50})
    win = TopN({"a": 80, "b": 20})
    assert linf_bounds(base, win) == pytest.approx((0.3, 0.3))


def test_linf_bounds_widen_for_values_hidden_in_other():
    # "c" is in the window's top-N but somewhere in the baseline's other bucket (<= 10)
    base = TopN({"a": 60, "b": 30}, other=10)
    win = TopN({"a": 60, "c": 40})
    lower, upper = linf_bounds(base, win)
    assert lower == pytest.approx(0.4 - 0.1)  # c: window 0.4, baseline at most 0.1
    assert upper >= lower


def test_new_category_mass_only_when_baseline_fully_known():
    win = TopN({"a": 70, "z": 30})
    assert new_category_mass(TopN({"a": 100}), win) == pytest.approx(0.3)
    assert new_category_mass(TopN({"a": 90}, other=10), win) == 0.0


def test_topn_from_feast_histogram():
    h = {"values": [{"value": "x", "count": 7}, {"value": "y", "count": 3}], "other_count": 2}
    t = TopN.from_feast(h)
    assert t.total == 12 and t.counts == {"x": 7.0, "y": 3.0} and t.other == 2


# ---------- null rate ----------


def test_null_rate_pvalue_flags_a_null_burst():
    assert null_rate_pvalue(10, 1000, 30, 100) < 1e-6


def test_null_rate_pvalue_ignores_equal_rates_and_drops():
    assert null_rate_pvalue(10, 1000, 1, 100) > 0.4
    assert null_rate_pvalue(100, 1000, 0, 100) > 0.9


def test_null_rate_pvalue_handles_no_nulls_anywhere():
    assert null_rate_pvalue(0, 1000, 0, 100) == 1.0
