import numpy as np
import pytest

from skewguard.stats import TopN, compare_numeric
from skewguard.thresholds import (
    categorical_null,
    compare_batch,
    feast_histogram,
    is_point_style,
    numeric_null,
    sample_histogram,
    sample_own_histograms,
    simulate_reference,
)


def test_feast_histogram_matches_numpy_and_point_style():
    values = np.array([0.0, 0.5, 1.0])
    edges, counts = feast_histogram(values, bins=2)
    assert edges == [0.0, 0.5, 1.0] and counts == [1.0, 2.0]
    assert feast_histogram(np.array([3.0, 3.0]), point_style=True) == ([3.0, 3.0], [2.0])
    assert is_point_style([3.0, 3.0]) and not is_point_style([3.0, 4.0])


def test_sample_histogram_stays_inside_occupied_bins():
    rng = np.random.default_rng(1)
    x = sample_histogram([0.0, 1.0, 2.0, 3.0], [0, 5, 0], 1000, rng)
    assert x.min() >= 1.0 and x.max() < 2.0


def test_noise_floor_shrinks_with_window_size():
    edges, counts = feast_histogram(np.random.default_rng(0).random(1808))
    small = numeric_null(edges, counts, 116, reps=300)[0].quantile(0.95)
    large = numeric_null(edges, counts, 5000, reps=300)[0].quantile(0.95)
    assert small > 3 * large


def test_fixed_psi_threshold_misfires_on_small_clean_windows():
    """The motivating finding: PSI 0.25 alerts on >5% of clean 116-row windows."""
    edges, counts = feast_histogram(np.random.default_rng(0).random(1808))
    null = numeric_null(edges, counts, 116, reps=500)[0]
    false_alarm_rate = np.mean(np.asarray(null.values) > 0.25)
    assert false_alarm_rate > 0.05


@pytest.mark.parametrize("n_window", [116, 1000])
def test_simulated_threshold_controls_false_positives_on_fresh_data(n_window):
    """Calibration: on truly independent clean samples, the 1 - alpha quantile
    should alert on roughly alpha of windows (here alpha = 5%)."""
    rng = np.random.default_rng(42)
    draw = lambda n: rng.beta(2, 5, size=n)  # noqa: E731 - a skewed, smooth distribution
    base_edges, base_counts = feast_histogram(draw(1808))
    threshold = numeric_null(base_edges, base_counts, n_window, reps=400)[0].quantile(0.95)

    trials = 300
    alerts = 0
    for _ in range(trials):
        we, wc = feast_histogram(draw(n_window))
        alerts += compare_numeric(base_edges, base_counts, we, wc)[0] > threshold
    # Monte Carlo slack: binomial sd at 5% over 300 trials is ~1.3%
    assert alerts / trials < 0.10


def test_null_is_cached_and_deterministic():
    edges, counts = [0.0, 1.0, 2.0], [10.0, 10.0]
    a = numeric_null(edges, counts, 50, reps=50, seed=3)
    b = numeric_null(edges, counts, 50, reps=50, seed=3)
    assert a is b


def test_pvalue_uses_plus_one_correction():
    null = numeric_null([0.0, 1.0, 2.0], [10.0, 10.0], 50, reps=50)[0]
    assert null.pvalue(1e9) == pytest.approx(1 / 51)


def test_categorical_null_quantile_shrinks_with_window_size():
    base = TopN({"a": 500, "b": 300, "c": 200})
    small = categorical_null(base, 50, reps=300).quantile(0.99)
    large = categorical_null(base, 5000, reps=300).quantile(0.99)
    assert small > 2 * large > 0


def test_compare_batch_matches_scalar_compare_numeric():
    rng = np.random.default_rng(5)
    base_e, base_c, win_e, win_c = [], [], [], []
    for _ in range(50):
        be, bc = feast_histogram(rng.normal(0, 1, 300))
        we, wc = feast_histogram(rng.normal(rng.uniform(-1, 1), rng.uniform(0.5, 3), 80))
        base_e.append(be), base_c.append(bc), win_e.append(we), win_c.append(wc)
    psi_b, js_b = compare_batch(*map(np.asarray, (base_e, base_c, win_e, win_c)))
    for i in range(50):
        p, j, _ = compare_numeric(base_e[i], base_c[i], win_e[i], win_c[i])
        assert psi_b[i] == pytest.approx(p, rel=1e-9, abs=1e-12)
        assert js_b[i] == pytest.approx(j, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("n", [1, 2, 7, 125, 10_000])
def test_sample_own_histograms_shapes_and_totals(n):
    rng = np.random.default_rng(0)
    edges, counts = [0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 3.0, 1.0]
    own, own_counts = sample_own_histograms(edges, counts, n, 300, rng)
    assert own.shape == (300, 21) and own_counts.shape == (300, 20)
    assert np.all(own_counts.sum(axis=1) == n)
    assert np.all(own_counts >= 0)
    assert np.all(np.diff(own, axis=1) > 0)
    if n > 1:  # the sample's own range stays inside the occupied source bins
        assert own[:, 0].min() >= 0.0 and own[:, -1].max() <= 4.0
        assert np.all(own_counts[:, 0] >= 1) and np.all(own_counts[:, -1] >= 1)


@pytest.mark.parametrize("n_window", [30, 125, 1000])
def test_fast_null_matches_reference_simulation(n_window):
    """The O(bins^2) sampler must give the same null distribution as drawing every value."""
    from scipy.stats import ks_2samp

    edges, counts = feast_histogram(np.random.default_rng(11).beta(2, 5, 1808))
    ref_psi, ref_js = simulate_reference(edges, counts, 1808, n_window, reps=1500, seed=1)
    fast_psi, fast_js = numeric_null(edges, counts, n_window, reps=1500, seed=2)
    assert ks_2samp(ref_psi, fast_psi.values).pvalue > 0.001
    assert ks_2samp(ref_js, fast_js.values).pvalue > 0.001


def test_fast_null_cost_does_not_grow_with_window_size():
    edges, counts = feast_histogram(np.random.default_rng(0).random(1808))
    null = numeric_null(edges, counts, 10_000_000, reps=200)[0]
    assert all(np.isfinite(null.values)) and null.quantile(0.99) < 0.05


def test_point_style_baseline_has_zero_null():
    null_psi, _ = numeric_null([2.0, 2.0], [500.0], 50, reps=20)
    assert set(null_psi.values) == {0.0}


def test_null_rejects_empty_inputs():
    with pytest.raises(ValueError):
        numeric_null([0.0, 1.0], [0.0], 10)
    with pytest.raises(ValueError):
        numeric_null([0.0, 1.0], [5.0], 0)
