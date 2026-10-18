import numpy as np
import pytest

from skewguard.stats import TopN, compare_numeric
from skewguard.thresholds import (
    categorical_null,
    feast_histogram,
    is_point_style,
    numeric_null,
    sample_histogram,
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


def test_null_rejects_empty_inputs():
    with pytest.raises(ValueError):
        numeric_null([0.0, 1.0], [0.0], 10)
    with pytest.raises(ValueError):
        numeric_null([0.0, 1.0], [5.0], 0)
