from datetime import timedelta

import pytest

from skewguard.policy import Policy, PolicySet, parse_duration


def test_parse_duration_forms():
    assert parse_duration("90m") == timedelta(minutes=90)
    assert parse_duration("26h") == timedelta(hours=26)
    assert parse_duration("1.5d") == timedelta(days=1.5)
    assert parse_duration(60) == timedelta(seconds=60)
    assert parse_duration(None) is None
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_feature_overrides_inherit_view_then_default():
    ps = PolicySet.from_dict(
        {
            "default": {"false_positive_rate": 0.02, "max_staleness": "26h"},
            "features": {
                "drivers": {"max_staleness": "2h"},
                "drivers.conv_rate": {"min_rows": 100},
            },
        }
    )
    assert ps.for_feature("other", "x") == ps.default
    assert ps.for_feature("drivers", "acc_rate").max_staleness == timedelta(hours=2)
    conv = ps.for_feature("drivers", "conv_rate")
    assert conv.min_rows == 100
    assert conv.max_staleness == timedelta(hours=2)
    assert conv.false_positive_rate == 0.02


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown policy keys"):
        PolicySet.from_dict({"default": {"false_positve_rate": 0.1}})
    with pytest.raises(ValueError, match="unknown top-level keys"):
        PolicySet.from_dict({"defaults": {}})


def test_from_yaml(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text("default:\n  statistic: js\n  max_staleness: 2d\n", encoding="utf-8")
    ps = PolicySet.from_yaml(path)
    assert ps.default == Policy(statistic="js", max_staleness=timedelta(days=2))
