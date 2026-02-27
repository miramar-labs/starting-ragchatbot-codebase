"""Deliberately failing test — demonstrates that branch protection blocks merging."""


def test_this_always_fails():
    assert 1 == 2, "This test is intentionally broken to show CI enforcement"
