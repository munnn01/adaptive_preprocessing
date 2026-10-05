import pytest


def test_ci_comparison_accepts_cross_platform_roundoff_but_rejects_changed_bound():
    from scripts.audit_v25 import assert_ci_matches
    original = {'lo': -10.881806660224484, 'hi': 1.4121557701797323,
                'draws': 2000, 'finite_draws': 2000, 'finite_fraction': 1.,
                'method': 'paired_source_video_bootstrap'}
    replayed = {**original, 'hi': 1.4121557701797374}
    assert_ci_matches(replayed, original)
    with pytest.raises(AssertionError):
        assert_ci_matches({**original, 'hi': 1.42}, original)
    with pytest.raises(AssertionError):
        assert_ci_matches({**original, 'finite_draws': 1999}, original)
