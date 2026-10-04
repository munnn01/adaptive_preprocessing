import math
import numpy as np
import pytest

from adaptive_vcm.metrics import curve_summary, paired_ar_bootstrap, pchip_bd
from adaptive_vcm.selection import Observation, ar_guard, detection_distance, select


def test_minimum_real_bytes_with_all_teacher_guards():
    rows = [Observation("identity", 1000, (0, 0), (True, True)),
            Observation("unsafe", 400, (.01, .2), (True, True)),
            Observation("safe", 750, (.01, -.2), (True, True)),
            Observation("flip", 600, (0, 0), (True, False)),
            Observation("missing", 100, (), ())]
    assert select(rows, .1) == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_teacher_fails_closed(bad):
    rows = [Observation("identity", 100, (0,), (True,)), Observation("broken", 50, (bad,), (True,))]
    assert select(rows, .1) == 0


def test_identity_fallback_rejects_bit_overhead_and_insufficient_savings():
    assert select([Observation("identity", 100, (0,), (True,)),
                   Observation("higher", 101, (0,), (True,)),
                   Observation("same", 100, (0,), (True,))], .1) == 0


def test_ar_guard_rejects_confident_source_class_flip_without_labels():
    source = np.array([.8, .1, .1])
    anchor = np.array([.7, .2, .1])
    trial = np.array([.2, .7, .1])
    distance, decision = ar_guard(source, anchor, trial)
    assert distance > 0 and not decision
    assert ar_guard(source, anchor, anchor) == (0, True)


def detections(boxes, scores, labels):
    return {"boxes": np.array(boxes).reshape(-1, 4), "scores": np.array(scores), "labels": np.array(labels)}


def test_detection_loss_matches_objects_uniquely_and_respects_classes():
    source = detections([[0, 0, 10, 10], [0, 0, 10, 10]], [.9, .9], [1, 1])
    trial = detections([[0, 0, 10, 10]], [.9], [1])
    assert detection_distance(source, source) == 0
    assert detection_distance(source, trial) == pytest.approx(.5)
    wrong = detections([[0, 0, 10, 10]], [.9], [2])
    assert detection_distance(source, wrong) == 1
    assert math.isinf(detection_distance(detections([], [], []), trial))


def test_known_bd_savings_and_strict_target():
    rate = np.array([.1, .2, .4, .8, 1.6])
    quality = [.3, .4, .5, .6, .7]
    summary = curve_summary(rate, quality, rate * .8, quality,
                            ci={"hi": -15, "finite_fraction": 1})
    assert summary["bd_rate_pct"] == pytest.approx(-20, abs=1e-7)
    assert summary["pchip_bd_rate_pct"] == pytest.approx(-20, abs=1e-7)
    assert summary["screen_passes"]
    assert not summary["target_confirmed"]


def test_degenerate_and_nonoverlap_curves_cannot_pass():
    rates = [.1, .2, .4, .8]
    for quality in ([.1] * 4, [.8, .85, .9, .95]):
        summary = curve_summary(rates, [.1, .2, .3, .4], np.array(rates) * .5, quality)
        assert summary["bd_rate_pct"] is None
        assert not summary["point_passes"]
    with pytest.raises(ValueError):
        curve_summary([.1, 0, .3], [.1, .2, .3], [.1, .2, .3], [.1, .2, .3])


def test_bootstrap_keeps_qps_and_arms_paired_and_rejects_missing_source():
    rows = [{"id": str(i), "qp": qp, "arm": arm, "bpp": rate * (.8 if arm == "adaptive" else 1),
             "correct": {"model": int(i < (j + 1) * 2)}}
            for i in range(12) for j, (qp, rate) in enumerate(zip([30, 35, 40, 45, 50], [.8, .6, .4, .2, .1]))
            for arm in ("anchor", "adaptive")]
    ci = paired_ar_bootstrap(rows, "model", [30, 35, 40, 45, 50], 50)
    assert ci["hi"] == pytest.approx(-20, abs=1e-6)
    with pytest.raises(ValueError):
        paired_ar_bootstrap(rows[:-1], "model", [30, 35, 40, 45, 50], 50)


def test_pchip_plateau_handling_is_explicit_and_finite():
    assert math.isfinite(pchip_bd([1, 2, 3, 4], [.1, .2, .2, .3], [1, 2, 3, 4], [.1, .2, .2, .3]))
