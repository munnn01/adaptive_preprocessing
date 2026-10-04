"""Canonical v1-v21 BD fit plus PCHIP sensitivity and fail-closed target gates."""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import PchipInterpolator

from .legacy_bd import bd_rate, bd_metric


def _curve(rate, quality):
    r, q = np.asarray(rate, np.float64), np.asarray(quality, np.float64)
    if r.shape != q.shape or r.ndim != 1 or len(r) < 3:
        raise ValueError("need at least three paired curve points")
    if not np.isfinite(r).all() or not np.isfinite(q).all() or np.any(r <= 0):
        raise ValueError("invalid curve point")
    return r, q


def pchip_bd(rate_a, quality_a, rate_b, quality_b) -> float:
    """Sensitivity readout: equal-quality plateaus use their minimum bitrate."""
    def fit(rate, quality):
        r, q = _curve(rate, quality)
        unique = np.unique(q)
        if len(unique) < 2:
            return None, None
        log_rate = [np.log(r[q == x].min()) for x in unique]
        return unique, PchipInterpolator(unique, log_rate)
    qa, pa = fit(rate_a, quality_a)
    qb, pb = fit(rate_b, quality_b)
    if pa is None or pb is None:
        return math.nan
    low, high = max(qa.min(), qb.min()), min(qa.max(), qb.max())
    if high <= low:
        return math.nan
    return float(np.expm1((pb.integrate(low, high) - pa.integrate(low, high)) / (high - low)) * 100)


def curve_summary(rate_a, quality_a, rate_b, quality_b, *, ci: dict | None = None,
                  max_gap_pp: float = .5) -> dict:
    ra, qa = _curve(rate_a, quality_a)
    rb, qb = _curve(rate_b, quality_b)
    if len(ra) != len(rb):
        raise ValueError("unpaired QP grid")
    primary, sensitivity = bd_rate(ra, qa, rb, qb), pchip_bd(ra, qa, rb, qb)
    delta = bd_metric(ra, qa, rb, qb) * 100
    gaps = (qb - qa) * 100
    finite = all(math.isfinite(v) for v in (primary, sensitivity, delta))
    guards = {"canonical_bd_rate_lt_minus10": finite and primary < -10.,
              "pchip_bd_rate_lt_minus10": finite and sensitivity < -10.,
              "bd_quality_nonnegative": finite and delta >= -1e-8,
              "worst_qp_gap_ge_minus0_5pp": float(gaps.min()) >= -max_gap_pp,
              "same_qp_rate_nonincrease": bool(np.all(rb <= ra * (1 + 1e-12)))}
    point_passes = all(guards.values())
    statistical = (ci is not None and ci.get("hi") is not None and ci["hi"] < 0
                   and ci.get("finite_fraction", 0) >= .9)
    return {"bd_rate_pct": float(primary) if math.isfinite(primary) else None,
            "pchip_bd_rate_pct": float(sensitivity) if math.isfinite(sensitivity) else None,
            "bd_quality_pp": float(delta) if math.isfinite(delta) else None,
            "same_qp_gap_pp": gaps.tolist(), "same_qp_rate_pct": ((rb / ra - 1) * 100).tolist(),
            "quality_monotone_by_rate": {"anchor": bool(np.all(np.diff(qa[np.argsort(ra)]) >= 0)),
                                          "adaptive": bool(np.all(np.diff(qb[np.argsort(rb)]) >= 0))},
            "guards": guards, "point_passes": point_passes,
            "ci": ci, "screen_passes": bool(point_passes and statistical),
            "target_confirmed": False}


def paired_ar_bootstrap(rows: list[dict], model: str, qps: list[int], draws: int,
                        seed: int = 20261004) -> dict:
    ids = sorted({r["id"] for r in rows})
    lookup = {(r["id"], r["qp"], r["arm"]): r for r in rows}
    if len(lookup) != len(rows) or len(lookup) != len(ids) * len(qps) * 2:
        raise ValueError("missing or duplicate paired AR observations")
    arrays = {}
    for arm in ("anchor", "adaptive"):
        arrays[arm] = (np.array([[lookup[(i, q, arm)]["bpp"] for q in qps] for i in ids]),
                       np.array([[lookup[(i, q, arm)]["correct"][model] for q in qps] for i in ids]))
    rng, values = np.random.default_rng(seed), []
    for _ in range(draws):
        sample = rng.integers(0, len(ids), len(ids))
        ar, aq = (a[sample].mean(0) for a in arrays["anchor"])
        br, bq = (a[sample].mean(0) for a in arrays["adaptive"])
        value = bd_rate(ar, aq, br, bq)
        if math.isfinite(value):
            values.append(value)
    return {"lo": float(np.percentile(values, 2.5)) if values else None,
            "hi": float(np.percentile(values, 97.5)) if values else None,
            "draws": draws, "finite_draws": len(values),
            "finite_fraction": len(values) / draws if draws else 0.,
            "method": "paired_source_video_bootstrap"}
