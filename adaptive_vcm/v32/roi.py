"""Fixed rectangular ROI transforms. No labels or learned parameters."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from ..preprocessing import validate_clip


@dataclass(frozen=True)
class Grid:
    start: int
    stop: int
    x_source: tuple[int, ...]
    x_coded: tuple[int, ...]
    y_source: tuple[int, ...]
    y_coded: tuple[int, ...]

    def __post_init__(self):
        if type(self.start) is not int or type(self.stop) is not int or not 0 <= self.start < self.stop <= 65535:
            raise ValueError('invalid grid segment')
        for source, coded in ((self.x_source, self.x_coded), (self.y_source, self.y_coded)):
            if type(source) is not tuple or type(coded) is not tuple or len(source) != len(coded):
                raise ValueError('grid axes must be matching tuples')
            for edges in (source, coded):
                if (not 2 <= len(edges) <= 4096 or edges[0] != 0
                        or any(type(v) is not int or not 0 <= v <= 65535 for v in edges)
                        or any(a >= b for a, b in zip(edges, edges[1:]))):
                    raise ValueError('grid edges must be bounded and strictly increasing')


def validated_support(support: dict, shape: tuple):
    if (len(shape) != 4 or shape[-1] != 3 or any(type(v) is not int or v < 1 for v in shape)
            or any(v > 65535 for v in shape[:3])):
        raise ValueError('invalid canonical source shape')
    if not isinstance(support, dict) or not {'protection', 'motion', 'cuts'} <= support.keys():
        raise ValueError('missing ROI support')
    p, m = (np.asarray(support[k], np.float32) for k in ('protection', 'motion'))
    for value in (p, m):
        if value.shape != shape[:3] or not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError('ROI support must be finite THW in [0,1]')
    cuts = np.asarray(support['cuts'])
    if cuts.shape != (shape[0],) or cuts.dtype != bool or not cuts[0]:
        raise ValueError('ROI cuts require a first-frame segment start')
    return p, m, cuts


def roi_boxes(support: dict, shape: tuple, margin: float = .05):
    p, m, cuts = validated_support(support, shape)
    if type(margin) not in (int, float) or not math.isfinite(margin) or not 0 <= margin <= .5:
        raise ValueError('invalid ROI context margin')
    _, h, w, _ = shape
    halo = math.ceil(min(h, w) * margin)
    starts = np.flatnonzero(cuts).tolist() + [shape[0]]
    result = []
    for start, stop in zip(starts, starts[1:]):
        core = np.any((p[start:stop] >= .5) | (m[start:stop] >= .5), axis=0).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        boxes = []
        for x, y, bw, bh, _ in stats[1:count]:
            boxes.append((max(0, int(x) - halo), max(0, int(y) - halo),
                          min(w, int(x + bw) + halo), min(h, int(y + bh) + halo)))
        result.append((start, stop, tuple(boxes)))
    return result


def _axis(length, boxes, coordinate, factor):
    intervals = [(box[coordinate], box[coordinate + 2]) for box in boxes]
    edges = tuple(sorted({0, length, *(v for pair in intervals for v in pair)}))
    dst = [0]
    for left, right in zip(edges, edges[1:]):
        protected = any(a < right and b > left for a, b in intervals)
        size = right - left if protected else max(1, math.ceil((right - left) / factor))
        dst.append(dst[-1] + size)
    return edges, tuple(dst)


def build_grids(shape: tuple, support: dict, factor: int = 2, margin: float = .05):
    if type(factor) is not int or factor not in (2, 4):
        raise ValueError('background factor must be 2 or 4')
    regions = roi_boxes(support, shape, margin)
    if not any(boxes for _, _, boxes in regions):
        return None
    _, h, w, _ = shape
    raw = []
    for start, stop, boxes in regions:
        # An uncertain segment keeps identity geometry; do not shrink an empty scene.
        if not boxes:
            xs, xd, ys, yd = (0, w), (0, w), (0, h), (0, h)
        else:
            xs, xd = _axis(w, boxes, 0, factor)
            ys, yd = _axis(h, boxes, 1, factor)
        raw.append(Grid(start, stop, xs, xd, ys, yd))
    cw = min(w, 2 * math.ceil(max(g.x_coded[-1] for g in raw) / 2))
    ch = min(h, 2 * math.ceil(max(g.y_coded[-1] for g in raw) / 2))
    if (cw, ch) == (w, h):
        return None
    def fit(source, axis, total):
        # Pay common resolution/even rounding by expanding only background.
        # Scaling the entire grid would resample ROI cores at scene boundaries.
        spare = np.diff(source) - np.diff(axis)
        extra = total - axis[-1]
        available = int(spare.sum())
        if not extra:
            return axis
        cumulative = np.concatenate(([0], np.cumsum(spare)))
        return tuple(value + int(round(extra * int(room) / available))
                     for value, room in zip(axis, cumulative))
    return tuple(Grid(g.start, g.stop, g.x_source, fit(g.x_source, g.x_coded, cw),
                      g.y_source, fit(g.y_source, g.y_coded, ch)) for g in raw)


def validate_grids(grids, source_shape, coded_shape):
    if not isinstance(grids, tuple) or not grids:
        raise ValueError('nonempty immutable retarget grid required')
    t, h, w, c = source_shape
    ct, ch, cw, cc = coded_shape
    if c != 3 or cc != 3 or t != ct or min(t, h, w, ch, cw) < 1:
        raise ValueError('grid source/coded dimensions mismatch')
    cursor = 0
    for g in grids:
        if not isinstance(g, Grid) or g.start != cursor:
            raise ValueError('grid segment coverage gap/overlap')
        if (g.x_source[-1], g.y_source[-1], g.x_coded[-1], g.y_coded[-1]) != (w, h, cw, ch):
            raise ValueError('grid endpoint geometry mismatch')
        cursor = g.stop
    if cursor != t:
        raise ValueError('grid must cover all source frames exactly')


def _transform(rgb, grids, source_shape, inverse):
    validate_clip(rgb)
    g = grids[0]
    coded_shape = (source_shape[0], g.y_coded[-1], g.x_coded[-1], 3)
    validate_grids(grids, source_shape, coded_shape)
    expected = coded_shape if inverse else source_shape
    if rgb.shape != expected:
        raise ValueError('retarget input geometry mismatch')
    result = np.empty(source_shape if inverse else coded_shape, np.uint8)
    for g in grids:
        sx, dx = (g.x_coded, g.x_source) if inverse else (g.x_source, g.x_coded)
        sy, dy = (g.y_coded, g.y_source) if inverse else (g.y_source, g.y_coded)
        for frame in range(g.start, g.stop):
            for row in range(len(sy) - 1):
                for col in range(len(sx) - 1):
                    patch = rgb[frame, sy[row]:sy[row + 1], sx[col]:sx[col + 1]]
                    target = (dx[col + 1] - dx[col], dy[row + 1] - dy[row])
                    interpolation = cv2.INTER_AREA if target[0] <= patch.shape[1] and target[1] <= patch.shape[0] else cv2.INTER_LINEAR
                    result[frame, dy[row]:dy[row + 1], dx[col]:dx[col + 1]] = cv2.resize(patch, target, interpolation=interpolation)
    return result


def retarget(rgb: np.ndarray, grids: tuple[Grid, ...]):
    if not grids:
        raise ValueError('retarget requires grids')
    return _transform(rgb, grids, rgb.shape, inverse=False)


def inverse_retarget(rgb: np.ndarray, grids: tuple[Grid, ...], source_shape: tuple):
    if not grids:
        raise ValueError('inverse retarget requires grids')
    return _transform(rgb, grids, source_shape, inverse=True)


def simplify(rgb: np.ndarray, support: dict, strength: float = 1., margin: float = .05):
    validate_clip(rgb)
    p, m, _ = validated_support(support, rgb.shape)
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('simplification strength must be in [0,1]')
    regions = roi_boxes(support, rgb.shape, margin)
    if strength == 0 or not any(boxes for _, _, boxes in regions):
        return rgb.copy()
    _, h, w, _ = rgb.shape
    closing_size = max(3, int(round(.05 * min(h, w))) | 1)
    n = np.arange(-12, 13, dtype=np.float32)
    kernel = np.sinc(.2 * n) * np.sinc(n / 13)
    kernel /= kernel.sum()
    result = rgb.copy()
    for start, stop, boxes in regions:
        if not boxes:
            continue
        core = np.zeros((h, w), np.uint8)
        for x1, y1, x2, y2 in boxes:
            core[y1:y2, x1:x2] = 1
        closed = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((closing_size, closing_size), np.uint8))
        envelope = (closed | core).astype(bool)
        for frame in range(start, stop):
            source = rgb[frame].astype(np.float32)
            filtered = cv2.sepFilter2D(source, -1, kernel, kernel, borderType=cv2.BORDER_REFLECT_101)
            expert = np.where(envelope[..., None], filtered, 128.)
            alpha = strength * (1 - p[frame]) * (1 - .75 * m[frame])
            rendered = np.clip(np.rint(source + alpha[..., None] * (expert - source)), 0, 255).astype(np.uint8)
            rendered[p[frame] == 1] = rgb[frame][p[frame] == 1]
            result[frame] = rendered
    return result
