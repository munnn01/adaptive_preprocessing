"""Fixed sampling metadata; no labels, regions or semantic features travel here."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import struct

import numpy as np

from .protocol import CODECS


_WIRE = struct.Struct(">4sBBBBHHHHQII")
RECIPE_BYTES = _WIRE.size
_TASKS = ("ar", "od")


@dataclass(frozen=True)
class Recipe:
    task: str
    codec: str
    width: int
    height: int
    coded_frames: int
    analyzer_frames: int
    duration_num: int
    duration_den: int
    repeat_factor: int

    def __post_init__(self) -> None:
        if self.task not in _TASKS or self.codec not in CODECS:
            raise ValueError("unknown transport task/codec")
        for field, bits in (("width", 16), ("height", 16), ("coded_frames", 16),
                            ("analyzer_frames", 16), ("duration_num", 64), ("duration_den", 32)):
            value = getattr(self, field)
            if type(value) is not int or not 0 < value < 2**bits:
                raise ValueError(f"invalid transport {field}")
        if type(self.repeat_factor) is not int or self.repeat_factor not in (1, 2):
            raise ValueError("transport repeat factor must be 1 or 2")
        if self.coded_frames * self.repeat_factor != self.analyzer_frames:
            raise ValueError("transport coded/analyzer count mismatch")
        if math.gcd(self.duration_num, self.duration_den) != 1:
            raise ValueError("transport duration must be a reduced positive rational")

    @property
    def duration(self) -> Fraction:
        return Fraction(self.duration_num, self.duration_den)

    @property
    def fps(self) -> Fraction:
        return self.coded_frames / self.duration


def pack_recipe(recipe: Recipe) -> bytes:
    if not isinstance(recipe, Recipe):
        raise ValueError("expected transport Recipe")
    return _WIRE.pack(b"VC31", 1, _TASKS.index(recipe.task), CODECS.index(recipe.codec),
                      recipe.repeat_factor, recipe.width, recipe.height, recipe.coded_frames,
                      recipe.analyzer_frames, recipe.duration_num, recipe.duration_den, 0)


def unpack_recipe(data: bytes) -> Recipe:
    if type(data) is not bytes or len(data) != RECIPE_BYTES:
        raise ValueError("transport header must be exactly 32 bytes")
    magic, version, task, codec, repeat, width, height, coded, analyzer, num, den, reserved = _WIRE.unpack(data)
    if magic != b"VC31" or version != 1 or reserved != 0:
        raise ValueError("unsupported transport magic/version/reserved field")
    if task >= len(_TASKS) or codec >= len(CODECS):
        raise ValueError("unknown transport task/codec code")
    return Recipe(_TASKS[task], CODECS[codec], width, height, coded, analyzer, num, den, repeat)


def restore_samples(decoded: np.ndarray, recipe: Recipe) -> np.ndarray:
    """Repeat decoded coded samples into analyzer positions, retaining coded geometry."""
    pack_recipe(recipe)
    shape = (recipe.coded_frames, recipe.height, recipe.width, 3)
    if not isinstance(decoded, np.ndarray) or decoded.dtype != np.uint8 or decoded.shape != shape:
        raise ValueError(f"decoded samples must be uint8 RGB {shape}")
    return np.repeat(decoded, recipe.repeat_factor, axis=0)
