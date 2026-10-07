"""Literal wire fixtures catch altered layout, unchecked fields and sampling drift."""
from dataclasses import FrozenInstanceError
from fractions import Fraction
import importlib
import importlib.util
import struct

import numpy as np
import pytest


def transport():
    assert importlib.util.find_spec("adaptive_vcm.v31.transport") is not None, "V31 transport is missing"
    return importlib.import_module("adaptive_vcm.v31.transport")


def recipe(**changes):
    mod = transport()
    fields = dict(task="ar", codec="h264", width=48, height=32, coded_frames=8,
                  analyzer_frames=16, duration_num=32, duration_den=25, repeat_factor=2)
    fields.update(changes)
    return mod.Recipe(**fields)


def test_exact_32_byte_wire_and_reduced_duration():
    mod = transport()
    value = recipe()
    wire = bytes.fromhex("56433331 01 00 00 02 0030 0020 0008 0010 0000000000000020 00000019 00000000")
    assert mod.pack_recipe(value) == wire
    assert len(wire) == 32
    assert mod.unpack_recipe(wire) == value
    assert value.duration == Fraction(32, 25)
    assert value.fps == Fraction(25, 4)
    assert Fraction(value.coded_frames * value.duration_den, value.duration_num) == Fraction(25, 4)
    with pytest.raises(FrozenInstanceError):
        value.width = 99


def test_od_and_separate_64_frame_gop_recipe_are_supported():
    mod = transport()
    for value in (recipe(task="od", codec="h265", coded_frames=1, analyzer_frames=1,
                         repeat_factor=1, duration_num=1, duration_den=25),
                  recipe(coded_frames=64, analyzer_frames=64, repeat_factor=1)):
        assert mod.unpack_recipe(mod.pack_recipe(value)) == value


@pytest.mark.parametrize("field,value", [
    ("task", "unknown"), ("codec", "hevc"), ("width", 0), ("height", -1),
    ("width", 65536), ("height", 65536), ("coded_frames", 0),
    ("coded_frames", 65536), ("analyzer_frames", 65536), ("repeat_factor", 3),
    ("repeat_factor", True), ("coded_frames", 8.0), ("duration_num", 0),
    ("duration_num", None), ("duration_num", 2**64), ("duration_den", 0),
    ("duration_den", 2**32), ("duration_den", True), ("duration_den", 24),
    ("analyzer_frames", 15),
])
def test_invalid_recipe_fields_cannot_be_transmitted(field, value):
    mod = transport()
    with pytest.raises(ValueError):
        mod.pack_recipe(recipe(**{field: value}))


@pytest.mark.parametrize("offset,value", [(0, 0), (4, 2), (5, 2), (6, 2), (7, 0), (31, 1)])
def test_unknown_magic_version_enum_repeat_or_reserved_is_rejected(offset, value):
    mod = transport()
    wire = bytearray(mod.pack_recipe(recipe()))
    wire[offset] = value
    with pytest.raises(ValueError):
        mod.unpack_recipe(bytes(wire))


@pytest.mark.parametrize("size", [0, 1, 31, 33, 64])
def test_truncated_and_extended_headers_are_rejected(size):
    mod = transport()
    with pytest.raises(ValueError):
        mod.unpack_recipe(b"\0" * size)


def test_drop2_ramp_restores_exact_analyzer_positions():
    mod = transport()
    coded = np.broadcast_to(np.arange(0, 16, 2, dtype=np.uint8)[:, None, None, None], (8, 32, 48, 3)).copy()
    restored = mod.restore_samples(coded, recipe())
    assert restored.shape == (16, 32, 48, 3)
    assert restored[:, 0, 0, 0].tolist() == [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14]
    assert np.array_equal(coded[:, 0, 0, 0], [0, 2, 4, 6, 8, 10, 12, 14])


@pytest.mark.parametrize("shape,dtype", [((7, 32, 48, 3), np.uint8),
    ((8, 30, 48, 3), np.uint8), ((8, 32, 46, 3), np.uint8),
    ((8, 32, 48, 4), np.uint8), ((8, 32, 48, 3), np.float32)])
def test_decoder_geometry_count_channel_or_dtype_mismatch_fails(shape, dtype):
    mod = transport()
    with pytest.raises(ValueError):
        mod.restore_samples(np.zeros(shape, dtype=dtype), recipe())


def test_unpack_validates_duration_and_count_not_just_struct_shape():
    mod = transport()
    for num, den, frames in ((64, 50, 16), (0, 25, 16), (32, 0, 16), (32, 25, 15)):
        wire = struct.pack(">4sBBBBHHHHQII", b"VC31", 1, 0, 0, 2, 48, 32, 8, frames, num, den, 0)
        with pytest.raises(ValueError):
            mod.unpack_recipe(wire)
