"""Real synthetic streams test transport accounting and the shared encoder boundary."""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import importlib
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from adaptive_vcm.codec import StandardCodec, locate_ffmpeg


def modules():
    assert importlib.util.find_spec("adaptive_vcm.v31.codec") is not None, "V31 codec is missing"
    return importlib.import_module("adaptive_vcm.v31.codec"), importlib.import_module("adaptive_vcm.v31.transport")


def probe_binary():
    ffmpeg = Path(locate_ffmpeg())
    probe = shutil.which("ffprobe") or str(ffmpeg.with_name("ffprobe" + ffmpeg.suffix))
    assert Path(probe).is_file(), "Real ffprobe is required for timing/keyframe evidence"
    return probe


def moving_checker(frames=8):
    yy, xx = np.indices((32, 48))
    result = np.zeros((frames, 32, 48, 3), dtype=np.uint8)
    for t in range(frames):
        result[t, :, :, 0] = ((xx // 4 + yy // 4 + t) % 2) * 180
        result[t, :, :, 1] = 40
        result[t, 8:16, (t * 3) % 36:(t * 3) % 36 + 8, 2] = 240
    return result


def drop_recipe(mod, codec):
    return mod.Recipe("ar", codec, 48, 32, 8, 16, 32, 25, 2)


def test_packet_counts_overhead_and_hashes_stream_recipe_and_decoder_contract():
    mod, wire = modules()
    value = drop_recipe(wire, "h264")
    decoded = np.zeros((16, 32, 48, 3), dtype=np.uint8)
    packet = mod.Packet(b"abc", value, decoded, .25)
    assert packet.elementary_bytes == 3
    assert packet.total_bytes == 35
    assert len(packet.packet_hash) == 64
    decoder = b'{"err_detect":"explode","pixel_format":"rgb24","schema":"adaptive-vcm-actions-v10","threads":2,"vsync":"0","xerror":true}'
    header = bytes.fromhex("56433331 01 00 00 02 0030 0020 0008 0010 0000000000000020 00000019 00000000")
    assert packet.packet_hash == hashlib.sha256(hashlib.sha256(decoder).digest() + header + b"abc").hexdigest()
    assert packet.packet_hash == mod.Packet(b"abc", value, decoded, 9).packet_hash
    assert packet.packet_hash != replace(packet, encoded=b"abd").packet_hash
    assert packet.packet_hash != replace(packet, recipe=replace(value, duration_num=31)).packet_hash
    with pytest.raises(FrozenInstanceError):
        packet.seconds = 1


def test_packet_owns_decoded_samples_and_rejects_invalid_observations():
    mod, wire = modules()
    value = drop_recipe(wire, "h264")
    decoded = np.zeros((16, 32, 48, 3), dtype=np.uint8)
    packet = mod.Packet(b"abc", value, decoded, .25)
    decoded[:] = 99
    assert np.max(packet.decoded) == 0
    assert not packet.decoded.flags.writeable
    with pytest.raises(ValueError):
        packet.decoded[:] = 4
    for fields in ({"encoded": b""}, {"encoded": bytearray(b"abc")},
                   {"decoded": decoded[:8]}, {"seconds": -1}, {"seconds": float("nan")}):
        with pytest.raises(ValueError):
            replace(packet, **fields)


@pytest.mark.codec
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_absent_gop_retains_original_ffmpeg_commands(codec, monkeypatch):
    modules()
    run = subprocess.run
    commands = []
    def record(command, **kwargs):
        commands.append(command)
        return run(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", record)
    StandardCodec(codec, 35).roundtrip(moving_checker(4))
    encode, decode = commands
    assert encode[encode.index("-preset") + 1] == "medium"
    assert encode[encode.index("-threads") + 1] == "2"
    assert encode[encode.index("-r") + 1] == "25"
    assert "-g" not in encode and "-keyint_min" not in encode
    if codec == "h265":
        assert encode[encode.index("-x265-params") + 1] == "pools=2:frame-threads=1:log-level=error"
    else:
        assert "-x264-params" not in encode
    assert decode[decode.index("-threads") + 1] == "2"
    assert "-xerror" in decode
    assert decode[decode.index("-err_detect") + 1] == "explode"
    assert decode[decode.index("-vsync") + 1] == "0"


@pytest.mark.codec
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_real_drop2_fractional_fps_header_and_exact_repeat(codec, tmp_path):
    mod, wire = modules()
    assert locate_ffmpeg() is not None, "Real FFmpeg is required; a skip is not codec evidence"
    value = drop_recipe(wire, codec)
    packet = mod.V31Codec(codec, 35, "medium", Fraction(25, 4)).roundtrip(moving_checker(), value)
    assert packet.elementary_bytes > 0
    assert packet.total_bytes == packet.elementary_bytes + 32
    assert wire.unpack_recipe(wire.pack_recipe(packet.recipe)) == value
    assert packet.decoded.shape == (16, 32, 48, 3)
    assert np.array_equal(packet.decoded[::2], packet.decoded[1::2])
    assert np.std(packet.decoded[0]) > 10
    assert not np.array_equal(packet.decoded[0], packet.decoded[2])
    assert packet.seconds > 0
    path = tmp_path / ("clip.h264" if codec == "h264" else "clip.hevc")
    path.write_bytes(packet.encoded)
    probe = probe_binary()
    result = subprocess.run([probe, "-v", "error", "-show_entries", "stream=codec_name,width,height,time_base:frame=duration",
                             "-of", "json", str(path)], capture_output=True, check=True, text=True)
    evidence = json.loads(result.stdout)
    stream = evidence["streams"][0]
    assert stream["codec_name"] == ("h264" if codec == "h264" else "hevc")
    assert (stream["width"], stream["height"]) == (48, 32)
    # H.264 r_frame_rate can reflect field ticks rather than displayed frames.
    durations = [frame["duration"] * Fraction(stream["time_base"]) for frame in evidence["frames"]]
    assert durations == [Fraction(4, 25)] * 8
    assert sum(durations) == Fraction(32, 25)


@pytest.mark.codec
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_primary_adapter_preserves_legacy_stream_defaults(codec):
    mod, wire = modules()
    rgb = moving_checker(4)
    value = wire.Recipe("ar", codec, 48, 32, 4, 4, 4, 25, 1)
    legacy = StandardCodec(codec, 35).roundtrip(rgb)
    packet = mod.V31Codec(codec, 35, "medium", Fraction(25)).roundtrip(rgb, value)
    assert packet.encoded == legacy.data
    assert np.array_equal(packet.decoded, legacy.decoded)


@pytest.mark.parametrize("change", ["codec", "width", "height", "count", "duration", "dtype"])
def test_stream_and_recipe_mismatch_is_rejected(change):
    mod, wire = modules()
    value = drop_recipe(wire, "h264")
    rgb = moving_checker()
    if change == "codec":
        value = replace(value, codec="h265")
    elif change == "width":
        value = replace(value, width=46)
    elif change == "height":
        value = replace(value, height=30)
    elif change == "count":
        rgb = rgb[:-1]
    elif change == "duration":
        value = replace(value, duration_num=31)
    else:
        rgb = rgb.astype(np.float32)
    with pytest.raises(ValueError):
        mod.V31Codec("h264", 35, "medium", Fraction(25, 4)).roundtrip(rgb, value)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), 6.25, None, True])
def test_v31_requires_known_positive_fraction_fps(fps):
    mod, _ = modules()
    with pytest.raises(ValueError):
        mod.V31Codec("h264", 35, "medium", fps)


@pytest.mark.parametrize("settings", [{}, {"gop": 0}, {"gop": True}, {"gop": 8.0},
    {"gop": 8, "other": 0}, {"gop": 8, "scenecut": 1}, {"gop": 8, "bframes": -1},
    {"gop": 8, "bframes": True}, {"gop": 8, "bframes": 1}, {"gop": 8, "closed_gop": 1}])
def test_explicit_gop_settings_fail_closed_on_invalid_contract(settings):
    with pytest.raises(ValueError):
        StandardCodec("h264", 35, gop_settings=settings)


@pytest.mark.codec
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_positive_subunit_fps_and_explicit_gop_have_real_keyframes(codec, tmp_path):
    mod, wire = modules()
    value = wire.Recipe("ar", codec, 48, 32, 16, 16, 32, 1, 1)
    packet = mod.V31Codec(codec, 35, "medium", Fraction(1, 2),
                          {"gop": 8, "scenecut": False, "bframes": 0, "closed_gop": True}).roundtrip(moving_checker(16), value)
    path = tmp_path / "stream.bin"
    path.write_bytes(packet.encoded)
    result = subprocess.run([probe_binary(), "-v", "error",
                             "-show_entries", "frame=key_frame", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True)
    assert [i for i, frame in enumerate(json.loads(result.stdout)["frames"]) if frame["key_frame"]] == [0, 8]
    assert packet.total_bytes == len(packet.encoded) + 32


@pytest.mark.codec
@pytest.mark.parametrize("codec", ["h264", "h265"])
@pytest.mark.parametrize("flags,want", [({}, "keyint=8:min-keyint=8:scenecut=0:bframes=0:open-gop=0"),
    ({"scenecut": True, "closed_gop": False}, "keyint=8:min-keyint=8:scenecut=40:bframes=0:open-gop=1")])
def test_explicit_gop_maps_flags_and_snapshots_caller_settings(codec, flags, want, monkeypatch):
    run = subprocess.run
    commands = []
    def record(command, **kwargs):
        commands.append(command)
        return run(command, **kwargs)
    settings = {"gop": 8, **flags}
    encoder = StandardCodec(codec, 35, fps=Fraction(25, 4), gop_settings=settings)
    settings["gop"] = 99
    monkeypatch.setattr(subprocess, "run", record)
    result = encoder.roundtrip(moving_checker())
    assert result.decoded.shape == (8, 32, 48, 3)
    encode = commands[0]
    assert encode[encode.index("-g") + 1] == "8"
    assert encode[encode.index("-keyint_min") + 1] == "8"
    assert encode[encode.index("-r") + 1] == "25/4"
    parameter = "-x264-params" if codec == "h264" else "-x265-params"
    prefix = "" if codec == "h264" else "pools=2:frame-threads=1:log-level=error:"
    assert encode[encode.index(parameter) + 1] == prefix + want
