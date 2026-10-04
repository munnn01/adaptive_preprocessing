import shutil
from pathlib import Path

import numpy as np
import pytest

from adaptive_vcm.codec import StandardCodec, reference_bpp, locate_ffmpeg
from adaptive_vcm.data import fingerprint, partition, read_video, read_image
from adaptive_vcm.coco_metrics import _remap_bootstrap_sample, coco_map


@pytest.mark.codec
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_real_roundtrip_strict_frame_count_and_original_pixel_denominator(codec):
    if locate_ffmpeg() is None:
        pytest.skip("FFmpeg unavailable")
    clip = np.random.default_rng(9).integers(0, 256, (4, 32, 48, 3), dtype=np.uint8)
    output = StandardCodec(codec, 35).roundtrip(clip)
    assert output.decoded.shape == clip.shape
    assert output.coded_bytes == len(output.data) > 0
    assert reference_bpp(output.coded_bytes, (4, 64, 96, 3)) == pytest.approx(8 * len(output.data) / (4 * 64 * 96))
    with pytest.raises(ValueError):
        StandardCodec(codec, 35).roundtrip(clip[:, :, :47])


def test_partition_and_fingerprint_are_stable_and_duplicate_sources_rejected():
    assert partition("class/clip.mp4") == partition("class/clip.mp4")
    assert fingerprint(["b", "a"]) == fingerprint(["a", "b"])
    with pytest.raises(ValueError):
        fingerprint(["a", "a"])


def test_bad_source_never_turns_into_black_video(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError):
        read_video(str(path), 16, 128)


def test_image_letterbox_geometry_matches_rounded_dimensions(tmp_path):
    from PIL import Image
    path = tmp_path / "source.jpg"
    Image.new("RGB", (61, 29)).save(path)
    clip, (sx, sy, left, top) = read_image({"path": str(path), "width": 61, "height": 29}, 64)
    assert clip.shape == (1, 64, 64, 3)
    assert sx == 64 / 61 and sy == 30 / 29 and left == 0 and top == 17


def test_coco_empty_detections_and_bootstrap_multiplicity():
    gt = {1: [{"id": 8, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}]}
    copied, ids, mapping = _remap_bootstrap_sample([1, 1], gt)
    assert ids == [1, 2] and mapping == [(1, 1), (1, 2)]
    assert copied[1][0]["id"] != copied[2][0]["id"]
    meta = {"categories": [{"id": 1, "name": "object"}]}
    assert coco_map([], gt, [1], meta) == (0, 0)
    perfect = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": .9}]
    assert coco_map(perfect, gt, [1], meta)[0] == pytest.approx(1)
