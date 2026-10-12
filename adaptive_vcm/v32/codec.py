"""Standard codec plus a separately parsed, paid ROI reconstruction recipe."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np

from ..codec import StandardCodec
from .roi import inverse_retarget
from .transport import pack_packet, unpack_packet


@dataclass(frozen=True)
class Packet:
    data: bytes
    decoded: np.ndarray
    coded_shape: tuple
    elementary_bytes: int
    metadata_bytes: int
    seconds: float

    @property
    def coded_bytes(self):
        return len(self.data)

    @property
    def packet_sha256(self):
        return hashlib.sha256(self.data).hexdigest()


class V32Codec:
    def __init__(self, codec: str, qp: int, preset: str = 'medium', fps: Fraction = Fraction(25), *, gop_settings=None):
        if not isinstance(fps, Fraction) or fps <= 0:
            raise ValueError('V32 requires rational FPS')
        self.codec, self.qp, self.fps = codec, qp, fps
        self.standard = StandardCodec(codec, qp, preset=preset, fps=fps, gop_settings=gop_settings)

    def roundtrip(self, rgb, *, source_shape, task, grids=(), encoded=None):
        start = time.perf_counter()
        # Validate wire recipe before touching the encoder.
        pack_packet(b'validate', task=task, codec=self.codec, source_shape=source_shape,
                    coded_shape=rgb.shape, fps=self.fps, grids=grids)
        result = self.standard.roundtrip(rgb) if encoded is None else encoded
        wire = pack_packet(result.data, task=task, codec=self.codec, source_shape=source_shape,
                           coded_shape=rgb.shape, fps=self.fps, grids=grids)
        received = self.decode_packet(wire)
        return Packet(wire, received.decoded, rgb.shape, len(result.data),
                      received.metadata_bytes, time.perf_counter() - start)

    def decode_packet(self, wire: bytes):
        start = time.perf_counter()
        packet = unpack_packet(wire)
        if packet.codec != self.codec or packet.fps != self.fps:
            raise ValueError('V32 receiver codec/FPS mismatch')
        fmt = 'h264' if packet.codec == 'h264' else 'hevc'
        t, h, w, _ = packet.coded_shape
        expected = t * h * w * 3
        # libavcodec's max_pixels also covers aligned/padded frame buffers.
        allocation_pixels = 4 * ((w + 63) // 64 * 64) * ((h + 63) // 64 * 64)
        # Bound allocations before decode. One extra frame detects truncation;
        # decoded RGB goes to a bounded file, never an unbounded stdout buffer.
        with tempfile.TemporaryDirectory(prefix='v32-receiver-') as directory:
            path = Path(directory) / 'received.rgb'
            command = [self.standard.ffmpeg, '-nostdin', '-nostats', '-y', '-v', 'info', '-xerror',
                '-max_alloc', str(max(16 * 1024 * 1024, h * w * 16)),
                '-err_detect', 'explode', '-threads', '2', '-max_pixels', str(allocation_pixels),
                '-f', fmt, '-i', 'pipe:0', '-vsync', '0', '-vf', 'showinfo',
                '-frames:v', str(t + 1), '-fs', str(expected + h * w * 3),
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', str(path)]
            result = subprocess.run(command, input=packet.stream, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, check=True, timeout=self.standard.timeout)
            # showinfo runs before output autoscaling, including resolution changes.
            actual = re.findall(rb'\[Parsed_showinfo_\d+ @ [^\]]+\]\s+n:\s*\d+[^\r\n]*?\s+s:(\d+)x(\d+)', result.stderr)
            if len(actual) != t or any((int(aw), int(ah)) != (w, h) for aw, ah in actual):
                raise ValueError('VC32 actual stream geometry/frames disagree with transmitted fields')
            if path.stat().st_size != expected:
                raise ValueError('VC32 decoded stream disagrees with transmitted geometry')
            output = path.read_bytes()
        rgb = np.frombuffer(output, np.uint8).reshape(packet.coded_shape).copy()
        if packet.grids:
            rgb = inverse_retarget(rgb, packet.grids, packet.source_shape)
        rgb.setflags(write=False)
        return Packet(wire, rgb, packet.coded_shape, len(packet.stream), packet.metadata_bytes,
                      time.perf_counter() - start)
