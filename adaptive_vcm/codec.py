"""Strict real-codec round trip; count elementary stream bytes, including headers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import sys

import numpy as np

from .preprocessing import validate_clip


@dataclass(frozen=True)
class Encoded:
    decoded: np.ndarray
    data: bytes
    seconds: float

    @property
    def coded_bytes(self) -> int:
        return len(self.data)


def locate_ffmpeg() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    # Conda on Windows ships FFmpeg beside its own dependency DLLs.
    candidate = Path(sys.executable).parent / "Library/bin/ffmpeg.exe"
    return str(candidate) if candidate.is_file() else None


class StandardCodec:
    def __init__(self, codec: str, qp: int, preset: str = "medium", fps: int = 25,
                 threads: int = 2, timeout: float = 120):
        if codec not in ("h264", "h265") or isinstance(qp, bool) or not isinstance(qp, int) or not 0 <= qp <= 51:
            raise ValueError("invalid codec or QP")
        if fps < 1 or threads < 1 or timeout <= 0:
            raise ValueError("invalid codec runtime settings")
        self.codec, self.qp, self.preset = codec, qp, preset
        self.fps, self.threads, self.timeout = fps, threads, timeout
        self.ffmpeg = locate_ffmpeg()
        if self.ffmpeg is None:
            raise RuntimeError("FFmpeg with libx264/libx265 is required on PATH")

    def roundtrip(self, clip: np.ndarray) -> Encoded:
        validate_clip(clip)
        t, h, w, _ = clip.shape
        if h % 2 or w % 2:
            raise ValueError("yuv420p requires even spatial dimensions")
        encoder, muxer, extension = ("libx264", "h264", "264") if self.codec == "h264" else ("libx265", "hevc", "265")
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="adaptive-vcm-") as directory:
            path = Path(directory) / f"clip.{extension}"
            command = [self.ffmpeg, "-nostdin", "-y", "-v", "error", "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.fps),
                       "-i", "pipe:0", "-an", "-c:v", encoder, "-preset", self.preset,
                       "-qp", str(self.qp), "-threads", str(self.threads), "-pix_fmt", "yuv420p"]
            if self.codec == "h265":
                command += ["-x265-params", f"pools={self.threads}:frame-threads=1:log-level=error"]
            command += ["-f", muxer, str(path)]
            subprocess.run(command, input=clip.tobytes(), stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, check=True, timeout=self.timeout)
            stream = path.read_bytes()
            if not stream:
                raise RuntimeError("encoder produced an empty stream")
            decoded = subprocess.run([self.ffmpeg, "-nostdin", "-v", "error", "-xerror",
                                      "-err_detect", "explode", "-threads", str(self.threads),
                                      "-i", str(path), "-vsync", "0", "-f", "rawvideo",
                                      "-pix_fmt", "rgb24", "pipe:1"], stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     check=True, timeout=self.timeout).stdout
        expected = t * h * w * 3
        if len(decoded) != expected:
            raise RuntimeError(f"decode size mismatch: expected {expected} bytes, got {len(decoded)}")
        return Encoded(np.frombuffer(decoded, np.uint8).reshape(t, h, w, 3).copy(),
                       stream, time.perf_counter() - start)


def reference_bpp(coded_bytes: int, source_shape: tuple[int, ...]) -> float:
    """All resized candidates use the ORIGINAL T*H*W denominator."""
    if coded_bytes <= 0 or len(source_shape) != 4 or min(source_shape[:3]) <= 0 or source_shape[3] != 3:
        raise ValueError("invalid bit count/source shape")
    return float(8 * coded_bytes / np.prod(source_shape[:3]))
