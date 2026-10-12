"""Strict real-codec round trip; count elementary stream bytes, including headers."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import sys
import math

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
    def __init__(self, codec: str, qp: int, preset: str = "medium", fps: int | Fraction = 25,
                 threads: int = 2, timeout: float = 120, gop_settings: dict | None = None):
        if codec not in ("h264", "h265") or isinstance(qp, bool) or not isinstance(qp, int) or not 0 <= qp <= 51:
            raise ValueError("invalid codec or QP")
        valid_fps = fps > 0 if isinstance(fps, Fraction) else math.isfinite(fps) and fps >= 1
        if not valid_fps or threads < 1 or timeout <= 0:
            raise ValueError("invalid codec runtime settings")
        if gop_settings is not None:
            if not isinstance(gop_settings, dict) or set(gop_settings) - {"gop", "scenecut", "bframes", "closed_gop"}:
                raise ValueError("unknown GOP settings")
            gop = gop_settings.get("gop")
            if type(gop) is not int or gop < 1:
                raise ValueError("GOP requires a positive integer gop")
            settings = {"gop": gop, "scenecut": False, "bframes": 0, "closed_gop": True,
                        **gop_settings}
            if type(settings["scenecut"]) is not bool or type(settings["closed_gop"]) is not bool:
                raise ValueError("scenecut and closed_gop must be booleans")
            if type(settings["bframes"]) is not int or settings["bframes"] != 0:
                raise ValueError("explicit GOP audit requires bframes=0")
            gop_settings = settings
        self.codec, self.qp, self.preset = codec, qp, preset
        self.fps, self.threads, self.timeout = fps, threads, timeout
        self.gop_settings = gop_settings
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
                params = f"pools={self.threads}:frame-threads=1:log-level=error"
            else:
                params = ""
            if self.gop_settings is not None:
                gop = self.gop_settings["gop"]
                command += ["-g", str(gop), "-keyint_min", str(gop)]
                options = (f"keyint={gop}:min-keyint={gop}:"
                           f"scenecut={40 if self.gop_settings['scenecut'] else 0}:"
                           f"bframes={self.gop_settings['bframes']}:"
                           f"open-gop={int(not self.gop_settings['closed_gop'])}")
                params = f"{params}:{options}" if params else options
            if params:
                command += ["-x265-params" if self.codec == "h265" else "-x264-params", params]
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
