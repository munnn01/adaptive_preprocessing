"""Paid VC32 wire recipe. Decoder derives every ROI grid from received bytes."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import struct
import zlib

from .roi import Grid, validate_grids


_HEADER = struct.Struct('>4sBBBBHHHHHHIIIII')
HEADER_BYTES = _HEADER.size  # 40: common header is paid by every arm, including identity.
_SEGMENT = struct.Struct('>HHHH')
MAX_PACKET_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 16 * 4096 * 4096
TASKS = ('ar', 'od', 'ot')
CODECS = ('h264', 'h265')


@dataclass(frozen=True)
class WirePacket:
    task: str
    codec: str
    source_shape: tuple
    coded_shape: tuple
    fps: Fraction
    grids: tuple[Grid, ...]
    stream: bytes
    metadata_bytes: int


def _validate(task, codec, source_shape, coded_shape, fps, grids):
    if task not in TASKS or codec not in CODECS:
        raise ValueError('unknown VC32 task/codec')
    for shape in (source_shape, coded_shape):
        if (type(shape) is not tuple or len(shape) != 4 or shape[3] != 3
                or any(type(v) is not int or not 0 < v <= 65535 for v in shape[:3])
                or shape[1] % 2 or shape[2] % 2 or shape[0] * shape[1] * shape[2] > MAX_PIXELS):
            raise ValueError('invalid or oversized VC32 dimensions')
    if source_shape[0] != coded_shape[0] or task == 'od' and source_shape[0] != 1:
        raise ValueError('VC32 temporal geometry mismatch')
    if not isinstance(fps, Fraction) or fps <= 0 or max(fps.numerator, fps.denominator) >= 2**32:
        raise ValueError('VC32 FPS must be a bounded positive rational')
    if type(grids) is not tuple or len(grids) > source_shape[0]:
        raise ValueError('invalid VC32 grid container')
    if grids:
        validate_grids(grids, source_shape, coded_shape)
    elif task in ('od', 'ot') and source_shape != coded_shape:
        raise ValueError('plain OD/OT must retain canonical geometry')


def pack_packet(stream: bytes, *, task: str, codec: str, source_shape: tuple,
                coded_shape: tuple, fps: Fraction, grids: tuple = ()) -> bytes:
    _validate(task, codec, source_shape, coded_shape, fps, grids)
    if type(stream) is not bytes or not stream:
        raise ValueError('VC32 elementary stream must be nonempty bytes')
    pieces = []
    for grid in grids:
        pieces.append(_SEGMENT.pack(grid.start, grid.stop, len(grid.x_source), len(grid.y_source)))
        for edges in (grid.x_source, grid.x_coded, grid.y_source, grid.y_coded):
            pieces.append(struct.pack('>' + 'H' * len(edges), *edges))
    metadata = b''.join(pieces)
    t, h, w, _ = source_shape
    _, ch, cw, _ = coded_shape
    prefix = _HEADER.pack(b'VC32', 1, TASKS.index(task), CODECS.index(codec), bool(grids),
        t, h, w, ch, cw, len(grids), fps.numerator, fps.denominator,
        len(metadata), len(stream), 0)
    wire = prefix[:-4] + struct.pack('>I', zlib.crc32(prefix[:-4] + metadata + stream)) + metadata + stream
    if len(wire) > MAX_PACKET_BYTES:
        raise ValueError('VC32 packet too large')
    return wire


def unpack_packet(wire: bytes) -> WirePacket:
    if type(wire) is not bytes or not HEADER_BYTES < len(wire) <= MAX_PACKET_BYTES:
        raise ValueError('invalid VC32 packet size')
    fields = _HEADER.unpack_from(wire)
    magic, version, task, codec, mode, t, h, w, ch, cw, count, num, den, meta_size, stream_size, crc = fields
    if magic != b'VC32' or version != 1 or task >= len(TASKS) or codec >= len(CODECS) or mode not in (0, 1):
        raise ValueError('unsupported VC32 packet header')
    if len(wire) != HEADER_BYTES + meta_size + stream_size or stream_size < 1 or den == 0:
        raise ValueError('VC32 packet length/FPS mismatch')
    if zlib.crc32(wire[:HEADER_BYTES - 4] + wire[HEADER_BYTES:]) != crc:
        raise ValueError('VC32 packet checksum mismatch')
    if bool(count) != bool(mode) or count > t or (not mode and meta_size != 0):
        raise ValueError('VC32 mode/segment mismatch')
    metadata = memoryview(wire)[HEADER_BYTES:HEADER_BYTES + meta_size]
    cursor, grids = 0, []
    for _ in range(count):
        if cursor + _SEGMENT.size > len(metadata):
            raise ValueError('truncated VC32 grid')
        start, stop, nx, ny = _SEGMENT.unpack_from(metadata, cursor)
        cursor += _SEGMENT.size
        if not 2 <= nx <= 4096 or not 2 <= ny <= 4096:
            raise ValueError('oversized VC32 grid axes')
        edges = []
        for n in (nx, nx, ny, ny):
            size = 2 * n
            if cursor + size > len(metadata):
                raise ValueError('truncated VC32 axis')
            edges.append(struct.unpack_from('>' + 'H' * n, metadata, cursor))
            cursor += size
        grids.append(Grid(start, stop, *edges))
    if cursor != len(metadata):
        raise ValueError('trailing VC32 grid bytes')
    source_shape, coded_shape = (t, h, w, 3), (t, ch, cw, 3)
    fps = Fraction(num, den)
    _validate(TASKS[task], CODECS[codec], source_shape, coded_shape, fps, tuple(grids))
    return WirePacket(TASKS[task], CODECS[codec], source_shape, coded_shape, fps, tuple(grids),
                      wire[HEADER_BYTES + meta_size:], HEADER_BYTES + meta_size)
