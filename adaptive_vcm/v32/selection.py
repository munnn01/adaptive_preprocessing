"""Measured fixed/hybrid ROI arms; guards see receiver reconstruction only."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np

from ..preprocessing import make_candidates
from ..selection import Observation, relative_guard, select
from .roi import build_grids, retarget, simplify, validated_support
from .transport import pack_packet


@dataclass(frozen=True)
class Candidate:
    name: str
    rgb: np.ndarray
    family: str
    grids: tuple = ()


@dataclass
class Bundle:
    anchor: object
    arms: dict
    audit: list
    seconds: float


def build_candidates(rgb, protection, support, task, codec, cfg, baseline=None, *, enable_retarget=True):
    validated_support(support, rgb.shape)
    control_task = 'od' if task == 'ot' else task
    candidates = [Candidate(c.name, c.clip, 'controls') for c in
                  make_candidates(rgb, protection, control_task, codec.qp, cfg[f'{control_task}_candidates'])]
    learned = [] if baseline is None else baseline.proposals(rgb, support, codec)
    candidates.extend(Candidate(c.name, c.clip, 'learned') for c in learned)
    margin = cfg['roi']['margin']
    simplified = {}
    for strength in cfg['roi']['strengths']:
        pixels = simplify(rgb, support, strength, margin)
        simplified[strength] = pixels
        if not np.array_equal(pixels, rgb):
            candidates.append(Candidate(f'roi_simplify_s{int(strength * 100):03}', pixels, 'simplification'))
    for factor in cfg['roi']['factors'] if enable_retarget else ():
        grids = build_grids(rgb.shape, support, factor, margin)
        if grids is None:
            continue
        candidates.append(Candidate(f'roi_retarget_f{factor}', retarget(simplified[1.], grids), 'retarget', grids))
        for c in learned:
            candidates.append(Candidate(f'hybrid_{c.name}_f{factor}', retarget(c.clip, grids), 'hybrid', grids))
    return candidates


def select_arms(rows, slack, min_savings, has_baseline, *, extra_families=None):
    families = dict(controls={'controls'}, simplification={'controls', 'simplification'},
                    retarget={'controls', 'retarget'},
                    v32={'controls', 'simplification', 'retarget', 'learned', 'hybrid'})
    if has_baseline:
        families.update(v30b={'controls', 'learned'}, hybrid={'controls', 'learned', 'hybrid'})
    for name, allowed in (extra_families or {}).items():
        if name in families or not isinstance(allowed, set) or 'controls' not in allowed:
            raise ValueError('extra arms must be explicit and retain controls fallback')
        families[name] = allowed
    result = {}
    for name, allowed in families.items():
        indices = [i for i, row in enumerate(rows) if row['family'] in allowed]
        result[name] = indices[select([rows[i]['observation'] for i in indices], slack, min_savings)]
    return result


def choose_stream(rgb, protection, support, task, codec, cfg, teachers, source_predictions, baseline=None, *, guard=None,
                  extra_candidates=(), extra_families=None, forced_names=None, unconditional_extra=False, enable_retarget=True):
    start = time.perf_counter()
    if task not in ('ar', 'od', 'ot') or len(teachers) != (2 if task == 'ar' else 1):
        raise ValueError('V32 task/teacher count mismatch')
    if task == 'ot' and (baseline is not None or guard is None):
        raise ValueError('OT requires an explicit tracking guard and fixed-only scope')
    known = task == 'ar' or np.any(source_predictions[0]['scores'] >= cfg['od_score_threshold'])
    routing = {} if enable_retarget else {'enable_retarget': False}
    candidates = (build_candidates(rgb, protection, support, task, codec, cfg, baseline, **routing) if known
                  else [Candidate('identity', rgb.copy(), 'controls')])
    if known or unconditional_extra:
        if any(c.family in {'controls', 'simplification', 'retarget', 'learned', 'hybrid'} for c in extra_candidates):
            raise ValueError('extension candidates cannot enter historical V32 families')
        candidates.extend(extra_candidates)
    slack = cfg['ar_kl_slack' if task == 'ar' else 'od_distance_slack']
    pixel_cache, packet_cache, teacher_cache = {}, {}, {}
    packets, rows, audit = [], [], []
    anchor_predictions = None
    for candidate in candidates:
        pixel_hash = hashlib.sha256(candidate.rgb.tobytes()).hexdigest()
        key = (candidate.rgb.shape, pixel_hash)
        new_encode = key not in pixel_cache
        if new_encode:
            pixel_cache[key] = codec.standard.roundtrip(candidate.rgb)
        encoded = pixel_cache[key]
        wire = pack_packet(encoded.data, task=task, codec=codec.codec, source_shape=rgb.shape,
            coded_shape=candidate.rgb.shape, fps=codec.fps, grids=candidate.grids)
        wire_hash = hashlib.sha256(wire).hexdigest()
        new_packet = wire_hash not in packet_cache
        if new_packet:
            packet_cache[wire_hash] = codec.decode_packet(wire)
        packet = packet_cache[wire_hash]
        decoded_key = (packet.decoded.shape, hashlib.sha256(packet.decoded.tobytes()).hexdigest())
        new_prediction = decoded_key not in teacher_cache
        if new_prediction:
            teacher_cache[decoded_key] = [t.probabilities(packet.decoded) if task == 'ar' else t.predict(packet.decoded) for t in teachers]
        predictions = teacher_cache[decoded_key]
        if anchor_predictions is None:
            anchor_predictions = predictions
            distances, decisions = (0.,) * len(teachers), (True,) * len(teachers)
        elif decoded_key == (packets[0].decoded.shape, hashlib.sha256(packets[0].decoded.tobytes()).hexdigest()):
            distances, decisions = (0.,) * len(teachers), (True,) * len(teachers)
        else:
            distances, decisions = (guard or relative_guard)(task, source_predictions, anchor_predictions, predictions, cfg)
        observation = Observation(candidate.name, packet.coded_bytes, distances, decisions)
        rows.append(dict(family=candidate.family, observation=observation))
        packets.append(packet)
        audit.append(dict(name=candidate.name, family=candidate.family, coded_shape=list(candidate.rgb.shape),
            reconstructed_shape=list(packet.decoded.shape), elementary_bytes=packet.elementary_bytes,
            metadata_bytes=packet.metadata_bytes, total_bytes=packet.coded_bytes, packet_sha256=wire_hash,
            source_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), coded_pixel_sha256=pixel_hash,
            reconstructed_pixel_sha256=decoded_key[1], distinct_encode=new_encode,
            distinct_packet_decode=new_packet, teacher_calls=len(teachers) if new_prediction else 0,
            relative_distances=[float(d) if np.isfinite(d) else None for d in distances],
            preserves_decision=list(decisions), encode_seconds=encoded.seconds if new_encode else 0.,
            receiver_seconds=packet.seconds if new_packet else 0.))
    indices = select_arms(rows, slack, cfg['min_savings'], baseline is not None, extra_families=extra_families)
    for arm,name in (forced_names or {}).items():
        matches=[i for i,c in enumerate(candidates) if c.name==name]
        if len(matches)!=1 or arm not in (extra_families or {}):
            raise ValueError('forced arm requires one explicit candidate and declared family')
        indices[arm]=matches[0]
    return Bundle(packets[0], {arm: (packets[index], candidates[index].name) for arm, index in indices.items()},
                  audit, time.perf_counter() - start)
