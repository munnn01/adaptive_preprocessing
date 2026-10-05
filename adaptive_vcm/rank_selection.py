"""Equal-budget encoder-only ranking evaluation with reusable actual streams."""
from __future__ import annotations

import hashlib
import numpy as np

from .preprocessing import Candidate, make_candidates
from .codec import reference_bpp
from .ranking import build_rank_context
from .selection import Observation, relative_guard, select
from .task_bank import ACTION_NAMES, build_task_bank


def choose_rank_stream(clip, protection, codec, cfg, teachers, source_predictions,
                       learned, *, learned_mask=None, components=False):
    """Select controls + learned top K; full-bank measurements are audit only.

    Teacher predictions and actual byte counts from unproposed actions must
    never enter learned context, ranking or its primary candidate subset.
    """
    mask = protection if learned_mask is None else learned_mask
    controls = make_candidates(clip, protection, 'ar', codec.qp, cfg['ar_candidates'])
    portfolio = getattr(learned, 'schema', None) == 'adaptive-vcm-portfolio-v6'
    utility = portfolio or getattr(learned, 'schema', None) == 'adaptive-vcm-utility-v5'
    anchor = codec.roundtrip(controls[0].clip)
    anchor_predictions = [teacher.probabilities(anchor.decoded) for teacher in teachers]
    if utility:
        from .stabilized_bank import ACTION_NAMES as STABILIZED_NAMES, build_stabilized_bank
        names = learned.action_names
        if tuple(names) == ACTION_NAMES:
            bank = build_task_bank(clip, mask, codec.qp)
        elif tuple(names) == STABILIZED_NAMES:
            bank = build_stabilized_bank(clip, mask, codec.qp)
        elif portfolio:
            from .anchor_bank import ACTION_NAMES as ANCHOR_NAMES, build_anchor_bank
            if tuple(names) != ANCHOR_NAMES:
                raise ValueError('unregistered portfolio action bank')
            bank = build_anchor_bank(clip, mask, codec.qp, anchor.decoded)
        else:
            raise ValueError('unregistered utility action bank')
    else:
        bank = build_task_bank(clip, mask, codec.qp)
        names = ACTION_NAMES
    if tuple(c.name for c in bank) != tuple(names):
        raise ValueError('task bank/action order mismatch')
    if utility:
        from .utility_ranking import build_utility_context
        context = build_utility_context(clip, codec.qp, codec.codec, mask,
                                        source_predictions, anchor_predictions,
                                        anchor_bpp=reference_bpp(anchor.coded_bytes, clip.shape))
    else:
        context = build_rank_context(clip, codec.qp, codec.codec, mask,
                                     source_predictions, anchor_predictions)
    top_k = cfg['rank_top_k']
    learned_indices = [int(i) for i in learned.rank(context, top_k=top_k)]
    details = learned.proposal_details(context, top_k) if utility else None
    static_indices = [int(i) for i in learned.static_action_order[:top_k]]
    group_indices = ([int(i) for i in learned.group_static_action_order(context, top_k)]
                     if utility else [])
    if len(learned_indices) != top_k or len(set(learned_indices)) != top_k:
        raise ValueError('ranking must propose K distinct nonidentity actions')
    if (len(static_indices) != top_k or len(set(static_indices)) != top_k
            or any(i < 1 or i >= len(bank) for i in learned_indices + static_indices + group_indices)
            or (utility and (len(group_indices) != top_k or len(set(group_indices)) != top_k))):
        raise ValueError('invalid learned/static action indices')
    measured_indices = list(range(1, len(bank))) if components else learned_indices
    prefix = 'trained_prior__' if utility and details['prior_only'] else 'learned_rank__'
    candidates = controls + [Candidate(prefix + bank[i].name, bank[i].clip)
                             for i in measured_indices]
    encoded, observations, audit = [], [], []
    # Hash cache includes geometry; identical byte arrays at different shapes
    # cannot reuse an encode. Codec and QP are fixed throughout this selection.
    pixel_cache = {}
    prediction_cache = {anchor.data: anchor_predictions}
    for index, candidate in enumerate(candidates):
        key = (candidate.clip.shape, hashlib.sha256(candidate.clip.tobytes()).digest())
        result = anchor if index == 0 else pixel_cache.get(key)
        if result is None:
            result = codec.roundtrip(candidate.clip)
        pixel_cache[key] = result
        encoded.append(result)
        if result.data not in prediction_cache:
            prediction_cache[result.data] = [t.probabilities(result.decoded) for t in teachers]
        distances, decisions = relative_guard('ar', source_predictions, anchor_predictions,
                                              prediction_cache[result.data], cfg)
        observations.append(Observation(candidate.name, result.coded_bytes, distances, decisions))
        action = measured_indices[index - len(controls)] if index >= len(controls) else None
        audit.append({'name': candidate.name, 'profile': names[action] if action else None,
                      'action_index': action, 'proposed_by_learned': action in learned_indices,
                      'proposed_by_static': action in static_indices,
                      'proposed_by_group_static': action in group_indices,
                      'coded_bytes': result.coded_bytes,
                      'relative_task_distance': [float(d) if np.isfinite(d) else None for d in distances],
                      'preserves_decision': list(decisions), 'codec_seconds': result.seconds,
                      'stream_sha256': hashlib.sha256(result.data).hexdigest(),
                      'geometry': list(candidate.clip.shape)})
    if utility:
        # Persist proposal order and its encoder-only input before bank trials.
        # This makes contribution and prior-only fallbacks independently auditable.
        audit[0]['ranking_context'] = context.tolist()
        audit[0]['ranking_context_sha256'] = hashlib.sha256(context.tobytes()).hexdigest()
        audit[0]['learned_order'] = learned_indices
        audit[0]['global_static_order'] = static_indices
        audit[0]['group_static_order'] = group_indices
        audit[0]['proposal_details'] = details
        if portfolio:
            audit[0]['anchor_decoded_sha256'] = hashlib.sha256(anchor.decoded.tobytes()).hexdigest()
    positions = {action: len(controls) + i for i, action in enumerate(measured_indices)}
    control_subset = list(range(len(controls)))
    learned_subset = [positions[i] for i in learned_indices]
    static_subset = [positions[i] for i in static_indices] if components else []

    def choose(subset):
        return subset[select([observations[i] for i in subset], cfg['ar_kl_slack'], cfg['min_savings'])]

    chosen = choose(control_subset + learned_subset)
    if not components:
        return anchor, encoded[chosen], candidates[chosen].name, audit
    subsets = {'controls': control_subset,
               'learned_guarded': [0] + learned_subset,
               'static_adaptive': control_subset + static_subset,
               'bank_oracle': list(range(len(candidates)))}
    if utility:
        subsets['group_static_adaptive'] = control_subset + [positions[i] for i in group_indices]
        # Old-bank upper bound isolates added filter capacity on the same source.
        subsets['v25_bank_oracle'] = control_subset + [positions[i] for i in range(1, len(ACTION_NAMES))]
        if portfolio and tuple(names[:len(STABILIZED_NAMES)]) == STABILIZED_NAMES:
            subsets['v26_bank_oracle'] = control_subset + [positions[i] for i in range(1, len(STABILIZED_NAMES))]
    alternatives = {}
    for arm, subset in subsets.items():
        winner = choose(subset)
        alternatives[arm] = (encoded[winner], candidates[winner].name)
    raw = learned_subset[0]
    alternatives['learned_raw'] = (encoded[raw], candidates[raw].name)
    return anchor, encoded[chosen], candidates[chosen].name, audit, alternatives


def ranking_diagnostics(rows, components, qps):
    """Actual marginal bytes versus controls and equal-budget static proposals."""
    output = {}
    for qp in qps:
        anchor = {r['id']: r for r in rows if r['arm'] == 'anchor' and r['qp'] == qp}
        adaptive = {r['id']: r for r in rows if r['arm'] == 'adaptive' and r['qp'] == qp}
        if not anchor:
            continue
        denominator = sum(r['coded_bytes'] for r in anchor.values())
        primary_bytes = sum(r['coded_bytes'] for r in adaptive.values())
        entry = {'points': len(anchor), 'anchor_bytes': denominator,
                 'adaptive_bytes': primary_bytes,
                 'selected_learned_points': sum(r['candidate'].startswith('learned_rank__') for r in adaptive.values()),
                 'selected_trained_prior_points': sum(r['candidate'].startswith('trained_prior__') for r in adaptive.values())}
        for arm in ('controls', 'static_adaptive', 'group_static_adaptive', 'v25_bank_oracle', 'v26_bank_oracle', 'bank_oracle'):
            comparison = {r['id']: r for r in components if r['arm'] == arm and r['qp'] == qp}
            if set(comparison) != set(anchor):
                continue
            total = sum(r['coded_bytes'] for r in comparison.values())
            entry[arm] = {'coded_bytes': total, 'incremental_saving_pct_of_anchor': 100 * (total - primary_bytes) / denominator,
                          'adaptive_byte_wins': sum(adaptive[i]['coded_bytes'] < comparison[i]['coded_bytes'] for i in anchor),
                          'adaptive_byte_losses': sum(adaptive[i]['coded_bytes'] > comparison[i]['coded_bytes'] for i in anchor)}
        output[str(qp)] = entry
    return output
