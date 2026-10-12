"""Source-blocked TRAIN fitting of complementary, conditional K3 portfolios."""
from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np
import torch
from torch import nn

from .data import partition
from .utility_ranking import (CONTEXT_DIM, CONTEXT_SCHEMA, QPS, _contexts,
                              _groups, _utility, build_utility_context, source_folds)


CV_GRID = tuple((neighbors, mix) for neighbors in (8, 16, 32) for mix in (0., .5, 1.))
REGIMES = ('low', 'high')
BUFFERS = ('memory_context', 'memory_utility', 'mean', 'scale')


def _greedy(utility, weights, top_k):
    """Weighted maximum coverage; each next action must add marginal utility."""
    covered = np.zeros(len(utility), np.float64)
    chosen, marginal = [], []
    for _ in range(top_k):
        scores = weights @ np.maximum(utility - covered[:, None], 0.)
        scores[chosen] = -np.inf
        action = int(np.argmax(scores))  # Registry order breaks every exact tie.
        chosen.append(action)
        marginal.append(float(scores[action]))
        covered = np.maximum(covered, utility[:, action])
    return [index + 1 for index in chosen], marginal


def _weights(state, query, neighbors, mix):
    memory = state['memory_context']
    group = int(_groups(query[None])[0])
    indices = np.flatnonzero(_groups(memory) == group)
    fallback = len(indices) == 0
    if fallback:
        indices = np.arange(len(memory))
    prior = np.full(len(indices), 1. / len(indices))
    if mix == 0. or fallback:
        return indices, prior, prior, fallback
    # Codec/QP select the group and are excluded from source/teacher distance.
    delta = ((memory[indices, 2:] - query[None, 2:]) / state['scale'][None, 2:])
    distance = np.mean(delta * delta, axis=1)
    nearest = np.argsort(distance, kind='stable')[:min(neighbors, len(indices))]
    bandwidth = max(float(np.median(distance[nearest])), 1e-12)
    kernel = np.exp(-(distance[nearest] - distance[nearest].min()) / bandwidth)
    local = np.zeros(len(indices))
    local[nearest] = kernel / kernel.sum()
    return indices, (1. - mix) * prior + mix * local, prior, fallback


def _fit_memory(x, y):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    return {'memory_context': x.copy(), 'memory_utility': y.copy(), 'mean': mean,
            'scale': np.where(scale < 1e-6, 1., scale)}


def _orders(state, x, neighbors, mix):
    orders = []
    for query in x:
        indices, weights, _, _ = _weights(state, query, neighbors, mix)
        orders.append(_greedy(state['memory_utility'][indices], weights, 3)[0])
    return np.asarray(orders, np.int64).reshape(-1, 3)


def _static_orders(state, x, *, group):
    if group:
        return _orders(state, x, 32, 0.)
    utility = state['memory_utility']
    order = _greedy(utility, np.full(len(utility), 1. / len(utility)), 3)[0]
    return np.tile(order, (len(x), 1))


def _metrics(orders, target, anchors, x):
    if len(target) == 0:
        return {'records': 0, 'saved_bytes': 0, 'feasible_records': 0,
                'mean_guarded_saving_pct': 0., 'total_byte_saving_pct': 0.,
                'macro_codec_qp_mean_saving_pct': 0., 'observed_codec_qp_groups': 0,
                'top1_counts': {}}
    best = np.take_along_axis(target, orders - 1, axis=1).max(axis=1)
    # Utility is derived from integer subtraction; rounding recovers actual bytes.
    saved = np.rint(best * anchors).astype(np.int64)
    groups = _groups(x)
    return {'records': len(target), 'saved_bytes': int(saved.sum()),
            'feasible_records': int((best > 0.).sum()),
            'mean_guarded_saving_pct': float(100. * best.mean()),
            'total_byte_saving_pct': float(100. * saved.sum() / anchors.sum()),
            'macro_codec_qp_mean_saving_pct': float(np.mean(
                [100. * best[groups == group].mean() for group in sorted(set(groups))])),
            'observed_codec_qp_groups': len(set(groups)),
            'top1_counts': dict(Counter(str(int(index)) for index in orders[:, 0]))}


def _regime_masks(x):
    high = np.rint(x[:, 0] * 51).astype(int) >= 40
    return {'low': ~high, 'high': high}


class PortfolioRankPreprocessor(nn.Module):
    schema = 'adaptive-vcm-portfolio-v6'
    candidate_name = 'learned_portfolio'
    context_dim = CONTEXT_DIM
    build_context = staticmethod(build_utility_context)

    def __init__(self, action_names, state, recipes, source_ids):
        super().__init__()
        names = tuple(action_names)
        if (len(names) < 4 or names[0] != 'identity' or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name for name in names)):
            raise ValueError('identity and at least three distinct proposed actions are required')
        self.action_names, self.task, self.width = names, 'ar', CONTEXT_DIM
        if not isinstance(recipes, dict) or set(recipes) != set(REGIMES):
            raise ValueError('portfolio checkpoint needs exactly two registered QP regimes')
        self.recipes = {regime: dict(recipes[regime]) for regime in REGIMES}
        self.source_ids = list(source_ids)
        self.utility_target_scope = 'anchor_referenced_saving'
        for name in BUFFERS:
            self.register_buffer(name, torch.as_tensor(state[name], dtype=torch.float64).clone())
        self._validate()
        y = self.memory_utility.cpu().numpy()
        self.static_action_order = _greedy(y, np.full(len(y), 1. / len(y)), len(names) - 1)[0]
        self.fit_diagnostics = None

    def _validate(self):
        x, y = self.memory_context.detach().cpu().numpy(), self.memory_utility.detach().cpu().numpy()
        _contexts(x)
        if (len(x) == 0 or y.shape != (len(x), len(self.action_names) - 1)
                or self.mean.shape != (CONTEXT_DIM,) or self.scale.shape != (CONTEXT_DIM,)
                or any(not torch.isfinite(buffer).all() for buffer in self.buffers())
                or torch.any(self.scale <= 0.) or np.any(y < 0.) or np.any(y > 1.)):
            raise ValueError('invalid fitted portfolio memory/scaler')
        if (len(self.source_ids) != len(x) or any(not isinstance(value, str) or not value
                or partition(value) != 'train' for value in self.source_ids)):
            raise ValueError('portfolio memory requires original TRAIN source identities')
        fitted = _fit_memory(x, y)
        if any(not np.allclose(getattr(self, name).detach().cpu().numpy(), fitted[name],
                               atol=1e-12, rtol=1e-12) for name in ('mean', 'scale')):
            raise ValueError('portfolio scaler differs from its saved TRAIN memory')
        for recipe in self.recipes.values():
            if (set(recipe) != {'neighbors', 'mix'} or type(recipe['neighbors']) is not int
                    or (recipe['neighbors'], recipe['mix']) not in CV_GRID):
                raise ValueError('checkpoint recipe was not preregistered')

    def _state(self):
        return {name: getattr(self, name).detach().cpu().numpy() for name in BUFFERS}

    @property
    def learned_mix(self):
        # Compatibility only; actual origin must be read from proposal_details.
        return max(recipe['mix'] for recipe in self.recipes.values())

    def proposal_details(self, context, top_k=3):
        x = _contexts(context)
        if len(x) != 1 or top_k != 3:
            raise ValueError('portfolio policy uses exactly three proposals for one operating point')
        regime = 'high' if round(x[0, 0] * 51) >= 40 else 'low'
        recipe = self.recipes[regime]
        state = self._state()
        indices, weights, prior, fallback = _weights(state, x[0], recipe['neighbors'], recipe['mix'])
        utility = state['memory_utility'][indices]
        order, gains = _greedy(utility, weights, top_k)
        scores, prior_scores = weights @ utility, prior @ utility
        prior_only = recipe['mix'] == 0. or fallback
        return {'action_indices': order, 'scores': scores.tolist(),
                'prior_scores': prior_scores.tolist(), 'residual_scores': (scores - prior_scores).tolist(),
                'marginal_scores': gains, 'selected_mix': 0. if fallback else recipe['mix'],
                'neighbors': recipe['neighbors'], 'regime': regime, 'prior_only': prior_only,
                'group_records': len(indices), 'missing_group_fallback': fallback,
                'origin': ('TRAIN_global_greedy_portfolio' if fallback else
                           'TRAIN_group_greedy_portfolio' if prior_only else
                           'TRAIN_context_kernel_greedy_portfolio')}

    def rank(self, context, top_k=3):
        return self.proposal_details(context, top_k)['action_indices']

    def group_static_action_order(self, context, top_k=3):
        x = _contexts(context)
        if len(x) != 1 or top_k != 3:
            raise ValueError('group static comparator uses exactly three proposals')
        return _static_orders(self._state(), x, group=True)[0].tolist()

    def render(self, clip, protection, qp, action_index, *, anchor_decoded=None):
        from .task_bank import ACTION_NAMES as old_names, build_task_bank
        from .stabilized_bank import ACTION_NAMES as stable_names, build_stabilized_bank
        if self.action_names == old_names:
            bank = build_task_bank(clip, protection, qp)
        elif self.action_names == stable_names:
            bank = build_stabilized_bank(clip, protection, qp)
        else:
            from .anchor_bank import ACTION_NAMES as anchor_names, build_anchor_bank
            if self.action_names != anchor_names:
                raise ValueError('portfolio checkpoint has no registered pixel bank')
            bank = build_anchor_bank(clip, protection, qp, anchor_decoded)
        if tuple(candidate.name for candidate in bank) != self.action_names:
            raise ValueError('portfolio checkpoint bank differs from executable bank')
        if type(action_index) is not int or not 1 <= action_index < len(bank):
            raise ValueError('invalid nonidentity action index')
        return bank[action_index]

    def checkpoint_state(self):
        return {'schema': self.schema, 'task': 'ar', 'width': CONTEXT_DIM,
                'model': self.state_dict(), 'action_names': list(self.action_names),
                'context_schema': CONTEXT_SCHEMA, 'context_dim': CONTEXT_DIM,
                'static_action_order': self.static_action_order,
                'utility_target_scope': self.utility_target_scope,
                'recipes': self.recipes, 'source_ids': self.source_ids,
                'fit_diagnostics': self.fit_diagnostics}


def fit_portfolio_model(context, safety, log_rate, source_ids, action_names, min_savings=.01,
                        *, folds=4, anchor_bytes=None, baseline_log_rate=None,
                        action_bytes=None, control_bytes=None):
    """Select low/high recipes only by source-blocked measured TRAIN retrieval."""
    x = _contexts(context)
    y = _utility(safety, log_rate, min_savings, baseline_log_rate,
                 anchor_bytes=anchor_bytes, action_bytes=action_bytes, control_bytes=control_bytes)
    ids = list(source_ids)
    if len(x) != len(y) or len(ids) != len(x) or y.shape[1] != len(action_names) - 1:
        raise ValueError('inconsistent source/action records')
    if any(not isinstance(value, str) or not value or partition(value) != 'train' for value in ids):
        raise ValueError('portfolio fitting cannot access DEV/TEST sources')
    assignment = source_folds(ids, folds)
    anchors = np.asarray(anchor_bytes, np.int64)
    states, fold_provenance = [], []
    global_static, group_static = np.zeros((len(x), 3), np.int64), np.zeros((len(x), 3), np.int64)
    for fold in range(folds):
        train, valid = assignment != fold, assignment == fold
        state = _fit_memory(x[train], y[train])
        states.append(state)
        global_static[valid] = _static_orders(state, x[valid], group=False)
        group_static[valid] = _static_orders(state, x[valid], group=True)
        fold_provenance.append({'fold': fold, 'train_source_ids': sorted(set(str(value) for value in np.asarray(ids)[train])),
            'valid_source_ids': sorted(set(str(value) for value in np.asarray(ids)[valid])),
            'train_records': int(train.sum()), 'valid_records': int(valid.sum()),
            'train_context_mean': state['mean'].tolist(), 'train_context_scale': state['scale'].tolist()})
    masks = _regime_masks(x)
    cv_results, predictions = [], []
    for neighbors, mix in CV_GRID:
        orders = np.zeros((len(x), 3), np.int64)
        for fold, state in enumerate(states):
            valid = assignment == fold
            orders[valid] = _orders(state, x[valid], neighbors, mix)
        result = {'neighbors': neighbors, 'mix': mix,
                  'overall': _metrics(orders, y, anchors, x)}
        for regime, mask in masks.items():
            result[regime] = _metrics(orders[mask], y[mask], anchors[mask], x[mask])
        cv_results.append(result)
        predictions.append(orders)
    recipes, selected = {}, np.zeros((len(x), 3), np.int64)
    for regime, mask in masks.items():
        metric = 'saved_bytes' if regime == 'low' else 'macro_codec_qp_mean_saving_pct'
        best = min(range(len(cv_results)), key=lambda index: (
            -cv_results[index][regime][metric], cv_results[index]['mix'], -cv_results[index]['neighbors']))
        recipes[regime] = {key: cv_results[best][key] for key in ('neighbors', 'mix')}
        selected[mask] = predictions[best][mask]
    state = _fit_memory(x, y)
    model = PortfolioRankPreprocessor(action_names, state, recipes, ids).eval()
    if baseline_log_rate is not None:
        model.utility_target_scope = 'marginal_saving_beyond_guarded_controls'
    oracle = np.argsort(-y, axis=1, kind='stable')[:, :3] + 1
    train_orders = np.asarray([model.rank(query) for query in x], np.int64)
    diagnostics = {'scope': 'source-blocked TRAIN crossvalidation; no DEV/TEST measurements',
        'context_dim': CONTEXT_DIM, 'records': len(x), 'unique_sources': len(set(ids)),
        'utility_target_scope': model.utility_target_scope, 'label_precision': 'actual_integer_bytes_v1',
        'rate_denominator': 'original anchor bytes, including every stream header',
        'folds': folds, 'fold_assignment': assignment.tolist(), 'source_ids': ids,
        'selection_metric': {'low': 'maximize OOF actual bytes saved',
                             'high': 'maximize macro codec/QP OOF mean marginal saving percentage'},
        'selected_recipes': recipes, 'prior_only': all(recipe['mix'] == 0. for recipe in recipes.values()),
        'cv_grid': cv_results, 'selected_oof': _metrics(selected, y, anchors, x),
        'global_static_oof': _metrics(global_static, y, anchors, x),
        'group_static_oof': _metrics(group_static, y, anchors, x),
        'bank_oracle': _metrics(oracle, y, anchors, x),
        'final_train_resubstitution': _metrics(train_orders, y, anchors, x),
        'oof_action_indices': selected.tolist(), 'global_static_oof_action_indices': global_static.tolist(),
        'group_static_oof_action_indices': group_static.tolist(),
        'records_sha256': hashlib.sha256(x.tobytes() + y.tobytes() + '\n'.join(ids).encode()).hexdigest(),
        'limitations': 'Recipe-selected TRAIN OOF is not independent DEV; every winner still requires actual encode and guard.'}
    arms = (('selected_oof', selected), ('global_static_oof', global_static),
            ('group_static_oof', group_static), ('bank_oracle', oracle))
    diagnostics['by_regime'] = {regime: {name: _metrics(order[mask], y[mask], anchors[mask], x[mask])
        for name, order in arms} for regime, mask in masks.items()}
    diagnostics['by_codec_qp'] = {}
    for group in sorted(set(_groups(x))):
        mask = _groups(x) == group
        label = f"{'h265' if group >= len(QPS) else 'h264'}/{QPS[group % len(QPS)]}"
        diagnostics['by_codec_qp'][label] = {name: _metrics(order[mask], y[mask], anchors[mask], x[mask])
                                            for name, order in arms}
    diagnostics['oof_folds'] = [{**provenance, **{name: _metrics(order[assignment == fold],
        y[assignment == fold], anchors[assignment == fold], x[assignment == fold])
        for name, order in arms}} for fold, provenance in enumerate(fold_provenance)]
    model.fit_diagnostics = diagnostics
    return model, diagnostics


def load_portfolio_preprocessor(state, task='ar', action_names=None):
    if action_names is None:
        from .task_bank import ACTION_NAMES as old_names
        from .stabilized_bank import ACTION_NAMES as stable_names
        registered = [old_names, stable_names]
        try:
            from .anchor_bank import ACTION_NAMES as anchor_names
            registered.append(anchor_names)
        except ImportError:
            pass  # Bank remains unavailable until the V27 pixel module is installed.
        saved = tuple(state.get('action_names', ()))
        if saved not in registered:
            raise ValueError('portfolio checkpoint has no registered pixel bank')
        action_names = saved
    if (state.get('schema') != PortfolioRankPreprocessor.schema or task != 'ar' or state.get('task') != 'ar'
            or state.get('context_schema') != CONTEXT_SCHEMA or state.get('context_dim') != CONTEXT_DIM
            or tuple(state.get('action_names', ())) != tuple(action_names)):
        raise ValueError('incompatible portfolio schema/context/action bank')
    buffers = state.get('model', {})
    if set(buffers) != set(BUFFERS) or any(not torch.is_tensor(value) for value in buffers.values()):
        raise ValueError('invalid portfolio checkpoint buffers')
    memory = {key: value.detach().cpu().numpy() for key, value in buffers.items()}
    model = PortfolioRankPreprocessor(action_names, memory, state['recipes'], state['source_ids'])
    if state.get('static_action_order') != model.static_action_order:
        raise ValueError('static comparator differs from TRAIN memory')
    model.utility_target_scope = state.get('utility_target_scope')
    if model.utility_target_scope not in ('anchor_referenced_saving', 'marginal_saving_beyond_guarded_controls'):
        raise ValueError('missing or unsupported utility target scope')
    model.fit_diagnostics = state.get('fit_diagnostics')
    return model.eval()
