"""Compact, source-blocked TRAIN utility ranking with auditable group priors."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .codec import reference_bpp
from .data import fingerprint, partition
from .ranking import CONTEXT_DIM as LEGACY_DIM, CONTEXT_SCHEMA as LEGACY_SCHEMA, build_rank_context


CONTEXT_DIM = 41
CONTEXT_SCHEMA = "teacher-scalars-source-statistics-anchor-bpp-v1"
QPS = (30, 35, 40, 45, 50)
CV_GRID = tuple((alpha, mix, pseudo) for alpha in (10., 100., 1000.)
                for mix in (0., .25, .5, 1.) for pseudo in (4., 16.))


def compact_context(legacy_context, anchor_bpp: float) -> np.ndarray:
    old = np.asarray(legacy_context, np.float64)
    if old.shape != (LEGACY_DIM,) or not np.isfinite(old).all() or not math.isfinite(anchor_bpp) or anchor_bpp <= 0:
        raise ValueError("invalid source/teacher context or actual anchor bpp")
    # The old layout is source statistics16, then two (probabilities800, scalars12)
    # blocks. Class-indexed probability entries are intentionally omitted.
    result = np.concatenate((old[:16], old[816:828], old[1628:1640], [math.log1p(anchor_bpp)]))
    return result.astype(np.float64)


def build_utility_context(clip, qp, codec, protection, source_predictions,
                          anchor_predictions, *, anchor_bpp: float) -> np.ndarray:
    return compact_context(build_rank_context(clip, qp, codec, protection,
                                              source_predictions, anchor_predictions), anchor_bpp)


def source_folds(source_ids, folds: int = 4) -> np.ndarray:
    """Fold identity depends only on source ID; all codec/QP records stay together."""
    ids = [str(value) for value in source_ids]
    unique = sorted(set(ids), key=lambda value: hashlib.sha256(("utility-train-cv-v1:" + value).encode()).digest())
    if type(folds) is not int or folds < 2 or len(unique) < folds:
        raise ValueError("need at least one distinct source per TRAIN fold")
    assignments = {value: index % folds for index, value in enumerate(unique)}
    return np.asarray([assignments[value] for value in ids], np.int64)


def _contexts(value):
    x = np.asarray(value, np.float64)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2 or x.shape[1] != CONTEXT_DIM or not np.isfinite(x).all():
        raise ValueError("invalid compact utility context")
    qps = np.rint(x[:, 0] * 51).astype(int)
    codecs = np.rint(x[:, 1]).astype(int)
    if not np.isin(qps, QPS).all() or not np.isin(codecs, (0, 1)).all() or np.any(x[:, -1] <= 0):
        raise ValueError("unregistered context codec/QP/anchor bpp")
    return x


def _groups(x):
    qp_indices = np.asarray([QPS.index(int(round(value * 51))) for value in x[:, 0]])
    return np.rint(x[:, 1]).astype(int) * len(QPS) + qp_indices


def _utility(safety, log_rate, min_savings, baseline_log_rate=None, *,
             anchor_bytes=None, action_bytes=None, control_bytes=None):
    safety, log_rate = np.asarray(safety, np.float64), np.asarray(log_rate, np.float64)
    if safety.ndim != 2 or safety.shape != log_rate.shape or not np.isfinite(safety).all() or not np.isfinite(log_rate).all():
        raise ValueError("invalid all-action measured targets")
    if not np.isin(safety, (0., 1.)).all() or not 0 <= min_savings < 1:
        raise ValueError("invalid measured safety or byte threshold")
    def integer_bytes(value, shape):
        if value is None:
            raise ValueError('utility labels require actual integer coded bytes')
        array = np.asarray(value, np.float64)
        if (array.shape != shape or not np.isfinite(array).all() or np.any(array <= 0)
                or np.any(array > 2 ** 53) or np.any(array != np.rint(array))):
            raise ValueError('invalid actual integer coded bytes')
        return array.astype(np.int64)
    anchors = integer_bytes(anchor_bytes, (len(log_rate),))
    actions = integer_bytes(action_bytes, log_rate.shape)
    controls = anchors if baseline_log_rate is None else integer_bytes(control_bytes, anchors.shape)
    if np.any(controls > anchors):
        raise ValueError('guarded controls cannot exceed anchor bytes')
    if not np.allclose(log_rate, np.log(actions / anchors[:, None]), atol=2e-7, rtol=2e-7):
        raise ValueError('log-rate labels differ from actual coded bytes')
    if baseline_log_rate is not None:
        baseline = np.asarray(baseline_log_rate, np.float64)
        if (baseline.shape != anchors.shape or not np.isfinite(baseline).all()
                or not np.allclose(baseline, np.log(controls / anchors), atol=1e-12, rtol=1e-12)):
            raise ValueError('baseline differs from actual guarded-control bytes')
    feasible = (safety == 1.) & (actions <= anchors[:, None] * (1 - min_savings))
    # Subtract integers first: equal-byte actions receive exactly zero credit.
    return np.where(feasible, np.maximum(0, controls[:, None] - actions) / anchors[:, None], 0.)


def _prior(x, y, pseudo):
    groups = _groups(x)
    global_mean = y.mean(axis=0)
    means = np.empty((2 * len(QPS), y.shape[1]), np.float64)
    counts = np.zeros(2 * len(QPS), np.int64)
    for group in range(len(means)):
        subset = y[groups == group]
        counts[group] = len(subset)
        means[group] = ((subset.sum(axis=0) + pseudo * global_mean) / (len(subset) + pseudo)
                        if len(subset) + pseudo else global_mean)
    return means, counts, global_mean


def _fit(x, y, alpha, mix, pseudo):
    prior, counts, global_mean = _prior(x, y, pseudo)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-6, 1., scale)
    z = (x - mean) / scale
    residual = y - prior[_groups(x)]
    intercept = residual.mean(axis=0)
    coefficients = np.linalg.solve(z.T @ z + alpha * np.eye(x.shape[1]), z.T @ (residual - intercept))
    return {"prior": prior, "counts": counts, "global_mean": global_mean,
            "group_static_prior": _prior(x, y, 0.)[0],
            "mean": mean, "scale": scale, "coefficients": coefficients,
            "intercept": intercept, "alpha": alpha, "mix": mix, "pseudo": pseudo}


def _predict(state, x):
    prior = state["prior"][_groups(x)]
    residual = ((x - state["mean"]) / state["scale"]) @ state["coefficients"] + state["intercept"]
    return np.maximum(0., prior + state["mix"] * residual), prior, residual


def _retrieval(scores, target, anchor_bytes, top_k):
    orders = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    retrieved = np.take_along_axis(target, orders, axis=1).max(axis=1)
    saved = np.rint(retrieved * anchor_bytes).astype(np.int64)
    return {"mean_guarded_saving_pct": float(100 * retrieved.mean()),
            "total_byte_saving_pct": float(100 * saved.sum() / anchor_bytes.sum()),
            "feasible_records": int((retrieved > 0).sum()), "records": len(target),
            "saved_bytes": int(saved.sum()),
            "top1_counts": dict(Counter(str(int(i + 1)) for i in orders[:, 0]))}


class UtilityRankPreprocessor(nn.Module):
    schema = "adaptive-vcm-utility-v5"
    candidate_name = "learned_utility"
    context_dim = CONTEXT_DIM
    build_context = staticmethod(build_utility_context)

    def __init__(self, action_names, state=None):
        super().__init__()
        names = tuple(action_names)
        if len(names) < 4 or names[0] != "identity" or len(set(names)) != len(names):
            raise ValueError("identity and at least three distinct proposed actions are required")
        self.action_names, self.task, self.width = names, "ar", CONTEXT_DIM
        count = len(names) - 1
        for name, values in (("prior", np.zeros((10, count))), ("group_static_prior", np.zeros((10, count))),
                             ("counts", np.zeros(10)),
                             ("global_mean", np.zeros(count)), ("mean", np.zeros(CONTEXT_DIM)),
                             ("scale", np.ones(CONTEXT_DIM)), ("coefficients", np.zeros((CONTEXT_DIM, count))),
                             ("intercept", np.zeros(count))):
            self.register_buffer(name, torch.as_tensor(values, dtype=torch.float64))
        self.alpha, self.mix, self.pseudo = 100., 0., 16.
        self.utility_target_scope = "anchor_referenced_saving"
        self.static_action_order = list(range(1, len(names)))
        if state is not None:
            for name in ("prior", "group_static_prior", "counts", "global_mean", "mean", "scale", "coefficients", "intercept"):
                getattr(self, name).copy_(torch.as_tensor(state[name], dtype=torch.float64))
            self.alpha, self.mix, self.pseudo = state["alpha"], state["mix"], state["pseudo"]
            self.static_action_order = [int(i + 1) for i in np.argsort(-state["global_mean"], kind="stable")]

    def _state(self):
        return {**{name: getattr(self, name).detach().cpu().numpy() for name in
                    ("prior", "group_static_prior", "counts", "global_mean", "mean", "scale", "coefficients", "intercept")},
                "alpha": self.alpha, "mix": self.mix, "pseudo": self.pseudo}

    @property
    def learned_mix(self):
        return self.mix

    def proposal_details(self, context, top_k=3):
        x = _contexts(context)
        if len(x) != 1 or top_k != 3 or top_k >= len(self.action_names):
            raise ValueError("utility policy uses exactly three proposals for one operating point")
        scores, prior, residual = _predict(self._state(), x)
        indices = [int(i + 1) for i in np.argsort(-scores[0], kind="stable")[:top_k]]
        return {"action_indices": indices, "scores": scores[0].tolist(),
                "prior_scores": prior[0].tolist(), "residual_scores": residual[0].tolist(),
                "selected_mix": self.mix, "prior_only": self.mix == 0.,
                "origin": "TRAIN_group_prior" if self.mix == 0. else "TRAIN_group_prior_plus_regularized_residual"}

    def rank(self, context, top_k=3):
        return self.proposal_details(context, top_k)["action_indices"]

    def group_static_action_order(self, context, top_k=3):
        x = _contexts(context)
        if len(x) != 1 or top_k != 3:
            raise ValueError("group static comparator uses exactly three proposals")
        scores = self.group_static_prior.detach().cpu().numpy()[_groups(x)[0]]
        return [int(i + 1) for i in np.argsort(-scores, kind="stable")[:top_k]]

    def render(self, clip, protection, qp, action_index):
        from .task_bank import ACTION_NAMES as old_names, build_task_bank
        from .stabilized_bank import ACTION_NAMES as new_names, build_stabilized_bank
        if self.action_names == old_names:
            bank = build_task_bank(clip, protection, qp)
        elif self.action_names == new_names:
            bank = build_stabilized_bank(clip, protection, qp)
        else:
            raise ValueError('utility checkpoint has no registered pixel bank')
        if tuple(candidate.name for candidate in bank) != self.action_names:
            raise ValueError("utility checkpoint bank differs from executable bank")
        if type(action_index) is not int or not 1 <= action_index < len(bank):
            raise ValueError("invalid nonidentity action index")
        return bank[action_index]

    def checkpoint_state(self):
        return {"schema": self.schema, "task": "ar", "model": self.state_dict(), "width": CONTEXT_DIM,
                "action_names": list(self.action_names), "context_schema": CONTEXT_SCHEMA,
                "context_dim": CONTEXT_DIM, "static_action_order": self.static_action_order,
                "utility_target_scope": self.utility_target_scope,
                "recipe": {"alpha": self.alpha, "mix": self.mix, "pseudo": self.pseudo}}


def fit_utility_model(context, safety, log_rate, source_ids, action_names, min_savings=.01,
                      *, folds=4, anchor_bytes=None, baseline_log_rate=None,
                      action_bytes=None, control_bytes=None):
    """Choose one fixed recipe exclusively by source-blocked TRAIN retrieval."""
    x = _contexts(context)
    y = _utility(safety, log_rate, min_savings, baseline_log_rate,
                 anchor_bytes=anchor_bytes, action_bytes=action_bytes, control_bytes=control_bytes)
    if len(x) != len(y) or len(source_ids) != len(x) or y.shape[1] != len(action_names) - 1:
        raise ValueError("inconsistent source/action records")
    if any(partition(str(value)) != "train" for value in source_ids):
        raise ValueError("utility fitting cannot access DEV/TEST sources")
    assignment = source_folds(source_ids, folds)
    bytes_ = np.asarray(anchor_bytes, np.float64)
    if bytes_.shape != (len(x),) or not np.isfinite(bytes_).all() or np.any(bytes_ <= 0):
        raise ValueError("invalid actual anchor bytes")
    cv_results, predictions = [], []
    global_static = np.zeros_like(y)
    group_static = np.zeros_like(y)
    for fold in range(folds):
        train, valid = assignment != fold, assignment == fold
        global_static[valid] = y[train].mean(axis=0)
        group_static[valid] = _prior(x[train], y[train], 0.)[0][_groups(x[valid])]
    for alpha, mix, pseudo in CV_GRID:
        scores = np.zeros_like(y)
        for fold in range(folds):
            train, valid = assignment != fold, assignment == fold
            scores[valid] = _predict(_fit(x[train], y[train], alpha, mix, pseudo), x[valid])[0]
        metric = _retrieval(scores, y, bytes_, 3)
        cv_results.append({"alpha": alpha, "mix": mix, "prior_pseudo_count": pseudo, **metric})
        predictions.append(scores)
    # Maximize actual held-out bytes saved; equal scores prefer prior-only and
    # stronger regularization. This tie rule is fixed before examining data.
    best = min(range(len(cv_results)), key=lambda index: (-cv_results[index]["saved_bytes"],
               cv_results[index]["mix"], -cv_results[index]["alpha"], -cv_results[index]["prior_pseudo_count"]))
    chosen = cv_results[best]
    state = _fit(x, y, chosen["alpha"], chosen["mix"], chosen["prior_pseudo_count"])
    model = UtilityRankPreprocessor(action_names, state).eval()
    if baseline_log_rate is not None:
        model.utility_target_scope = "marginal_saving_beyond_guarded_controls"
    train_scores = _predict(state, x)[0]
    diagnostics = {"scope": "source-blocked TRAIN crossvalidation; no DEV/TEST measurements",
                   "context_dim": CONTEXT_DIM, "unique_sources": len(set(source_ids)), "records": len(x),
                   "utility_target_scope": model.utility_target_scope,
                   "label_precision": "actual_integer_bytes_v1",
                   "rate_denominator": "original anchor bytes, including every stream header",
                   "folds": folds, "fold_assignment": assignment.tolist(), "source_ids": list(source_ids),
                   "selection_metric": "maximize total held-out TRAIN actual coded bytes saved with exactly3proposals",
                   "selected_recipe": {key: chosen[key] for key in ("alpha", "mix", "prior_pseudo_count")},
                   "prior_only": chosen["mix"] == 0., "cv_grid": cv_results,
                   "selected_oof": _retrieval(predictions[best], y, bytes_, 3),
                   "global_static_oof": _retrieval(global_static, y, bytes_, 3),
                   "group_static_oof": _retrieval(group_static, y, bytes_, 3),
                   "bank_oracle": _retrieval(y, y, bytes_, 3),
                   "final_train_resubstitution": _retrieval(train_scores, y, bytes_, 3),
                   "limitations": "Recipe selection itself uses TRAIN OOF folds; selected OOF is not an independent DEV score."}
    diagnostics["by_codec_qp"] = {}
    for group in sorted(set(_groups(x))):
        slot = _groups(x) == group
        label = f"{'h265' if group >= len(QPS) else 'h264'}/{QPS[group % len(QPS)]}"
        diagnostics["by_codec_qp"][label] = {
            name: _retrieval(scores[slot], y[slot], bytes_[slot], 3)
            for name, scores in (("selected_oof", predictions[best]), ("global_static_oof", global_static),
                                ("group_static_oof", group_static), ("bank_oracle", y))}
    diagnostics["oof_folds"] = [{"fold": fold,
        "selected_oof": _retrieval(predictions[best][assignment == fold], y[assignment == fold], bytes_[assignment == fold], 3),
        "global_static_oof": _retrieval(global_static[assignment == fold], y[assignment == fold], bytes_[assignment == fold], 3),
        "group_static_oof": _retrieval(group_static[assignment == fold], y[assignment == fold], bytes_[assignment == fold], 3)}
        for fold in range(folds)]
    return model, diagnostics


def load_record_directory(path: Path):
    """Read reusable TRAIN measurements with manifest/hash/action/split gates."""
    path = Path(path)
    manifest = json.loads((path / "training_manifest.json").read_text(encoding="utf-8"))
    measured_path = path / "measurements.jsonl"
    if hashlib.sha256(measured_path.read_bytes()).hexdigest() != manifest.get("measurements_sha256"):
        raise ValueError("TRAIN measurement archive hash differs from manifest")
    legacy = manifest.get('context_schema') == LEGACY_SCHEMA and manifest.get('context_dim') == LEGACY_DIM
    compact = manifest.get('context_schema') == CONTEXT_SCHEMA and manifest.get('context_dim') == CONTEXT_DIM
    if not (legacy or compact):
        raise ValueError("unsupported replay measurement context")
    bank_path = Path(__file__).with_name("task_bank.py")
    bank_hash = hashlib.sha256(bank_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if manifest["code"]["files_sha256"].get("adaptive_vcm/task_bank.py") != bank_hash:
        raise ValueError("cached measurement bank code differs from executable bank")
    names = tuple(manifest["action_names"])
    from .task_bank import ACTION_NAMES as old_names
    from .stabilized_bank import ACTION_NAMES as new_names
    anchor_bank = False
    if names not in (old_names, new_names):
        from .anchor_bank import ACTION_NAMES as anchor_names
        if names != anchor_names:
            raise ValueError('unregistered cached action bank')
        anchor_bank = True
    if names == new_names or anchor_bank:
        new_path = Path(__file__).with_name('stabilized_bank.py')
        new_hash = hashlib.sha256(new_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if manifest['code']['files_sha256'].get('adaptive_vcm/stabilized_bank.py') != new_hash:
            raise ValueError('cached stabilization bank differs from executable bank')
    if anchor_bank:
        anchor_path = Path(__file__).with_name('anchor_bank.py')
        anchor_hash = hashlib.sha256(anchor_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if manifest['code']['files_sha256'].get('adaptive_vcm/anchor_bank.py') != anchor_hash:
            raise ValueError('cached anchor bank differs from executable bank')
    rows = list(map(json.loads, measured_path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != manifest["measurements"] or any(partition(row["source_id"]) != "train" for row in rows):
        raise ValueError("incomplete or non-TRAIN measured records")
    if (manifest.get('task') != 'ar' or not manifest['config'].get('ar_require_anchor_decision')
            or manifest['config']['min_savings'] != .01
            or fingerprint(manifest['train_ids']) != manifest['train_ids_sha256']
            or any(r['source_id'] not in manifest['train_ids'] for r in rows)):
        raise ValueError('cached source/guard training protocol mismatch')
    if anchor_bank:
        cfg = manifest['config']
        expected = {(identifier, codec, qp) for identifier in manifest['train_ids']
                    for codec in ('h264', 'h265') for qp in QPS}
        observed = [(r['source_id'], r['codec'], r['qp']) for r in rows]
        if (manifest.get('schema') != 'adaptive-vcm-training-v6' or
                cfg.get('ar_training') != 'source_validated_portfolio' or cfg['ar_kl_slack'] != .1 or
                cfg.get('ar_guard_rule') != 'anchor_relative_v2' or cfg.get('rank_top_k') != 3 or
                tuple(cfg['qps']) != QPS or len(observed) != len(expected) or set(observed) != expected):
            raise ValueError('cached complete TRAIN codec/QP grid or strict protocol differs')
        for identifier in manifest['train_ids']:
            if len({r['source_sha256'] for r in rows if r['source_id'] == identifier}) != 1:
                raise ValueError('source pixels changed inside a complete TRAIN codec/QP grid')
        archive = path / 'train_records.npz'
        if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest.get('train_records_sha256'):
            raise ValueError('cached TRAIN array archive hash differs from manifest')
    x, safety, rates, ids, sizes = [], [], [], [], []
    from .ranking import measurement_targets
    for row in rows:
        if tuple(action["name"] for action in row["actions"]) != names:
            raise ValueError("measurement bank/action order mismatch")
        if len(row.get("source_sha256", "")) != 64:
            raise ValueError("measurement is missing original source fingerprint")
        shape = tuple(row["source_shape"])
        if len(shape) != 4 or shape[-1] != 3 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError('invalid original source geometry')
        size = row["actions"][0]["coded_bytes"]
        bpp = reference_bpp(size, shape)
        if not np.isclose(bpp, row["actual_anchor_bpp"], atol=1e-12, rtol=1e-12):
            raise ValueError("measured anchor bpp differs from original-pixel denominator")
        x.append(compact_context(row['context'], bpp) if legacy else _contexts(row['context'])[0])
        expected = [row['qp'] / 51, float(row['codec'] == 'h265'),
                    math.log2(shape[1]) / 10, math.log2(shape[2]) / 10, shape[0] / 32]
        if (row['codec'] not in ('h264', 'h265') or row['qp'] not in QPS
                or not np.allclose(x[-1][:5], expected, atol=1e-7, rtol=0)):
            raise ValueError('cached context codec/QP/geometry differs from measured source')
        if compact and not np.isclose(np.expm1(x[-1][-1]), bpp, atol=1e-12, rtol=1e-12):
            raise ValueError('compact context anchor rate differs from measured bytes')
        safe, rate, _ = measurement_targets(row["actions"], slack=manifest["config"]["ar_kl_slack"],
                                            min_savings=manifest["config"]["min_savings"])
        safety.append(safe)
        rates.append(rate)
        ids.append(row["source_id"])
        sizes.append(size)
    baseline = None
    if compact:
        from .selection import Observation, select
        baseline = []
        for row in rows:
            controls = row['controls']
            anchor = row['actions'][0]
            if (tuple(c['name'] for c in controls) != tuple(manifest['config']['ar_candidates'])
                    or controls[0]['name'] != 'identity'
                    or controls[0]['coded_bytes'] != anchor['coded_bytes']
                    or controls[0]['stream_sha256'] != anchor['stream_sha256']):
                raise ValueError('cached controls are incomplete or have a different anchor')
            winner = select([Observation(r['name'], r['coded_bytes'], tuple(r['distances']), tuple(r['decisions']))
                             for r in controls], manifest['config']['ar_kl_slack'], .01)
            if (row['controls_selected'] != controls[winner]['name']
                    or row['controls_coded_bytes'] != controls[winner]['coded_bytes']):
                raise ValueError('cached guarded-control winner fields disagree')
            ratio = np.log(controls[winner]['coded_bytes'] / row['actions'][0]['coded_bytes'])
            if not np.isclose(ratio, row['baseline_log_rate'], atol=1e-12, rtol=1e-12):
                raise ValueError('cached marginal baseline differs from guarded controls')
            baseline.append(ratio)
        baseline = np.asarray(baseline)
    if anchor_bank:
        with np.load(path / 'train_records.npz') as arrays:
            expected_arrays = dict(context=np.stack(x), safety=np.stack(safety), log_rate=np.stack(rates),
                                   baseline_log_rate=np.asarray([r['baseline_log_rate'] for r in rows]))
            for key, value in expected_arrays.items():
                if key not in arrays or not np.array_equal(arrays[key], value):
                    raise ValueError('cached TRAIN arrays disagree with measured records')
    return {"context": np.stack(x), "safety": np.stack(safety), "log_rate": np.stack(rates),
            'baseline_log_rate': baseline,
            "source_ids": ids, "action_names": names, "anchor_bytes": np.asarray(sizes),
            "action_bytes": np.asarray([[a['coded_bytes'] for a in r['actions'][1:]] for r in rows]),
            "control_bytes": np.asarray([r['controls_coded_bytes'] for r in rows]) if compact else None,
            "manifest": manifest, "measurements_sha256": hashlib.sha256(measured_path.read_bytes()).hexdigest()}


def utility_checkpoint(model, *, training_config, provenance):
    measured = provenance['measurement_manifest']
    return {**model.checkpoint_state(), 'training_config': training_config, 'provenance': provenance,
            'fit_method': 'sourceblocked_train_cv_ridge', 'measurements': measured['measurements'],
            'train_ids_sha256': measured['train_ids_sha256']}


def load_utility_preprocessor(state, task="ar", action_names=None):
    if action_names is None:
        from .task_bank import ACTION_NAMES as old_names
        from .stabilized_bank import ACTION_NAMES as new_names
        saved_names = tuple(state.get('action_names', ()))
        if saved_names not in (old_names, new_names):
            raise ValueError('utility checkpoint has no registered pixel bank')
        action_names = saved_names
    if (state.get("schema") != UtilityRankPreprocessor.schema or task != "ar" or state.get("task") != "ar"
            or state.get("context_schema") != CONTEXT_SCHEMA or state.get("context_dim") != CONTEXT_DIM
            or tuple(state.get("action_names", ())) != tuple(action_names)):
        raise ValueError("incompatible utility schema/context/action bank")
    model = UtilityRankPreprocessor(action_names)
    model.load_state_dict(state["model"], strict=True)
    if any(not torch.isfinite(buffer).all() for buffer in model.buffers()) or torch.any(model.scale <= 0):
        raise ValueError("invalid fitted utility coefficients/scaler")
    model.alpha, model.mix, model.pseudo = [float(state["recipe"][key]) for key in ("alpha", "mix", "pseudo")]
    model.utility_target_scope = state.get("utility_target_scope")
    if model.utility_target_scope not in ("anchor_referenced_saving", "marginal_saving_beyond_guarded_controls"):
        raise ValueError("missing or unsupported utility target scope")
    if (model.alpha, model.mix, model.pseudo) not in CV_GRID:
        raise ValueError("checkpoint recipe was not preregistered")
    order = state["static_action_order"]
    if sorted(order) != list(range(1, len(action_names))):
        raise ValueError("invalid TRAIN static comparator order")
    model.static_action_order = list(order)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("replay output must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    bundle = load_record_directory(args.records)
    model, report = fit_utility_model(bundle["context"], bundle["safety"], bundle["log_rate"],
        bundle["source_ids"], bundle["action_names"], anchor_bytes=bundle["anchor_bytes"],
        baseline_log_rate=bundle['baseline_log_rate'], action_bytes=bundle['action_bytes'],
        control_bytes=bundle['control_bytes'])
    (args.out / "cv_diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    provenance = {"source_measurements_sha256": bundle["measurements_sha256"],
                  "measurement_config": bundle["manifest"]["config"],
                  'measurement_manifest': bundle['manifest'],
                  "measurement_code": bundle["manifest"]["code"], "cv": report}
    torch.save(utility_checkpoint(model, training_config=bundle["manifest"]["config"], provenance=provenance),
               args.out / "preprocessor_last.pth")
    print(json.dumps({key: report[key] for key in ("selected_recipe", "prior_only", "selected_oof",
                "global_static_oof", "group_static_oof", "bank_oracle")}, indent=2))


if __name__ == "__main__":
    main()
