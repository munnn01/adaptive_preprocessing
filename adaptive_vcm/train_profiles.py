"""TRAIN-only feasible-profile imitation using exact H.264/H.265 measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .analyzers import ActionAnalyzer
from .codec import StandardCodec, reference_bpp
from .data import ar_plan, fingerprint, read_video
from .evaluate import ROOT, code_manifest, validate_config, write_json
from .preprocessing import normalize_map
from .profiles import ProfilePreprocessor, feasible_profile_target
from .rateaware import semantic_protection
from .selection import relative_guard
from .train_rateaware import pixels


def train(args):
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(cfg)
    if args.task != 'ar' or cfg.get('ar_training') != 'feasible_profile_imitation':
        raise ValueError('V24 profile training requires the registered AR recipe')
    identity_weight = cfg.get('identity_target_weight', .25)
    if not 0 < identity_weight <= 1:
        raise ValueError('identity target weight must be in (0, 1]')
    if min(args.steps, args.count, args.width) < 1 or not 0 < args.lr < 1:
        raise ValueError('invalid training budget')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('training output must be empty')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    plan, _ = ar_plan(args.root, 'train', args.count)
    teachers = [ActionAnalyzer(name, device) for name in cfg['ar_teachers']]
    model = ProfilePreprocessor(args.width).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    groups = [(c, q) for c in ('h264', 'h265') for q in cfg['qps']]
    weights = np.array([1 if q < 40 else 2 if q < 45 else 3 for _, q in groups], float)
    weights /= weights.sum()
    manifest = {'schema': 'adaptive-vcm-training-v3', 'task': 'ar', 'seed': args.seed,
                'steps': args.steps, 'width': args.width, 'lr': args.lr, 'device': device,
                'train_ids': [r['id'] for r in plan], 'train_ids_sha256': fingerprint([r['id'] for r in plan]),
                'config': cfg, 'code': code_manifest(), 'profiles': [list(p) for p in model.profiles],
                'qp_sampling': {f'{c}/{q}': float(p) for (c, q), p in zip(groups, weights)},
                'objective': 'weighted cross_entropy to minimum actual_bytes feasible TRAIN profile; identity is a valid label',
                'identity_target_weight': identity_weight,
                'source': 'fresh initialization; final-LAST only; no DEV/TEST model selection',
                'limitations': 'Learns filter-profile selection, not filter coefficients. Teacher feasibility does not guarantee held-out evaluator accuracy.'}
    write_json(args.out / 'training_manifest.json', manifest)
    order = []
    for step in range(args.steps):
        if step % len(plan) == 0:
            order = rng.permutation(len(plan)).tolist()
        item = plan[order[step % len(plan)]]
        group = groups[step] if step < len(groups) else groups[int(rng.choice(len(groups), p=weights))]
        codec_name, qp = group
        clip = read_video(item['path'], cfg['frames'], cfg['ar_size'], cfg['temporal_stride'])
        source_predictions = [t.probabilities(clip) for t in teachers]
        semantic = np.maximum.reduce([normalize_map(t.saliency(clip)) for t in teachers])
        mask = semantic_protection(semantic)
        x = torch.from_numpy(clip.copy()).to(device).float().permute(3, 0, 1, 2)[None] / 255
        protected = torch.from_numpy(mask.copy()).to(x)[None, None, None].expand(1, 1, len(clip), *clip.shape[1:3])
        q, c = x.new_tensor([qp]), x.new_tensor([int(codec_name == 'h265')])
        logits = model.profile_logits(x, q, c, protected)
        predicted = int(logits.detach().argmax(1))
        codec = StandardCodec(codec_name, qp, cfg['preset'], cfg['fps'])
        anchor = codec.roundtrip(clip)
        anchor_predictions = [t.probabilities(anchor.decoded) for t in teachers]
        measurements, alphas, changes = [], [], []
        with torch.no_grad():
            bank = model.filter_bank(x, q)
            for index, (name, *_) in enumerate(model.profiles):
                if index == 0:
                    edited, stream, predictions, alpha = clip, anchor, anchor_predictions, 0.
                else:
                    output, aux = model.render_profile(x, q, c, protected, [index], bank=bank, return_aux=True)
                    edited = pixels(output)
                    stream = codec.roundtrip(edited)
                    predictions = (anchor_predictions if stream.data == anchor.data else
                                   [t.probabilities(stream.decoded) for t in teachers])
                    alpha = float(aux['alpha'].mean())
                distances, decisions = relative_guard('ar', source_predictions, anchor_predictions, predictions, cfg)
                measurements.append({'profile': name, 'coded_bytes': stream.coded_bytes,
                                     'distances': [float(d) for d in distances], 'decisions': list(decisions),
                                     'identity_stream': stream.data == anchor.data})
                alphas.append(alpha)
                changes.append(float(np.any(edited != clip, axis=-1).mean()))
        target = feasible_profile_target(measurements, slack=cfg['ar_kl_slack'], min_savings=cfg['min_savings'])
        # Missing a feasible edit loses bytes; an unsafe edit is rejected by the
        # encoder guard. Downweight the frequent identity label without removing it.
        saving = 1 - measurements[target]['coded_bytes'] / anchor.coded_bytes
        weight = identity_weight if target == 0 else 1 + min(1., 2 * saving)
        loss = F.cross_entropy(logits, torch.tensor([target], device=device)) * weight
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        chosen = measurements[predicted]
        row = {'step': step + 1, 'source_id': item['id'], 'codec': codec_name, 'qp': qp,
               'loss': float(loss.detach()), 'gradient_norm': float(norm),
               'predicted_profile': chosen['profile'], 'target_profile': measurements[target]['profile'],
               'target_rate_saving_pct': 100 * saving,
               'actual_anchor_bpp': reference_bpp(anchor.coded_bytes, clip.shape),
               'actual_preprocessed_bpp': reference_bpp(chosen['coded_bytes'], clip.shape),
               'same_qp_overhead_pct': (chosen['coded_bytes'] / anchor.coded_bytes - 1) * 100,
               'alpha_mean': alphas[predicted], 'changed_pixel_fraction': changes[predicted],
               'center_equals_identity_stream': chosen['identity_stream'], 'profiles': measurements}
        with (args.out / 'train.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
            print(json.dumps(row, allow_nan=False), flush=True)
    torch.save({'schema': model.schema, 'model': model.state_dict(), 'width': args.width, 'task': 'ar',
                'steps': args.steps, 'seed': args.seed, 'train_ids_sha256': manifest['train_ids_sha256'],
                'training_config': cfg, 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest()},
               args.out / 'preprocessor_last.pth')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['ar'], default='ar')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/v24_screen.json')
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--width', type=int, default=24)
    parser.add_argument('--seed', type=int, default=302001)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', type=Path, required=True)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
