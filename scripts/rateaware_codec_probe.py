"""Actual-byte initialization smoke; synthetic inputs, no task-accuracy claim."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from adaptive_vcm.codec import StandardCodec, reference_bpp
from adaptive_vcm.rateaware import RateAwarePreprocessor


def run(out):
    torch.manual_seed(23)
    torch.set_num_threads(2)
    rng = np.random.default_rng(23)
    rows = []
    for task in ('ar', 'od'):
        t, size = (16, 128) if task == 'ar' else (1, 320)
        background = rng.integers(0, 256, (size, size, 3), np.uint8)
        clip = np.repeat(background[None], t, 0)
        # Static textured background; small source variation exercises reuse.
        if t > 1:
            clip = np.clip(clip.astype(np.int16) + rng.integers(-2, 3, clip.shape), 0, 255).astype(np.uint8)
        start, stop = size * 3 // 8, size * 5 // 8
        clip[:, start:stop, start:stop] = [240, 32, 16]
        x = torch.from_numpy(clip.copy()).float().permute(3, 0, 1, 2)[None] / 255
        protection = torch.zeros(1, 1, t, size, size)
        protection[..., start-8:stop+8, start-8:stop+8] = 1
        model = RateAwarePreprocessor(24, task).eval()
        for codec in ('h264', 'h265'):
            for qp in (40, 45, 50):
                with torch.no_grad():
                    output, aux = model(x, torch.tensor([qp]), torch.tensor([int(codec == 'h265')]),
                                        protection, return_aux=True)
                pixels = output[0].permute(1, 2, 3, 0).mul(255).round().clamp(0, 255).byte().numpy()
                exact = bool(np.array_equal(clip[:, start-8:stop+8, start-8:stop+8], pixels[:, start-8:stop+8, start-8:stop+8]))
                assert exact
                encoder = StandardCodec(codec, qp)
                anchor, candidate = encoder.roundtrip(clip), encoder.roundtrip(pixels)
                row = {'task': task, 'codec': codec, 'qp': qp,
                       'anchor_bytes': anchor.coded_bytes, 'learned_init_bytes': candidate.coded_bytes,
                       'rate_change_pct': (candidate.coded_bytes / anchor.coded_bytes - 1) * 100,
                       'anchor_bpp': reference_bpp(anchor.coded_bytes, clip.shape),
                       'candidate_bpp': reference_bpp(candidate.coded_bytes, clip.shape),
                       'alpha_mean': float(aux['alpha'].mean()),
                       'changed_pixel_fraction': float(np.any(pixels != clip, axis=-1).mean()),
                       'protected_pixels_exact': exact,
                       'anchor_sha256': hashlib.sha256(anchor.data).hexdigest(),
                       'candidate_sha256': hashlib.sha256(candidate.data).hexdigest()}
                rows.append(row)
                print(json.dumps({k: row[k] for k in ('task', 'codec', 'qp', 'rate_change_pct')}), flush=True)
    result = {'scope': 'Synthetic static textured-background smoke at AR16x128x128 and OD1x320x320. INITIAL policy, not a trained task experiment.',
              'trained': False, 'quality_evaluated': False, 'target_confirmed': False, 'rows': rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    run(parser.parse_args().out)
