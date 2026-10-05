"""Actual encoder byte probe; synthetic texture, no AR teacher or task score."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from adaptive_vcm.codec import StandardCodec, reference_bpp
from adaptive_vcm.profiles import ProfilePreprocessor
from adaptive_vcm.task_bank import build_task_bank


def source(block_size):
    rng = np.random.default_rng(25)
    # Pixel texture and resolvable two-pixel texture expose the header floor
    # and a detail-suppression opportunity under identical codec settings.
    frame = rng.integers(40, 210, (128//block_size, 128//block_size, 3), dtype=np.uint8)
    frame = np.repeat(np.repeat(frame, block_size, 0), block_size, 1)
    clip = np.stack([np.clip(frame.astype(float) + rng.normal(0, 2, frame.shape), 0, 255).round().astype(np.uint8)
                     for _ in range(16)])
    clip[:, 48:80, 48:80] = [232, 48, 16]
    mask = np.zeros((128, 128), np.float32)
    mask[40:88, 40:88] = 1
    return clip, mask


def run(out):
    torch.set_num_threads(2)
    old = ProfilePreprocessor(8).eval()
    rows = []
    for block in (1, 2):
        clip, mask = source(block)
        x = torch.from_numpy(clip.copy()).float().permute(3, 0, 1, 2)[None] / 255
        m = torch.from_numpy(mask)[None, None, None].expand(1, 1, len(clip), 128, 128)
        for codec in ('h264', 'h265'):
            for qp in (40, 45, 50):
                encoder = StandardCodec(codec, qp)
                raw = encoder.roundtrip(clip)
                record = dict(source=f'synthetic_block{block}', codec=codec, qp=qp,
                              anchor_bytes=raw.coded_bytes, anchor_bpp=reference_bpp(raw.coded_bytes, clip.shape),
                              task_teacher_feasibility=None, top1=None, bd_rate=None)
                for bank_name, bank in [('task_bank', build_task_bank(clip, mask, qp))]:
                    actions = []
                    for candidate in bank[1:]:
                        encoded = encoder.roundtrip(candidate.clip)
                        decoded = np.stack([cv2.resize(f, (128, 128), interpolation=cv2.INTER_LINEAR)
                                            for f in encoded.decoded])
                        # Report a pixel sanity distance only, not a frozen-task guard.
                        actions.append(dict(name=candidate.name, bytes=encoded.coded_bytes,
                                            rate_change_pct=100 * (encoded.coded_bytes / raw.coded_bytes - 1),
                                            decoded_mae=float(np.abs(decoded.astype(float)-clip.astype(float)).mean()),
                                            source_edit_mae=float(np.abs(np.stack([cv2.resize(f,(128,128)) for f in candidate.clip]).astype(float)-clip.astype(float)).mean())))
                    record[bank_name] = actions
                with torch.no_grad():
                    spectral = old.filter_bank(x, torch.tensor([qp]))
                    legacy = []
                    for index in range(1, len(old.profiles)):
                        rendered = old.render_profile(x, torch.tensor([qp]), torch.tensor([int(codec == 'h265')]), m, [index], bank=spectral)
                        pixels = rendered[0].permute(1,2,3,0).mul(255).round().byte().numpy()
                        encoded = encoder.roundtrip(pixels)
                        legacy.append(dict(name=old.profiles[index][0], bytes=encoded.coded_bytes,
                                           rate_change_pct=100*(encoded.coded_bytes/raw.coded_bytes-1),
                                           source_edit_mae=float(np.abs(pixels.astype(float)-clip.astype(float)).mean())))
                    record['v24_bank'] = legacy
                for name in ('task_bank', 'v24_bank'):
                    a = record[name]
                    record[name + '_byte_saving_actions'] = sum(r['bytes'] <= .99 * raw.coded_bytes for r in a)
                    # The mild-source-edit slice illustrates extra useful capacity,
                    # but is NOT a claim of teacher feasibility or AR accuracy.
                    mild = [r for r in a if r['source_edit_mae'] <= 10]
                    record[name + '_mild_byte_saving_actions'] = sum(r['bytes'] <= .99 * raw.coded_bytes for r in mild)
                    record[name + '_best_mild_rate_change_pct'] = min([0.] + [r['rate_change_pct'] for r in mild])
                rows.append(record)
                print(json.dumps({k:v for k,v in record.items() if not isinstance(v,list)}), flush=True)
    result = dict(scope='Synthetic 16x128x128 static color ROI with one-/two-pixel random texture and small temporal noise. Actual H.264/H.265 medium QP40/45/50, 25fps. No action teachers or evaluator.',
                  dev_score=None, task_teacher_feasibility=None, top1=None, bd_rate=None, target_confirmed=False,
                  mild_definition='source-space RGB MAE <=10 after resizing, solely a diagnostic, never used by selection', rows=rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
