"""Verify published V34 artifacts and optionally repeat the frozen paired merge."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT/'results/v34_r1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_path(relative):
    path = (ROOT/relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('manifest path outside repository')
    return path


def verified_bytes(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError('SHA-256 mismatch: '+str(path.relative_to(ROOT)))
    return raw


def verify(manifest):
    for relative, expected in manifest['runtime_sha256'].items():
        verified_bytes(checked_path(relative), expected)
    verified_bytes(RESULTS/'protocol/protocol.json', manifest['protocol_file_sha256'])
    verified_bytes(RESULTS/'independent_summary.json', manifest['independent_summary_sha256'])
    protocol = json.loads((RESULTS/'protocol/protocol.json').read_text())
    for name, key in [('annotations.csv','annotation_csv_sha256'),('exclusions.json','exclusions_sha256')]:
        verified_bytes(RESULTS/'protocol'/name, protocol[key])
    for seed, value in manifest['checkpoints'].items():
        verified_bytes(ROOT/f'checkpoints/v34_r1/controller_{seed}.pth', value['sha256'])
    for index, files in manifest['shard_original_sha256'].items():
        directory = RESULTS/f'shards/shard{index}'
        for name, expected in files.items():
            if name != 'test_rows.json':
                verified_bytes(directory/name, expected)
    for relative, value in manifest['compressed_files'].items():
        raw = verified_bytes(checked_path(relative), value['sha256'])
        unpacked = gzip.decompress(raw)
        if len(unpacked) != value['unpacked_bytes'] or sha(unpacked) != value['unpacked_sha256']:
            raise ValueError('unpacked paired rows mismatch')
    print('Verified exact runtime, protocol, annotations/exclusions, 3 checkpoints and 4 complete paired-row archives.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT/'output/reproduced_v34')
    args = parser.parse_args()
    manifest = json.loads((RESULTS/'publication_manifest.json').read_text())
    verify(manifest)
    if args.verify_only:
        return
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('reproduction requires an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(4):
        src = RESULTS/f'shards/shard{i}'
        dst = out/f'shards/shard{i}'
        dst.mkdir(parents=True)
        for name in manifest['shard_original_sha256'][str(i)]:
            if name == 'test_rows.json':
                raw = gzip.decompress((src/(name+'.gz')).read_bytes())
                (dst/name).write_bytes(raw)
            else:
                shutil.copyfile(src/name, dst/name)
        paths.append(dst)
    from adaptive_vcm.v34.holdout import merge
    actual = merge(paths, out/'analysis', protocol_path=RESULTS/'protocol/protocol.json')
    expected = json.loads((RESULTS/'independent_summary.json').read_text())
    if actual != expected:
        raise ValueError('recomputed summary differs from publication; inspect software versions and numeric precision')
    print('Reproduced 112,640 paired rows and all 12 × 5,000 bootstrap comparisons; summary matches exactly.', flush=True)


if __name__ == '__main__':
    main()
