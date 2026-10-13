"""Prepare and submit V34 OD benchmark on COCO 2017 to Kaggle GPU."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]


def pool_environment(path: Path, account: str):
    pool = json.loads(path.read_text(encoding="utf-8-sig"))
    value = pool[account]
    environment = dict(os.environ)
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        environment.pop(name, None)
    if isinstance(value, str):
        token = value
        environment["KAGGLE_API_TOKEN"] = token
    elif isinstance(value, dict) and value.get("key"):
        token = value["key"]
        environment["KAGGLE_USERNAME"] = value.get("username", account)
        environment["KAGGLE_KEY"] = token
    else:
        raise ValueError("unsupported pool credential format")
    return environment, token


def kaggle_call(arguments, pool, account):
    environment, secret = pool_environment(pool, account)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    command = [sys.executable, "-m", "kaggle", *arguments]
    result = subprocess.run(command, env=environment, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=300)
    safe_output = (result.stdout + result.stderr).replace(secret, "[REDACTED]")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(safe_output, end="" if safe_output.endswith("\n") else "\n")
    if result.returncode:
        raise RuntimeError(f"Kaggle command failed (exit {result.returncode}); no success inferred")
    return safe_output


def get_precomputed_b64(cache_dir: Path) -> str:
    if not cache_dir.is_dir() or not (cache_dir / "measurements.json").is_file():
        return ""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(cache_dir.iterdir()):
            if p.is_file():
                tar.add(p, arcname=p.name)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_notebook_script(commit: str, slug: str, runner_code: str, precomputed_b64: str) -> str:
    escaped_runner = repr(runner_code)
    escaped_cache = repr(precomputed_b64)
    return f'''import base64, hashlib, io, json, os, pathlib, subprocess, sys, tarfile, time, shutil

WORK = pathlib.Path('/kaggle/working')
INPUT = pathlib.Path('/kaggle/input')
REPO = WORK / 'adaptive_preprocessing'
SLUG = {repr(slug)}
OUT = WORK / 'outputs' / SLUG
OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONPATH=str(REPO))

def cmd(argv, cwd=None):
    print(">> Executing:", ' '.join(argv), flush=True)
    subprocess.run(argv, check=True, cwd=cwd or str(WORK))

print("=== V34 OD Benchmark on COCO 2017 ===", flush=True)
start_time = time.time()

# 1. Clone new upstream repo
cmd(['git', 'clone', '-q', 'https://github.com/munnn01/adaptive_preprocessing.git', str(REPO)])
cmd(['git', '-C', str(REPO), 'checkout', '-q', {repr(commit)}])
print("Cloned munnn01/adaptive_preprocessing at commit", {repr(commit)}, flush=True)

# 2. Install dependencies
cmd([sys.executable, '-m', 'pip', 'install', '-q', 'pycocotools'])

# 3. Write runner script
(REPO / 'scripts').mkdir(parents=True, exist_ok=True)
runner_path = REPO / 'scripts' / 'run_v34_od.py'
runner_path.write_text({escaped_runner}, encoding='utf-8')
print("Installed scripts/run_v34_od.py into repository", flush=True)

# 4. Unpack precomputed candidate cache if provided
cache_b64 = {escaped_cache}
if cache_b64:
    raw_tar = base64.b64decode(cache_b64)
    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode='r:gz') as tar:
        tar.extractall(OUT)
    print("Extracted precomputed candidate measurements and checkpoints into", str(OUT), flush=True)

# 5. Locate COCO dataset mounts
ann_matches = list(INPUT.rglob('instances_val2017.json'))
assert ann_matches, "instances_val2017.json not found in /kaggle/input"
ann_file = str(ann_matches[0])

img_matches = [p for p in INPUT.rglob('val2017') if p.is_dir()]
assert img_matches, "val2017 image directory not found in /kaggle/input"
img_dir = str(img_matches[0])

print(f"COCO Images: {{img_dir}}", flush=True)
print(f"COCO Annotations: {{ann_file}}", flush=True)

# 6. Execute OD Benchmark
cmd([
    sys.executable, str(runner_path),
    '--od-root', img_dir,
    '--annotations', ann_file,
    '--config', str(REPO / 'configs' / 'v34.json'),
    '--out', str(OUT),
    '--train-count', '48',
    '--cal-count', '16',
    '--dev-count', '16',
    '--test-count', '64',
    '--device', 'cuda'
], cwd=str(REPO))

# 7. Copy important reports to top-level working directory for immediate visibility
for name in ('OD_REPORT.md', 'test_summary.json', 'dev_summary.json', 'complete.json'):
    src = OUT / name
    if src.is_file():
        shutil.copyfile(src, WORK / name)

# 8. Create output tarball
tgz_path = WORK / f'{{SLUG}}.tgz'
with tarfile.open(tgz_path, 'w:gz') as tar:
    tar.add(OUT, arcname=f'outputs/{{SLUG}}')

elapsed = time.time() - start_time
print(f"=== Job complete in {{elapsed:.1f}} seconds ===", flush=True)
print(f"Archive written to {{tgz_path}}", flush=True)
'''


def prepare_and_submit(account: str, slug: str, pool_path: Path, commit: str = "22c756aa6e9dba33364fc8602b667a5cd0d9a476"):
    payload_dir = ROOT / "outputs" / "kaggle" / slug
    payload_dir.mkdir(parents=True, exist_ok=True)
    
    runner_code = (ROOT / "scripts" / "run_v34_od.py").read_text(encoding="utf-8")
    script_source = build_notebook_script(commit, slug, runner_code, "")
    
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": script_source.splitlines(True)
            }
        ]
    }
    
    metadata = {
        "id": f"{account}/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [
            "awsaf49/coco-2017-dataset",
            "qktttttttttt/kineticscleaned"
        ],
        "competition_sources": [],
        "kernel_sources": []
    }
    
    (payload_dir / "notebook.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    (payload_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Created payload at {payload_dir.resolve()}", flush=True)
    
    # Push kernel
    print(f"Pushing kernel to Kaggle account {account} with NvidiaTeslaT4 accelerator...", flush=True)
    push_out = kaggle_call(["kernels", "push", "--path", str(payload_dir), "--accelerator", "NvidiaTeslaT4"], pool_path, account)
    
    # Wait and check status
    time.sleep(3)
    status_out = kaggle_call(["kernels", "status", f"{account}/{slug}"], pool_path, account)
    print(f"Initial status: {status_out.strip()}", flush=True)
    return payload_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--account", default="tranthihongdieu", help="Kaggle account username")
    p.add_argument("--slug", default="v34-od-eval-coco", help="Kernel slug")
    p.add_argument("--pool", type=Path, default=Path(r"C:\Users\Wagur1\Downloads\pool (1).json"))
    p.add_argument("--commit", default="22c756aa6e9dba33364fc8602b667a5cd0d9a476")
    args = p.parse_args()
    prepare_and_submit(args.account, args.slug, args.pool, args.commit)


if __name__ == "__main__":
    main()
