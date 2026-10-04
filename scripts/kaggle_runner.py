"""Prepare/submit private free-GPU VCM jobs; secrets stay in subprocess env only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

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


def prepare(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit or ""):
        raise ValueError("an immutable full Git commit SHA is required")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{5,70}", args.slug) or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.account):
        raise ValueError("invalid Kaggle handle")
    if args.count < 2 or args.steps < 1 or args.bootstrap < 0:
        raise ValueError("invalid job budget")
    dataset = "qktttttttttt/kineticscleaned" if args.task == "ar" else "awsaf49/coco-2017-dataset"
    bash = f'''%%bash
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
REPO=/kaggle/working/adaptive_preprocessing
OUT=/kaggle/working/outputs/{args.slug}
mkdir -p "$OUT"
finish() {{
  result=$?
  trap - EXIT
  set +e
  cd /kaggle/working
  tar -czf {args.slug}.tgz outputs/{args.slug}
  echo "[exit] status=$result"
  exit "$result"
}}
trap finish EXIT
git clone -q https://github.com/munnn01/adaptive_preprocessing.git "$REPO"
git -C "$REPO" checkout -q {args.commit}
test "$(git -C "$REPO" rev-parse HEAD)" = {args.commit}
cd "$REPO"
python -m pip install -q pycocotools
command -v ffmpeg
python -c 'import torch; assert torch.cuda.is_available(), "Free GPU required"; print("GPU", torch.cuda.get_device_name())'
'''
    if args.task == "ar":
        bash += '''KIN_ROOT=""
for candidate in /kaggle/input/kineticscleaned /kaggle/input/datasets/qktttttttttt/kineticscleaned; do
  if [ -d "$candidate" ]; then KIN_ROOT="$candidate"; break; fi
done
test -n "$KIN_ROOT"
'''
        data_arguments = '--task ar --root "$KIN_ROOT"'
    else:
        bash += '''ANN=$(find /kaggle/input -name instances_val2017.json -print -quit)
IMAGES=$(find /kaggle/input -type d -name val2017 -print -quit)
test -n "$ANN" && test -n "$IMAGES"
'''
        data_arguments = '--task od --root "$IMAGES" --annotations "$ANN"'
    checkpoint = ""
    if args.mode == "learned":
        bash += f'python -m adaptive_vcm.train {data_arguments} --count 512 --steps {args.steps} --seed {args.seed} --out "$OUT/train"\n'
        checkpoint = ' --checkpoint "$OUT/train/preprocessor_last.pth"'
    bash += f'python -m adaptive_vcm.evaluate {data_arguments} --count {args.count} --split dev --codecs h264 h265 --bootstrap {args.bootstrap}{checkpoint} --out "$OUT/eval"\n'
    directory = args.directory or ROOT / "outputs/kaggle" / args.slug
    directory.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                            "source": bash.splitlines(keepends=True)}],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "nbformat": 4, "nbformat_minor": 5}
    metadata = {"id": f"{args.account}/{args.slug}", "title": args.slug, "code_file": "notebook.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True, "dataset_sources": [dataset],
                "kernel_sources": [], "competition_sources": [], "model_sources": []}
    (directory / "notebook.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (directory / "job.json").write_text(json.dumps({"commit": args.commit, "task": args.task, "mode": args.mode,
                                                   "steps": args.steps, "count": args.count, "bootstrap": args.bootstrap,
                                                   "seed": args.seed, "handle": metadata["id"], "private": True}, indent=2), encoding="utf-8")
    print(json.dumps({"prepared": str(directory), "handle": metadata["id"], "commit": args.commit}))
    return directory


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "submit", "status", "download", "list"])
    parser.add_argument("--pool", type=Path, default=Path("D:/STUDY/LAB/pool.json"))
    parser.add_argument("--account", required=True)
    parser.add_argument("--slug")
    parser.add_argument("--task", choices=["ar", "od"], default="ar")
    parser.add_argument("--mode", choices=["analytic", "learned"], default="learned")
    parser.add_argument("--commit")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=302001)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.action != "list" and not args.slug:
        parser.error("--slug is required")
    if args.action == "prepare":
        prepare(args)
    elif args.action == "submit":
        directory = args.directory or ROOT / "outputs/kaggle" / args.slug
        metadata = json.loads((directory / "kernel-metadata.json").read_text())
        if metadata["id"] != f"{args.account}/{args.slug}" or metadata["is_private"] is not True or metadata["enable_gpu"] is not True:
            raise ValueError("prepared payload/account mismatch")
        message = kaggle_call(["kernels", "push", "-p", str(directory), "--accelerator", "NvidiaTeslaT4"], args.pool, args.account)
        (directory / "submission.txt").write_text(message, encoding="utf-8")
    elif args.action == "list":
        kaggle_call(["kernels", "list", "--mine", "--page-size", "30"], args.pool, args.account)
    elif args.action == "status":
        kaggle_call(["kernels", "status", f"{args.account}/{args.slug}"], args.pool, args.account)
    else:
        directory = args.directory or ROOT / "outputs/downloads" / args.slug
        directory.mkdir(parents=True, exist_ok=True)
        kaggle_call(["kernels", "output", f"{args.account}/{args.slug}", "-p", str(directory)], args.pool, args.account)


if __name__ == "__main__":
    main()

