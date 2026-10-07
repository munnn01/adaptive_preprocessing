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
    if args.task=='both':
        return prepare_joint(args)
    if args.task not in ('ar','od'):
        raise ValueError('unsupported job task')
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
    recipe = getattr(args, "recipe", "v22")
    if recipe not in ("v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30"):
        raise ValueError("unsupported recipe")
    if recipe == "v24" and (args.task != "ar" or args.mode != "learned"):
        raise ValueError("V24 recipe is registered for learned AR only")
    if recipe in ('v25', 'v26', 'v27') and (args.task != 'ar' or args.mode != 'learned'):
        raise ValueError('ranking recipes are registered for learned AR only')
    measurements = getattr(args, 'measurements', 512)
    train_count = getattr(args, 'train_count', 512)
    if recipe in ('v25', 'v26', 'v27') and (measurements < 10 or train_count < (4 if recipe in ('v26', 'v27') else 2)):
        raise ValueError('ranking collection needs at least ten codec/QP groups')
    if recipe == 'v27' and measurements != train_count * 10:
        raise ValueError('V27 requires a complete TRAIN codec/QP grid with train_count*10 measurements')
    epochs=getattr(args,'epochs',4)
    width=getattr(args,'width',12)
    variant = getattr(args, 'variant', None)
    if recipe == 'v29' and variant not in ('a', 'b', 'c'):
        raise ValueError('V29 requires a declared a/b/c variant')
    if recipe == 'v30' and variant not in ('a', 'b', 'c'):
        raise ValueError('V30 requires a declared a/b/c variant')
    if recipe in ('v28', 'v29', 'v30'):
        if (args.mode!='learned' or type(train_count) is not int or train_count<2
                or measurements!=train_count*10 or type(epochs) is not int or epochs<1
                or type(width) is not int or width<4):
            if recipe == 'v30':
                raise ValueError('V30 requires a learned recipe, complete TRAIN codec/QP grid and valid epochs/width')
            raise ValueError('V28 requires a learned recipe, complete TRAIN codec/QP grid and valid epochs/width')
    config_name = f'{recipe}_{variant}' if recipe in ('v29', 'v30') else recipe
    config_option = f" --config configs/{config_name}_screen.json" if recipe != "v22" else ""
    trainer = {"v22": "adaptive_vcm.train", "v23": "adaptive_vcm.train_rateaware",
               "v24": "adaptive_vcm.train_profiles", 'v25': 'adaptive_vcm.train_ranking',
               'v26': 'adaptive_vcm.train_utility', 'v27': 'adaptive_vcm.train_portfolio',
               'v28': 'adaptive_vcm.train_motion', 'v29': 'adaptive_vcm.train_motion',
               'v30': 'adaptive_vcm.train_motion'}[recipe]
    if recipe == "v24":
        # Paired guard-only control: exact V23 final checkpoint, unchanged config,
        # current corrected guard. Never train from or select a model on DEV.
        bash += '''BASELINE=$(python scripts/find_v23_checkpoint.py /kaggle/input)
test -f "$BASELINE"
'''
        bash += f'python -m adaptive_vcm.evaluate {data_arguments} --config configs/v23_screen.json --checkpoint "$BASELINE" --count {args.count} --split dev --codecs h264 h265 --bootstrap {args.bootstrap} --ablate-learned --out "$OUT/guard_only"\n'
    if args.mode == "learned":
        count = train_count if recipe in ('v25', 'v26', 'v27', 'v28', 'v29', 'v30') else 512
        extras = f' --measurements {measurements}' if recipe in ('v25', 'v26', 'v27') else ''
        if recipe in ('v28', 'v29', 'v30'):
            extras=f' --epochs {epochs} --width {width}'
        steps_option = '' if recipe in ('v26', 'v27', 'v28', 'v29', 'v30') else f' --steps {args.steps}'
        bash += f'python -m {trainer} {data_arguments}{config_option} --count {count}{steps_option}{extras} --seed {args.seed} --out "$OUT/train"\n'
        checkpoint = ' --checkpoint "$OUT/train/preprocessor_last.pth"'
    ablation_option = " --ablate-learned" if recipe in ("v23", "v24", 'v25', 'v26', 'v27', 'v28', 'v29', 'v30') and args.mode == "learned" else ""
    bash += f'python -m adaptive_vcm.evaluate {data_arguments}{config_option} --count {args.count} --split dev --codecs h264 h265 --bootstrap {args.bootstrap}{checkpoint}{ablation_option} --out "$OUT/eval"\n'
    directory = args.directory or ROOT / "outputs/kaggle" / args.slug
    directory.mkdir(parents=True, exist_ok=True)
    # Stream the new recipe's subprocess output as it happens. %%bash buffers
    # the entire cell, which hid collection progress on the older long jobs.
    source = bash
    if recipe in ('v26', 'v27', 'v28', 'v29', 'v30'):
        command = bash.removeprefix('%%bash\n')
        source = ('import subprocess\n'
                  f'command = {command!r}\n'
                  'process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, '
                  'stderr=subprocess.STDOUT, text=True, bufsize=1)\n'
                  'for line in process.stdout:\n    print(line, end="", flush=True)\n'
                  'status = process.wait()\n'
                  'if status:\n    raise RuntimeError(f"VCM job failed with exit {status}")\n')
    notebook = {"cells": [{"cell_type": "code", "id": "run-" + args.slug[:48], "execution_count": None, "metadata": {}, "outputs": [],
                            "source": source.splitlines(keepends=True)}],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "nbformat": 4, "nbformat_minor": 5}
    metadata = {"id": f"{args.account}/{args.slug}", "title": args.slug, "code_file": "notebook.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True, "dataset_sources": [dataset],
                "kernel_sources": ["qktttttttttt/v23-rateaware-ar-s302001"] if recipe == "v24" else [],
                "competition_sources": [], "model_sources": []}
    (directory / "notebook.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (directory / "job.json").write_text(json.dumps({"commit": args.commit, "task": args.task, "mode": args.mode, "recipe": recipe,
                                                   "steps": None if recipe in ('v26', 'v27', 'v28', 'v29', 'v30') else args.steps, "count": args.count, "bootstrap": args.bootstrap,
                                                   "measurements": measurements if recipe in ('v25', 'v26', 'v27', 'v28', 'v29', 'v30') else None,
                                                   "train_count": train_count if recipe in ('v25', 'v26', 'v27', 'v28', 'v29', 'v30') else None,
                                                   "epochs": epochs if recipe in ('v28', 'v29', 'v30') else None,
                                                   "width": width if recipe in ('v28', 'v29', 'v30') else None,
                                                   "variant": variant if recipe in ('v29', 'v30') else None,
                                                   "seed": args.seed, "handle": metadata["id"], "private": True}, indent=2), encoding="utf-8")
    print(json.dumps({"prepared": str(directory), "handle": metadata["id"], "commit": args.commit}))
    return directory


def prepare_joint(args):
    """One GPU job executes independent AR/OD development screens sequentially."""
    if getattr(args,'recipe',None) not in ('v28', 'v29', 'v30') or args.mode!='learned':
        raise ValueError('joint jobs are registered for learned V28/V29/V30 only')
    directory=args.directory or ROOT/'outputs/kaggle'/args.slug
    notebooks=[]
    for task in ('ar','od'):
        values=vars(args).copy()
        values.update(task=task,slug=f'{args.slug}-{task}',directory=directory/'task_payloads'/task)
        task_directory=prepare(argparse.Namespace(**values))
        notebook=json.loads((task_directory/'notebook.ipynb').read_text(encoding='utf-8'))
        cell=notebook['cells'][0]
        source=''.join(cell['source']).replace('REPO=/kaggle/working/adaptive_preprocessing',
                                             f'REPO=/kaggle/working/adaptive_preprocessing_{task}')
        cell['source']=source.splitlines(keepends=True)
        notebooks.append(cell)
    payload=json.loads((directory/'task_payloads/ar/notebook.ipynb').read_text(encoding='utf-8'))
    payload['cells']=notebooks
    metadata=json.loads((directory/'task_payloads/ar/kernel-metadata.json').read_text(encoding='utf-8'))
    metadata.update(id=f'{args.account}/{args.slug}',title=args.slug,
                    dataset_sources=['qktttttttttt/kineticscleaned','awsaf49/coco-2017-dataset'])
    job=json.loads((directory/'task_payloads/ar/job.json').read_text(encoding='utf-8'))
    job.update(task='both',handle=metadata['id'],sequential_tasks=['ar','od'],
               measurements_per_task=job.pop('measurements'),
               scope='Sequential independent AR/OD TRAIN and DEV, shared GPU job; child payloads are unsubmitted templates')
    for filename,value in [('notebook.ipynb',payload),('kernel-metadata.json',metadata),('job.json',job)]:
        (directory/filename).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(dict(prepared=str(directory),handle=metadata['id'],commit=args.commit,task='both')))
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
    parser.add_argument("--task", choices=["ar", "od", "both"], default="ar")
    parser.add_argument("--mode", choices=["analytic", "learned"], default="learned")
    parser.add_argument("--recipe", choices=["v22", "v23", "v24", 'v25', 'v26', 'v27', 'v28', 'v29', 'v30'], default="v22")
    parser.add_argument('--variant', choices=['a', 'b', 'c'])
    parser.add_argument("--commit")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument('--measurements', type=int)
    parser.add_argument('--train-count', type=int)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--width',type=int,default=12)
    parser.add_argument("--count", type=int)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    defaults = ({'train_count': 32, 'count': 32, 'epochs': 8,
                 'bootstrap': 2000, 'seed': 302901} if args.recipe in ('v29', 'v30') else
                {'train_count': 512, 'count': 128, 'epochs': 4,
                 'bootstrap': 200, 'seed': 302001})
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.measurements is None:
        args.measurements = args.train_count * 10 if args.recipe in ('v29', 'v30') else 512
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

