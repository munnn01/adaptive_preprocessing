"""Frozen four-QP protocol, source allocation and portable code provenance."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from . import SCHEMA


QPS = (30, 35, 40, 45)
CODECS = ("h264", "h265")
FIT_CAL_HASH_PREFIX = "adaptive-v31-fit-cal"
SOURCE_COUNTS = {
    "ar": {"fit": 96, "cal": 32, "tune": 128, "dev": 128},
    "od": {"fit": 75, "cal": 25, "tune": 100, "dev": 100},
}
_LOCKED_CONFIG = {
    "schema": 1, "seed": 303101, "qps": list(QPS), "codecs": list(CODECS),
    "preset": "medium", "frames": 16, "temporal_stride": 2, "ar_size": 128, "od_size": 320,
    "ar_teachers": ["r3d_18", "mc3_18"], "ar_evaluators": ["r2plus1d_18", "r3d_18"],
    "od_teacher": "mobilenet", "od_evaluator": "resnet50", "ar_kl_slack": 0.1,
    "ar_confidence": 0.6, "od_distance_slack": 0.03, "od_score_threshold": 0.25,
    "min_savings": 0.01, "bootstrap_draws": 2000, "ar_require_anchor_decision": True,
    "ar_guard_rule": "anchor_relative_v2", "target_bd_rate_pct": -10,
    "ar_candidates": ["identity", "area112", "area96", "area112_up", "blur20", "blur40",
                      "protected_mild", "protected_strong", "protected_temporal"],
    "od_candidates": ["identity", "background2", "background4", "background8", "background12"],
    "epochs": 8, "width": 64, "proposal_k": 3,
    "optimizer": {"name": "Adam", "lr": 0.001, "batch_size": 32,
                  "early_stopping": False, "scheduler": None},
}


def _json_value(value) -> None:
    """Reject key coercion and values outside the JSON data model."""
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("canonical JSON object keys must be strings")
        for child in value.values():
            _json_value(child)
    elif isinstance(value, list):
        for child in value:
            _json_value(child)
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("canonical JSON numbers must be finite")
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("value is not canonical JSON")


def canonical_hash(value: dict | list) -> str:
    """SHA256 of sorted, compact UTF-8 JSON; list order remains significant."""
    if not isinstance(value, (dict, list)):
        raise ValueError("canonical hash requires a JSON object or array")
    _json_value(value)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_config(cfg: dict) -> dict:
    """Reject drift in the preregistered protocol and return an independent copy."""
    if not isinstance(cfg, dict):
        raise ValueError("V31 configuration must be an object")
    arm = cfg.get("v31_arm")
    if arm not in ("a", "b", "c") or cfg.get("experiment") != f"v31-{arm}":
        raise ValueError("V31 experiment/arm mismatch")
    if any(key in cfg for key in ("v29_variant", "v30_variant")):
        raise ValueError("historical variants are incompatible with V31")
    _json_value(cfg)
    for key, expected in _LOCKED_CONFIG.items():
        # JSON comparison distinguishes booleans/integers and QP floats/integers.
        actual = cfg.get(key)
        if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
            raise ValueError(f"V31 frozen configuration changed: {key}")
    return copy.deepcopy(cfg)


def _source_ids(records: list[dict]) -> list[str]:
    if not isinstance(records, list):
        raise ValueError("source plan must be a list")
    ids = []
    for record in records:
        identifier = record.get("id") if isinstance(record, dict) else None
        if type(identifier) is not str or not identifier or identifier != identifier.strip():
            raise ValueError("source IDs must be nonempty canonical strings")
        ids.append(identifier)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source IDs in plan")
    return ids


def allocate_sources(train_plan: list[dict], dev_plan: list[dict], task: str, seed: int) -> dict[str, list[dict]]:
    """Allocate already-ordered legacy TRAIN/DEV plans without consulting labels.

    The lowest seeded SHA256 values in the first TRAIN pool form FIT. CAL is
    the remaining pool. Every output retains its input plan's source order.
    """
    if task not in SOURCE_COUNTS or type(seed) is not int or seed < 0:
        raise ValueError("invalid V31 source task/seed")
    train_ids, dev_ids = _source_ids(train_plan), _source_ids(dev_plan)
    overlap = set(train_ids) & set(dev_ids)
    if overlap:
        raise ValueError(f"duplicate source IDs across TRAIN/DEV: {sorted(overlap)}")
    if task == "od":
        for record in train_plan + dev_plan:
            image_id = record.get("image_id")
            if type(image_id) is not int or image_id < 0 or str(image_id) != record["id"]:
                raise ValueError("OD canonical source ID must match its integer COCO image_id")
    counts = SOURCE_COUNTS[task]
    pool_size = counts["fit"] + counts["cal"]
    if len(train_plan) < pool_size + counts["tune"] or len(dev_plan) < counts["dev"]:
        raise ValueError(f"insufficient {task.upper()} TRAIN/DEV sources for frozen V31 budget")
    pool = train_plan[:pool_size]
    ranked_ids = sorted(train_ids[:pool_size], key=lambda identifier: (
        hashlib.sha256(f"{FIT_CAL_HASH_PREFIX}:{seed}:{identifier}".encode("utf-8")).hexdigest(),
        identifier,
    ))
    fit_ids = set(ranked_ids[:counts["fit"]])
    return copy.deepcopy({
        "fit": [record for record in pool if record["id"] in fit_ids],
        "cal": [record for record in pool if record["id"] not in fit_ids],
        "tune": train_plan[pool_size:pool_size + counts["tune"]],
        "dev": dev_plan[:counts["dev"]],
    })


def validate_partitions(plans: dict, pixel_hashes: dict) -> None:
    """Fail on overlapping identities or pixels, including TEST when supplied.

    Pixel evidence maps canonical source ID to the SHA256 of source pixels.
    Additional evidence may be present, but every planned source needs a hash.
    This validator checks membership/evidence; allocation locks cardinalities.
    """
    if not isinstance(plans, dict) or not plans or set(plans) - {"fit", "cal", "tune", "dev", "test"}:
        raise ValueError("unknown or empty V31 partitions")
    if not isinstance(pixel_hashes, dict):
        raise ValueError("pixel evidence must map source IDs to SHA256 hashes")
    seen_ids, seen_pixels = {}, {}
    for split, records in plans.items():
        for identifier in _source_ids(records):
            if identifier in seen_ids:
                raise ValueError(f"duplicate source ID {identifier!r}: {seen_ids[identifier]}/{split}")
            seen_ids[identifier] = split
    for identifier, split in seen_ids.items():
        digest = pixel_hashes.get(identifier)
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"missing or invalid pixel SHA256 for {split}/{identifier}")
        if digest in seen_pixels:
            previous_id = seen_pixels[digest]
            raise ValueError(f"duplicate source pixel content: {previous_id!r}/{identifier!r}; SHA256={digest}")
        seen_pixels[digest] = identifier


def expected_conditions(source_ids: list[str]) -> set[tuple[str, str, int]]:
    """Complete source/codec/QP grid; duplicate or noncanonical IDs fail."""
    if not isinstance(source_ids, list):
        raise ValueError("source IDs must be a list")
    ids = _source_ids([{"id": identifier} for identifier in source_ids])
    return {(identifier, codec, qp) for identifier in ids for codec in CODECS for qp in QPS}


def code_manifest_v31(root: Path) -> dict:
    """Hash recursive package code, frozen V31 configs and V31 runner scripts.

    LF normalization preserves provenance across Windows/Linux checkouts. A
    full immutable Git HEAD is required; file hashes also expose local drift.
    """
    root = Path(root).resolve()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise ValueError("V31 provenance requires a full Git commit") from error
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("V31 provenance requires a full Git SHA")
    configs = [root / "configs" / f"v31_{arm}.json" for arm in "abc"]
    package_files = list((root / "adaptive_vcm").rglob("*.py"))
    if not package_files or any(not path.is_file() for path in configs):
        raise ValueError("V31 provenance requires package code and all three frozen configs")
    files = package_files + configs + list((root / "scripts").glob("*v31*.py"))
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for path in sorted(files)}
    manifest = {"schema": SCHEMA, "commit": commit, "files_sha256": hashes}
    return {**manifest, "manifest_hash": canonical_hash(manifest)}
