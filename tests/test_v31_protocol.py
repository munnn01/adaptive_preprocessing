"""Frozen protocol behavior; source memberships are independent of task labels."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "adaptive_vcm" / "v31" / "protocol.py"


@pytest.fixture
def protocol():
    if not PROTOCOL.is_file():
        class MissingProtocol:
            def __getattr__(self, name):
                assert PROTOCOL.is_file(), "V31 protocol feature has not been implemented"
        return MissingProtocol()
    from adaptive_vcm.v31 import protocol
    return protocol


def source_plans(task):
    # Literal cardinalities derive from the approved source budget, not the API.
    if task == "ar":
        return ([{"id": f"ar-{i:03d}", "label": i % 7} for i in range(270)],
                [{"id": f"dev-ar-{i:03d}", "label": i % 7} for i in range(140)])
    return ([{"id": str(i), "image_id": i, "label": i % 7} for i in range(1, 211)],
            [{"id": str(i), "image_id": i, "label": i % 7} for i in range(1001, 1111)])


def config(arm="a"):
    path = ROOT / "configs" / f"v31_{arm}.json"
    assert path.is_file(), "V31 frozen configuration has not been implemented"
    return json.loads(path.read_text(encoding="utf-8"))


def test_eight_conditions_are_complete_without_qp50(protocol):
    assert protocol.expected_conditions(["clip-a"]) == {
        ("clip-a", "h264", 30), ("clip-a", "h264", 35),
        ("clip-a", "h264", 40), ("clip-a", "h264", 45),
        ("clip-a", "h265", 30), ("clip-a", "h265", 35),
        ("clip-a", "h265", 40), ("clip-a", "h265", 45),
    }
    assert len(protocol.expected_conditions(["clip-a", "clip-b"])) == 16
    for bad in (["clip-a", "clip-a"], [1], [""]):
        with pytest.raises(ValueError):
            protocol.expected_conditions(bad)


@pytest.mark.parametrize("task,counts,condition_counts", [
    ("ar", {"fit": 96, "cal": 32, "tune": 128, "dev": 128},
     {"fit": 768, "cal": 256, "tune": 1024, "dev": 1024}),
    ("od", {"fit": 75, "cal": 25, "tune": 100, "dev": 100},
     {"fit": 600, "cal": 200, "tune": 800, "dev": 800}),
])
def test_source_budgets_and_grid_counts(protocol, task, counts, condition_counts):
    train, dev = source_plans(task)
    before = copy.deepcopy((train, dev))
    plans = protocol.allocate_sources(train, dev, task, 303101)
    assert {key: len(rows) for key, rows in plans.items()} == counts
    assert {key: len(protocol.expected_conditions([r["id"] for r in rows]))
            for key, rows in plans.items()} == condition_counts
    pool_size = 128 if task == "ar" else 100
    assert {r["id"] for r in plans["fit"] + plans["cal"]} == {
        r["id"] for r in train[:pool_size]}
    assert plans["tune"] == train[pool_size:2 * pool_size]
    assert plans["dev"] == dev[:pool_size]
    if task == "od":
        assert plans["tune"][0]["id"] == "101"
        assert plans["dev"][0]["id"] == "1001"
        assert all(type(r["image_id"]) is int for rows in plans.values() for r in rows)
    assert (train, dev) == before


def test_ar_memberships_seed_and_input_order_are_frozen_without_label_use(protocol):
    train, dev = source_plans("ar")
    plans = protocol.allocate_sources(train, dev, "ar", 303101)
    # Fixed source IDs from the preregistered seeded split, independent of labels.
    assert [r["id"] for r in plans["cal"]] == [
        "ar-001", "ar-002", "ar-004", "ar-011", "ar-012", "ar-013", "ar-016", "ar-019",
        "ar-022", "ar-026", "ar-034", "ar-035", "ar-037", "ar-038", "ar-040", "ar-041",
        "ar-048", "ar-054", "ar-066", "ar-075", "ar-077", "ar-082", "ar-083", "ar-085",
        "ar-086", "ar-092", "ar-096", "ar-101", "ar-107", "ar-108", "ar-120", "ar-122",
    ]
    changed = copy.deepcopy(train)
    for record in changed:
        record["label"] = -123
        record["prediction"] = {"class": "irrelevant"}
    altered = protocol.allocate_sources(changed, dev, "ar", 303101)
    assert {k: [r["id"] for r in rows] for k, rows in altered.items()} == {
        k: [r["id"] for r in rows] for k, rows in plans.items()}
    other_seed = protocol.allocate_sources(train, dev, "ar", 303102)
    assert [r["id"] for r in other_seed["cal"]] != [r["id"] for r in plans["cal"]]
    assert other_seed["tune"] == plans["tune"]
    for key in ("fit", "cal"):
        indices = [next(i for i, r in enumerate(train) if r["id"] == row["id"])
                   for row in plans[key]]
        assert indices == sorted(indices)
    altered["fit"][0]["label"] = 800
    assert changed[0]["label"] == -123


def test_od_allocation_ignores_labels_predictions_and_annotation_order(protocol):
    train, dev = source_plans("od")
    original = protocol.allocate_sources(train, dev, "od", 303101)
    for record in train + dev:
        record["label"] = 900
        record["annotations"] = [{"category_id": 900}, {"category_id": 1}]
        record["prediction"] = [0.99, 0.01]
    altered = protocol.allocate_sources(train, dev, "od", 303101)
    assert {k: [r["id"] for r in rows] for k, rows in original.items()} == {
        k: [r["id"] for r in rows] for k, rows in altered.items()}


@pytest.mark.parametrize("mutation", ["short_train", "short_dev", "duplicate_train",
                                          "overlap_dev", "noncanonical_id", "od_id_mismatch"])
def test_tampered_input_plans_fail_closed(protocol, mutation):
    train, dev = source_plans("od")
    if mutation == "short_train":
        train = train[:199]
    elif mutation == "short_dev":
        dev = dev[:99]
    elif mutation == "duplicate_train":
        train[-1] = copy.deepcopy(train[0])
    elif mutation == "overlap_dev":
        dev[-1] = copy.deepcopy(train[0])
    elif mutation == "noncanonical_id":
        train[0]["id"] = 1
    elif mutation == "od_id_mismatch":
        train[0]["image_id"] = 2
    with pytest.raises(ValueError):
        protocol.allocate_sources(train, dev, "od", 303101)


def test_unknown_task_and_invalid_seed_are_rejected(protocol):
    train, dev = source_plans("ar")
    with pytest.raises(ValueError):
        protocol.allocate_sources(train, dev, "video", 303101)
    with pytest.raises(ValueError):
        protocol.allocate_sources(train, dev, "ar", True)


def test_partitions_reject_duplicate_ids_and_duplicate_pixels_including_test(protocol):
    plans = {"fit": [{"id": "a"}], "cal": [{"id": "b"}],
             "tune": [{"id": "c"}], "dev": [{"id": "d"}], "test": [{"id": "e"}]}
    hashes = {key: hashlib.sha256(key.encode()).hexdigest() for key in "abcde"}
    assert protocol.validate_partitions(plans, hashes) is None
    for first, second in (("fit", "cal"), ("fit", "tune"), ("fit", "dev"), ("fit", "test")):
        altered = copy.deepcopy(plans)
        altered[second][0]["id"] = altered[first][0]["id"]
        with pytest.raises(ValueError, match="duplicate.*ID"):
            protocol.validate_partitions(altered, hashes)
        content = dict(hashes, **{plans[second][0]["id"]: hashes[plans[first][0]["id"]]})
        with pytest.raises(ValueError, match="duplicate.*(content|pixel)"):
            protocol.validate_partitions(plans, content)
    plans["fit"].append({"id": "a"})
    with pytest.raises(ValueError, match="duplicate.*ID"):
        protocol.validate_partitions(plans, hashes)


@pytest.mark.parametrize("hashes", [{}, {"a": "not-a-digest"}, {"a": None}])
def test_missing_or_malformed_pixel_evidence_rejected(protocol, hashes):
    with pytest.raises(ValueError):
        protocol.validate_partitions({"fit": [{"id": "a"}]}, hashes)


@pytest.mark.parametrize("arm", ["a", "b", "c"])
def test_frozen_configs_validate_without_mutating_input(protocol, arm):
    cfg = config(arm)
    frozen = copy.deepcopy(cfg)
    result = protocol.validate_config(cfg)
    assert result == cfg == frozen
    assert result is not cfg
    assert cfg["experiment"] == f"v31-{arm}" and cfg["v31_arm"] == arm
    assert cfg["qps"] == [30, 35, 40, 45]
    assert cfg["seed"] == 303101 and cfg["bootstrap_draws"] == 2000
    assert cfg["epochs"] == 8 and cfg["width"] == 64 and cfg["proposal_k"] == 3
    assert cfg["optimizer"] == {"name": "Adam", "lr": 0.001, "batch_size": 32,
                                "early_stopping": False, "scheduler": None}
    old = json.loads((ROOT / "configs/v30_a_screen.json").read_text(encoding="utf-8"))
    for key in ("ar_candidates", "od_candidates", "ar_teachers", "ar_evaluators", "od_teacher",
                "od_evaluator", "ar_kl_slack", "ar_confidence", "od_distance_slack",
                "od_score_threshold", "min_savings", "ar_require_anchor_decision", "ar_guard_rule"):
        assert cfg[key] == old[key]


@pytest.mark.parametrize("key,value", [
    ("qps", [30, 35, 40, 45, 50]), ("qps", [30, 35, 40]),
    ("qps", [30, 35, 40, 40]), ("qps", [30.0, 35, 40, 45]),
    ("codecs", ["h264"]), ("schema", "adaptive-vcm-conditional-v9"),
    ("experiment", "v30-a"), ("v31_arm", "c"), ("seed", 302901),
    ("frames", 8), ("width", 128), ("epochs", 9), ("proposal_k", 4),
    ("bootstrap_draws", 100), ("ar_kl_slack", 0.2), ("od_distance_slack", 0.1),
    ("ar_candidates", ["identity"]), ("ar_teachers", ["r2plus1d_18"]),
    ("od_evaluator", "mobilenet"),
])
def test_config_drift_is_rejected(protocol, key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError):
        protocol.validate_config(cfg)


@pytest.mark.parametrize("key,value", [("name", "SGD"), ("lr", 0.01), ("batch_size", 16),
                                      ("early_stopping", True), ("scheduler", "cosine")])
def test_optimizer_drift_is_rejected(protocol, key, value):
    cfg = config()
    cfg["optimizer"][key] = value
    with pytest.raises(ValueError):
        protocol.validate_config(cfg)


def test_cross_version_config_fields_are_rejected(protocol):
    cfg = config()
    cfg["v30_variant"] = "a"
    with pytest.raises(ValueError):
        protocol.validate_config(cfg)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_config_metadata_is_rejected(protocol, bad):
    cfg = config()
    cfg["metadata"] = {"measured_value": bad}
    with pytest.raises(ValueError):
        protocol.validate_config(cfg)


def test_canonical_hash_is_order_independent_but_detects_changed_plans(protocol):
    assert protocol.canonical_hash({"b": 2, "a": 1}) == (
        "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777")
    assert protocol.canonical_hash({"a": 1, "b": 2}) == protocol.canonical_hash({"b": 2, "a": 1})
    assert protocol.canonical_hash([{"id": "a"}, {"id": "b"}]) != protocol.canonical_hash([
        {"id": "b"}, {"id": "a"}])
    assert protocol.canonical_hash({"id": "a", "label": 1}) != protocol.canonical_hash({"id": "a", "label": 2})
    for bad in (float("nan"), {"rate": float("nan")}, {1: "ambiguous"}, "scalar"):
        with pytest.raises((ValueError, TypeError)):
            protocol.canonical_hash(bad)


def make_provenance_repo(root):
    (root / "adaptive_vcm/v31/nested").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "scripts").mkdir()
    (root / "adaptive_vcm/legacy.py").write_bytes(b"value = 1\r\n")
    (root / "adaptive_vcm/v31/nested/module.py").write_bytes(b"value = 2\r\n")
    for arm in "abc":
        (root / f"configs/v31_{arm}.json").write_text("{}\n", encoding="utf-8")
    (root / "scripts/kaggle_v31.py").write_text("value = 3\n", encoding="utf-8")
    (root / "scripts/other.py").write_text("value = 99\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Protocol Test",
                    "-c", "user.email=protocol@example.invalid", "commit", "-qm", "fixture"], check=True)


def test_provenance_hashes_nested_code_configs_runners_and_lf(protocol, tmp_path):
    make_provenance_repo(tmp_path)
    manifest = protocol.code_manifest_v31(tmp_path)
    assert manifest["schema"] == "adaptive-vcm-actions-v10"
    assert len(manifest["commit"]) == 40 and int(manifest["commit"], 16) > 0
    assert set(manifest["files_sha256"]) == {
        "adaptive_vcm/legacy.py", "adaptive_vcm/v31/nested/module.py",
        "configs/v31_a.json", "configs/v31_b.json", "configs/v31_c.json", "scripts/kaggle_v31.py"}
    nested = tmp_path / "adaptive_vcm/v31/nested/module.py"
    nested.write_bytes(b"value = 2\n")
    assert protocol.code_manifest_v31(tmp_path) == manifest
    nested.write_bytes(b"value = 4\n")
    changed = protocol.code_manifest_v31(tmp_path)
    assert changed["manifest_hash"] != manifest["manifest_hash"]
    assert changed["files_sha256"]["adaptive_vcm/v31/nested/module.py"] != manifest["files_sha256"]["adaptive_vcm/v31/nested/module.py"]
    nested.write_bytes(b"value = 2\n")
    (tmp_path / "configs/v31_b.json").write_text('{"modified":true}\n', encoding="utf-8")
    assert protocol.code_manifest_v31(tmp_path)["manifest_hash"] != manifest["manifest_hash"]
    (tmp_path / "configs/v31_b.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "scripts/kaggle_v31.py").write_text("value = 8\n", encoding="utf-8")
    assert protocol.code_manifest_v31(tmp_path)["manifest_hash"] != manifest["manifest_hash"]


def test_provenance_requires_full_git_commit(protocol, tmp_path):
    with pytest.raises(ValueError, match="Git|commit"):
        protocol.code_manifest_v31(tmp_path)
