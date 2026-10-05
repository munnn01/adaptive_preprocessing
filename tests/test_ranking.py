import argparse
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adaptive_vcm.ranking import (CONTEXT_DIM, RankPreprocessor, build_rank_context,
                                  measurement_targets, ranking_loss, static_action_order)


def probability(index=0):
    p = np.full(400, .05 / 399, np.float32)
    p[index] = .95
    return p


def test_context_records_codec_fragility_without_task_labels():
    clip = np.random.default_rng(25).integers(0, 256, (4, 16, 24, 3), np.uint8)
    mask = np.zeros((16, 24), np.float32)
    source = [probability(), probability(1)]
    anchor = [probability(2), probability(1)]
    c = build_rank_context(clip, 50, "h265", mask, source, anchor)
    assert c.shape == (CONTEXT_DIM,) and np.isfinite(c).all()
    np.testing.assert_array_equal(c, build_rank_context(clip.copy(), 50, "h265", mask.copy(), source, anchor))
    assert not np.array_equal(c, build_rank_context(clip, 45, "h264", mask, source, source))
    with pytest.raises(ValueError):
        build_rank_context(clip, 50, "h265", mask, source[:1], anchor[:1])
    with pytest.raises(ValueError):
        build_rank_context(clip, 50, "h265", mask, [np.ones(400)] * 2, anchor)


def test_all_action_targets_separate_safe_overhead_and_unsafe_saving():
    rows = [dict(name=name, coded_bytes=bytes_, distances=distances, decisions=decisions)
            for name, bytes_, distances, decisions in (
                ("identity", 1000, [0., 0.], [True, True]),
                ("safe_gain", 800, [.05, -.1], [True, True]),
                ("safe_overhead", 1100, [.0, 0.], [True, True]),
                ("unsafe_flip", 100, [-1., -1.], [True, False]),
                ("unsafe_kl", 500, [.10001, 0.], [True, True]))]
    safety, log_rate, eligible = measurement_targets(rows, slack=.1)
    np.testing.assert_array_equal(safety, [1, 1, 0, 0])
    np.testing.assert_allclose(np.exp(log_rate), [.8, 1.1, .1, .5])
    np.testing.assert_array_equal(eligible, [1, 0, 0, 0])
    assert static_action_order(safety[None], log_rate[None]) == [1, 2, 3, 4]
    rows[0]["decisions"] = [False, True]
    with pytest.raises(ValueError):
        measurement_targets(rows, slack=.1)


def test_minibatch_ranking_learns_conditional_useful_actions_without_identity_ce():
    torch.set_num_threads(2)
    torch.manual_seed(25)
    model = RankPreprocessor(16, action_names=("identity", "left", "right", "unsafe"))
    contexts = torch.zeros(32, CONTEXT_DIM)
    contexts[:16, 0], contexts[16:, 1] = 1., 1.
    safety = torch.ones(32, 3)
    safety[:, 2] = 0.
    rates = torch.full((32, 3), np.log(.9))
    rates[:16, 0], rates[16:, 1], rates[:, 2] = np.log(.6), np.log(.6), np.log(.1)
    weight = torch.ones(3)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(160):
        loss, _ = ranking_loss(model, contexts, safety, rates, weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert model.rank(contexts[0], 1) == [1]
    assert model.rank(contexts[-1], 1) == [2]
    assert len(set(model.rank(contexts[0], 3))) == 3
    assert 0 not in model.rank(contexts[0], 3)
    with pytest.raises(ValueError):
        model.rank(contexts[0], 4)


def test_weighted_safety_prediction_is_corrected_to_probability():
    model = RankPreprocessor(4, action_names=("identity", "safe", "rare"))
    model.safety_log_weight.copy_(torch.tensor([0., np.log(9.)]))
    scores = model.scores(torch.zeros(1, 2), torch.full((1, 2), np.log(.5)))
    torch.testing.assert_close(scores, torch.tensor([[.25, .05]]))


@pytest.mark.codec
def test_real_codec_collection_and_replay_keep_unsafe_byte_saving_out_of_targets(tmp_path, monkeypatch):
    from adaptive_vcm.codec import locate_ffmpeg
    if locate_ffmpeg() is None:
        pytest.skip("FFmpeg unavailable")
    import adaptive_vcm.train_ranking as training
    from adaptive_vcm.data import partition
    from adaptive_vcm.evaluate import ROOT
    from adaptive_vcm.preprocessing import Candidate
    frame = np.random.default_rng(25).integers(60, 140, (24, 32, 3), np.uint8)
    clip = np.repeat(frame[None], 4, axis=0)
    class FixtureTeacher:
        def __init__(self, *args):
            pass
        def probabilities(self, frames):
            return probability(int(frames.mean() >= 150))
        def saliency(self, frames):
            return np.ones(frames.shape[1:3], np.float32)
    names = ("identity", "safe_dc_fixture", "unsafe_dc_fixture")
    def fixture_bank(source, mask, qp):
        return [Candidate(names[0], source.copy()),
                Candidate(names[1], np.full_like(source, 100)),
                Candidate(names[2], np.full_like(source, 250))]
    ids = [f"fixture/{i}.mp4" for i in range(20) if partition(f"fixture/{i}.mp4") == "train"][:2]
    plan = [dict(id=id_, path="fixture", label=0) for id_ in ids]
    monkeypatch.setattr(training, "ar_plan", lambda *args: (plan, {}))
    monkeypatch.setattr(training, "read_video", lambda *args: clip.copy())
    monkeypatch.setattr(training, "ActionAnalyzer", FixtureTeacher)
    monkeypatch.setattr(training, "_task_bank", lambda: SimpleNamespace(ACTION_NAMES=names, build_task_bank=fixture_bank))
    cfg = json.loads((ROOT / "configs/v24_screen.json").read_text())
    cfg.update(ar_training="all_action_ranking", rank_top_k=1)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    args = argparse.Namespace(task="ar", root=tmp_path, config=config, count=2, measurements=10,
                              steps=5, width=8, batch_size=4, seed=25, lr=.001, out=tmp_path / "train")
    manifest = training.train(args)
    records = list(map(json.loads, (args.out / "measurements.jsonl").read_text().splitlines()))
    assert len(records) == 10
    assert {row["codec"] for row in records} == {"h264", "h265"}
    assert {row["qp"] for row in records} == {30, 35, 40, 45, 50}
    assert all(row["source_sha256"] == hashlib.sha256(clip.tobytes()).hexdigest() for row in records)
    assert all(len(row["context"]) == CONTEXT_DIM for row in records)
    assert all(row["actions"][0]["decisions"] == [True, True] for row in records)
    assert all(row["actions"][2]["decisions"] == [False, False] for row in records)
    assert any(row["feasible_actions"] > 0 for row in records)
    pack = np.load(args.out / "train_records.npz")
    assert pack["context"].shape == (10, CONTEXT_DIM)
    assert np.all(pack["safety"][:, 1] == 0) and np.all(pack["eligible"][:, 1] == 0)
    assert any(row["gradient_norm"] > 0 for row in map(json.loads, (args.out / "train.jsonl").read_text().splitlines()))
    state = torch.load(args.out / "preprocessor_last.pth", weights_only=False)
    assert state["schema"] == "adaptive-vcm-ranking-v4"
    assert state["action_names"] == list(names)
    assert state["static_action_order"][0] == 1
    assert manifest["fit_diagnostics"]["scope"].startswith("TRAIN resubstitution")
