import numpy as np

from adaptive_vcm.codec import Encoded
from adaptive_vcm.rank_selection import choose_rank_stream, ranking_diagnostics
from adaptive_vcm.task_bank import ACTION_NAMES, build_task_bank


def test_unproposed_bank_oracle_never_enters_primary_pool():
    clip = np.random.default_rng(25).integers(40, 200, (3, 32, 48, 3), np.uint8)
    mask = np.full(clip.shape[1:3], .5, np.float32)
    bank = build_task_bank(clip, mask, 50)
    costs = [1000, 950, 960, 970, 930, 940, 980, *([600] * 11)]
    costs[10] = 980
    values = {(c.clip.shape, c.clip.tobytes()): b for c, b in zip(bank, costs)}
    class Codec:
        codec, qp = 'h265', 50
        def roundtrip(self, source):
            size = values[(source.shape, source.tobytes())]
            return Encoded(source.copy(), str(size).encode().ljust(size, b'x'), 0.)
    class Teacher:
        def probabilities(self, source):
            result = np.zeros(400)
            result[:2] = [.8, .2]
            return result
    class Learned:
        static_action_order = [4, 5, 10] + [i for i in range(1, 18) if i not in (4, 5, 10)]
        calls = 0
        def rank(self, context, top_k):
            self.calls += 1
            assert context.shape == (1640,) and top_k == 3
            return [1, 2, 3]
    cfg = dict(ar_candidates=['identity'], rank_top_k=3, ar_confidence=.6,
               ar_require_anchor_decision=True, ar_kl_slack=.1, min_savings=.01)
    teachers, model = [Teacher(), Teacher()], Learned()
    source = [t.probabilities(clip) for t in teachers]
    anchor, chosen, name, audit, arms = choose_rank_stream(
        clip, mask, Codec(), cfg, teachers, source, model, components=True)
    assert model.calls == 1
    assert anchor.coded_bytes == 1000 and chosen.coded_bytes == 950
    assert name == 'learned_rank__' + ACTION_NAMES[1]
    assert arms['controls'][0].coded_bytes == 1000
    assert arms['static_adaptive'][0].coded_bytes == 930
    assert arms['bank_oracle'][0].coded_bytes == 600
    assert sum(c['proposed_by_learned'] for c in audit) == 3
    assert sum(c['proposed_by_static'] for c in audit) == 3
    assert len(audit) == 18
    small = choose_rank_stream(clip, mask, Codec(), cfg, teachers, source, model, components=False)
    assert len(small[3]) == 4 and small[1].data == chosen.data


def test_policy_contribution_uses_marginal_bytes_and_equal_budget():
    rows = [dict(id=i, qp=50, arm=a, coded_bytes=b, candidate=c)
            for i, a, b, c in [('a', 'anchor', 1000, 'identity'),
                               ('a', 'adaptive', 700, 'learned_rank__x'),
                               ('b', 'anchor', 500, 'identity'),
                               ('b', 'adaptive', 500, 'identity')]]
    component = [dict(id=i, qp=50, arm=a, coded_bytes=b)
                 for i, a, b in [('a', 'controls', 800), ('b', 'controls', 500),
                                 ('a', 'static_adaptive', 750), ('b', 'static_adaptive', 450)]]
    result = ranking_diagnostics(rows, component, [50])['50']
    assert result['selected_learned_points'] == 1
    assert np.isclose(result['controls']['incremental_saving_pct_of_anchor'], 100/15)
    assert result['static_adaptive']['incremental_saving_pct_of_anchor'] == 0
    assert result['static_adaptive']['adaptive_byte_wins'] == 1
    assert result['static_adaptive']['adaptive_byte_losses'] == 1


def test_ranking_checkpoint_and_all_component_arms_with_actual_codecs(tmp_path, monkeypatch):
    import argparse
    import json
    import pytest
    import torch
    import adaptive_vcm.evaluate as evaluation
    from adaptive_vcm.codec import locate_ffmpeg
    from adaptive_vcm.ranking import CONTEXT_DIM, CONTEXT_SCHEMA, RankPreprocessor
    if locate_ffmpeg() is None:
        pytest.skip('FFmpeg unavailable')
    torch.set_num_threads(2)
    clip = np.repeat(np.random.default_rng(25).integers(40, 200, (1, 24, 32, 3), np.uint8), 4, axis=0)
    class Teacher:
        def __init__(self, name, device):
            self.name = name
        def probabilities(self, frames):
            p = np.zeros(400)
            p[:2] = [.8, .2]
            return p
        def saliency(self, frames):
            return np.ones(frames.shape[1:3], np.float32)
    monkeypatch.setattr(evaluation, 'ActionAnalyzer', Teacher)
    monkeypatch.setattr(evaluation, 'ar_plan', lambda *a: ([dict(id='fixture', path='fixture', label=0)], {}))
    monkeypatch.setattr(evaluation, 'read_video', lambda *a: clip.copy())
    cfg = json.loads((evaluation.ROOT / 'configs/v25_screen.json').read_text())
    cfg.update(qps=[30, 40, 50], ar_candidates=['identity', 'protected_mild'])
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(cfg))
    model = RankPreprocessor(8)
    checkpoint = tmp_path / 'checkpoint.pth'
    torch.save(dict(schema=model.schema, model=model.state_dict(), width=8, task='ar',
                    steps=1, train_ids_sha256='fixture', training_config=cfg,
                    context_schema=CONTEXT_SCHEMA, context_dim=CONTEXT_DIM,
                    action_names=list(ACTION_NAMES), static_action_order=model.static_action_order), checkpoint)
    args = argparse.Namespace(task='ar', root=tmp_path, annotations=None, config=config, split='dev',
                              count=1, codecs=['h264', 'h265'], bootstrap=0, checkpoint=checkpoint,
                              ablate_learned=True, save_streams=False, out=tmp_path / 'eval')
    summary = evaluation.run(args)
    assert not summary['target_confirmed']
    for codec in args.codecs:
        assert set(summary['component_results'][codec]) == {'controls', 'static_adaptive', 'bank_oracle',
                                                          'learned_guarded', 'learned_raw'}
        for qp in cfg['qps']:
            d = summary['policy_contribution'][codec][str(qp)]
            assert d['adaptive_bytes'] <= d['controls']['coded_bytes']
            assert d['bank_oracle']['coded_bytes'] <= d['adaptive_bytes']
    for row in map(json.loads, (args.out / 'selection_audit.jsonl').read_text().splitlines()):
        assert sum(c['proposed_by_learned'] for c in row['candidates']) == 3
        assert sum(c['proposed_by_static'] for c in row['candidates']) == 3
