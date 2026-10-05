import copy

import pytest

from adaptive_vcm.data import partition, fingerprint
from scripts import audit_v27


def fixture():
    ids = [f'v27-audit-fixture-{i}' for i in range(100) if partition(f'v27-audit-fixture-{i}') == 'dev'][:2]
    cfg = dict(frames=16, ar_size=128, ar_evaluators=['r2plus1d_18', 'r3d_18'])
    manifest = dict(ids=ids, ids_sha256=fingerprint(ids), count=2, split='dev',
                    codecs=['h264', 'h265'], rate_denominator='original pre-transform T*H*W pixels')
    rows = [dict(id=i, codec=c, qp=q, arm=a, source_sha256='a' * 64,
            coded_bytes=1234, bpp=8*1234/(16*128*128), correct={m: 1 for m in cfg['ar_evaluators']})
        for i in ids for c in manifest['codecs'] for q in audit_v27.QPS for a in audit_v27.ARMS]
    return manifest, cfg, rows


@pytest.mark.parametrize('change', ['none', 'bpp', 'denominator', 'ids_hash', 'ids', 'split', 'grid', 'source_pixels'])
def test_evaluation_audit_binds_actual_rate_and_complete_dev_source_grid(change):
    manifest, cfg, rows = fixture()
    assert hasattr(audit_v27, 'validate_evaluation_records'), 'evaluation provenance gate is missing'
    if change == 'bpp':
        rows[0]['bpp'] *= .5
    elif change == 'denominator':
        manifest['rate_denominator'] = 'transformed pixels'
    elif change == 'ids_hash':
        manifest['ids_sha256'] = 'invalid'
    elif change == 'ids':
        manifest['ids'] = manifest['ids'][:1]
        manifest['ids_sha256'] = fingerprint(manifest['ids'])
    elif change == 'split':
        manifest['split'] = 'train'
    elif change == 'grid':
        rows[1] = copy.deepcopy(rows[0])
    elif change == 'source_pixels':
        rows[1]['source_sha256'] = 'b' * 64
    if change == 'none':
        audit_v27.validate_evaluation_records(manifest, cfg, rows)
    else:
        with pytest.raises(AssertionError):
            audit_v27.validate_evaluation_records(manifest, cfg, rows)


@pytest.mark.parametrize('change', ['none', 'bpp_with_recomputed_summary', 'ids', 'raw_bytes'])
def test_full_evaluation_audit_rejects_rehashed_rate_or_source_evidence(tmp_path, monkeypatch, change):
    from v27_eval_fixture import make_eval_fixture
    evidence = make_eval_fixture(tmp_path, monkeypatch)
    if change == 'bpp_with_recomputed_summary':
        for row in evidence['rows']:
            if row['arm'] == 'adaptive':
                row['bpp'] *= .5
    elif change == 'ids':
        evidence['manifest']['ids'] = ['unrelated-source']
        evidence['manifest']['ids_sha256'] = 'invalid'
    elif change == 'raw_bytes':
        raw = next(r for r in evidence['rows'] if r['arm'] == 'learned_raw')
        raw['coded_bytes'] += 1
        raw['bpp'] = 8 * raw['coded_bytes'] / (16 * 128 * 128)
    evidence['write']()
    if change == 'none':
        assert audit_v27.audit(evidence['run'])['passes'] is True
    else:
        with pytest.raises(AssertionError):
            audit_v27.audit(evidence['run'])
