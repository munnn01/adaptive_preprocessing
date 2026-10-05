"""Synthetic full V27 evaluation fixture for artifact-integrity regressions.

Use fixture = make_eval_fixture(tmp_path, monkeypatch), fixture['write'](),
then audit_v27.audit(fixture['run']). audit_training alone is mocked, keeping
all evaluation-side checks active. Source geometry and config are registered.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import scripts.audit_v27 as auditmod
from adaptive_vcm.anchor_bank import ACTION_NAMES
from adaptive_vcm.portfolio_ranking import PortfolioRankPreprocessor, _fit_memory
from adaptive_vcm.data import partition, fingerprint
from adaptive_vcm.evaluate import _ar_curves


def make_eval_fixture(tmp_path, monkeypatch):
    cfg = json.loads((auditmod.ROOT / 'configs/v27_screen.json').read_text())
    qps = cfg['qps']
    trainids = [f'review-fixture/{i}' for i in range(100) if partition(f'review-fixture/{i}') == 'train'][:4]
    devids = [f'review-fixture/{i}' for i in range(100) if partition(f'review-fixture/{i}') == 'dev'][:2]
    bpp = 8*2000/(16*128*128)
    x = np.zeros((4,41)); x[:,0] = 50/51; x[:,2:4] = .7; x[:,4] = .5; x[:,-1] = np.log1p(bpp)
    model = PortfolioRankPreprocessor(ACTION_NAMES, _fit_memory(x,np.zeros((4,41))), {'low':{'neighbors':32,'mix':0.},'high':{'neighbors':32,'mix':0.}}, trainids)
    training = {'config':cfg, 'code':{}, 'train_ids':trainids, 'train_ids_sha256':fingerprint(trainids)}
    measured = [{'source_sha256':hashlib.sha256(id_.encode()).hexdigest(), 'source_id':id_} for id_ in trainids]
    monkeypatch.setattr(auditmod, 'audit_training', lambda *a, **kw: (model, training, measured))
    run = Path(tmp_path); evaldir = run/'eval'; evaldir.mkdir(parents=True)
    manifest = {'config':cfg, 'code':{}, 'task':'ar', 'codecs':['h264','h265'], 'count':2, 'split':'dev', 'bootstrap_draws':0, 'ids':devids, 'ids_sha256':fingerprint(devids), 'rate_denominator':'original pre-transform T*H*W pixels', 'component_evaluation':True, 'ar_guard_rule':'anchor_relative_v2', 'proposal_budget':3}
    (run / 'train').mkdir()
    (run / 'train/preprocessor_last.pth').write_bytes(b'synthetic-checkpoint')
    manifest['checkpoint_sha256'] = hashlib.sha256(b'synthetic-checkpoint').hexdigest()
    allrows, audits = [], []
    digest = lambda s: hashlib.sha256(s.encode()).hexdigest()
    for codec in manifest['codecs']:
        for id_ in devids:
            for qp in qps:
                context = np.zeros(41); context[0]=qp/51; context[1]=int(codec=='h265'); context[2:4]=.7; context[4]=.5; context[-1]=np.log1p(bpp)
                details=model.proposal_details(context); learned=model.rank(context); static=model.static_action_order[:3]; group=model.group_static_action_order(context)
                prefix='trained_prior__'
                base={'coded_bytes':2000, 'relative_task_distance':[0.,0.], 'preserves_decision':[True,True], 'codec_seconds':0., 'geometry':[16,128,128,3]}
                identity={**base, 'name':'identity', 'profile':None, 'action_index':None, 'stream_sha256':digest('identity'), 'proposed_by_learned':False, 'proposed_by_static':False, 'proposed_by_group_static':False, 'ranking_context':context.tolist(), 'ranking_context_sha256':hashlib.sha256(context.tobytes()).hexdigest(), 'learned_order':learned, 'global_static_order':static, 'group_static_order':group, 'proposal_details':details, 'anchor_decoded_sha256':digest('decoded')}
                controls=[identity] + [{**base, 'name':name, 'profile':None, 'action_index':None, 'stream_sha256':digest(name), 'proposed_by_learned':False, 'proposed_by_static':False, 'proposed_by_group_static':False} for name in cfg['ar_candidates'][1:]]
                bank=[{**base, 'name':prefix+name, 'profile':name, 'action_index':i, 'stream_sha256':digest(name), 'proposed_by_learned':i in learned, 'proposed_by_static':i in static, 'proposed_by_group_static':i in group} for i,name in enumerate(ACTION_NAMES[1:],1)]
                audits.append({'id':id_, 'codec':codec, 'qp':qp, 'selected':'identity', 'candidates':controls+bank})
                for arm in auditmod.ARMS:
                    chosen=bank[learned[0]-1] if arm=='learned_raw' else identity
                    correct={name:int(qp<=40) for name in cfg['ar_evaluators']}
                    allrows.append({'id':id_, 'codec':codec, 'qp':qp, 'arm':arm, 'candidate':chosen['name'], 'stream_sha256':chosen['stream_sha256'], 'coded_bytes':chosen['coded_bytes'], 'source_sha256':digest(id_), 'bpp':bpp, 'correct':correct})
    def write():
        (evaldir/'manifest.json').write_text(json.dumps(manifest))
        for codec in manifest['codecs']:
            for suffix,primary in [('rows',True),('components',False)]:
                rows=[r for r in allrows if r['codec']==codec and ((r['arm'] in ('anchor','adaptive'))==primary)]
                (evaldir/f'{codec}_{suffix}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (evaldir/'selection_audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in audits))
        summary={'target_confirmed':False, 'results':{codec:_ar_curves([r for r in allrows if r['codec']==codec and r['arm'] in ('anchor','adaptive')],cfg['ar_evaluators'],qps,0,cfg['seed']) for codec in manifest['codecs']}}
        (evaldir/'summary.json').write_text(json.dumps(summary))
    return dict(run=run, manifest=manifest, rows=allrows, audits=audits, write=write, model=model, training=training)
