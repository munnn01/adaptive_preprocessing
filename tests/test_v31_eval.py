import importlib
import importlib.util
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.test_v31_metrics import ar_rows


def mod():
    assert importlib.util.find_spec('adaptive_vcm.v31.select') is not None, 'V31 frozen DEV evaluator missing'
    return importlib.import_module('adaptive_vcm.v31.select')


def test_exploratory_claim_requires_direct_ci_against_both_baselines():
    from adaptive_vcm.v31.metrics import paired_comparisons
    g = mod()
    rows = ar_rows()
    for row in rows:
        row['choices']['fixed_spatial']=2
        row['choices']['expanded_static_k3']=2
    results = paired_comparisons(rows,'ar',[('anchor','learned'),('expanded_static_k3','learned'),('fixed_spatial','learned')],100,2)
    claims = g.adaptive_claims(results)
    assert all(value['adaptive_evidence'] for value in claims.values())
    assert all(value['learned_bd_lt_minus10'] for value in claims.values())
    assert all(value['target_confirmed'] is False for value in claims.values())
    results['comparisons']['fixed_spatial->learned']['h264']['pchip_ci']['finite_fraction']=.89
    assert not g.adaptive_claims(results)['h264']['adaptive_evidence']
    results['comparisons']['fixed_spatial->learned']['h265']['pchip_ci']['hi']=0.
    assert not g.adaptive_claims(results)['h265']['adaptive_evidence']


def test_dev_rejects_unfrozen_selector_before_reading_plan(tmp_path):
    g = mod()
    with pytest.raises(ValueError,match='frozen'):
        g.evaluate_dev({'not':'a plan'},object(),{}, {}, {},tmp_path/'dev',{})
    assert not (tmp_path/'dev').exists()


def test_real_packet_dev_replay_is_frozen_and_toy_scope_inconclusive(tmp_path):
    from adaptive_vcm.v31.actions import action_registry
    from adaptive_vcm.v31.guard import fit_policy
    from adaptive_vcm.v31.measure import collect
    from adaptive_vcm.v31.oracle import fit_static
    from adaptive_vcm.v31.protocol import canonical_hash,code_manifest_v31
    from adaptive_vcm.v31.selector import ActionSelector,CONTEXT_FIELDS
    from adaptive_vcm.v31.train import make_checkpoint,load_selector,CHECKPOINT_SCHEMA
    from tests.test_v31_guard import cfg,rows as cal_rows
    from tests.test_v31_measure import models
    from tests.test_v31_select import bind_policy
    g = mod()
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        config = cfg('b')
        nets = models('od')
        records = []
        for image_id in (1000,1):
            rgb = np.full((100,201,3),0 if image_id==1000 else 64,np.uint8)
            path = tmp_path/f'{image_id}.png'; Image.fromarray(rgb).save(path)
            records.append({'id':str(image_id),'image_id':image_id,'path':str(path),'width':201,'height':100,
                            'annotations':[{'id':image_id,'image_id':image_id,'category_id':1,
                                            'bbox':[0,0,10,10],'area':100,'iscrowd':0}],
                            'categories':[{'id':1,'name':'x'}]})
        store = tmp_path/'measurements'
        measured = collect({'fit':[records[0]],'dev':[records[1]]},{**config,'task':'od'},store,nets)
        calibration = cal_rows('od')
        for row in calibration:
            row['source_id']='2000'
        policy = bind_policy(fit_policy(calibration,'od',config),nets['od']['teachers'])
        registry = action_registry('od','b')
        statics = fit_static(measured['rows'],policy,registry,'od')
        partitions = {'fit':[str(i) for i in range(1000,1075)],'cal':[str(i) for i in range(2000,2025)],
                      'tune':[str(i) for i in range(3000,3100)]}
        pixels = {split:{s:hashlib.sha256(s.encode()).hexdigest() for s in sources} for split,sources in partitions.items()}
        pixels['fit']['1000']=next(row['source']['source_sha256'] for row in measured['rows'] if row['split']=='fit')
        model = ActionSelector('od',tuple(a.name for a in registry))
        metadata = {'schema':CHECKPOINT_SCHEMA,'task':'od','arm':'b','width':64,'action_names':list(model.action_names),
                    'context_fields':list(CONTEXT_FIELDS),'policy_hash':policy['policy_hash'],
                    'registry_hash':statics['registry_hash'],'checkpoint_role':'LAST','gate_hash':'a'*64,
                    'config_hash':canonical_hash(config),'fit_rows_hash':statics['fit_rows_hash'],
                    'code_manifest_hash':code_manifest_v31(Path(__file__).parents[1])['manifest_hash'],
                    'epochs':8,'seed':303101,'training_source_ids':partitions['fit'],'training_conditions':600,
                    'source_partitions':partitions,'source_pixels_sha256':pixels,
                    'history':[{'epoch':i+1,'conditions':600,'sources':75,'checkpoint_role':'LAST' if i==7 else 'training'} for i in range(8)]}
        frozen = load_selector(make_checkpoint(model,metadata),{})
        before = {name:value.clone() for name,value in frozen.state_dict().items()}
        report = g.evaluate_dev({'measurement_store':str(store)},frozen,policy,statics,config,tmp_path/'dev',nets)
        assert report['source_count']==1 and not report['complete_full_dev']
        assert report['target_confirmed'] is False
        assert not any(value['adaptive_evidence'] for value in report['claims'].values())
        assert all(torch.equal(before[name],value) for name,value in frozen.state_dict().items())
        assert all(budget['learned']['slots']==4 for budget in report['probe_budgets'])
        assert 'reconstructed' in report['probe_cost_scope']
        assert (tmp_path/'dev'/'dev_report.json').exists()
    finally:
        torch.set_num_threads(old_threads)
