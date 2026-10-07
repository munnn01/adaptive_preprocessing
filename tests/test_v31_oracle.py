import copy
from dataclasses import asdict
import importlib
import importlib.util

import pytest

from adaptive_vcm.v31.actions import Action, action_registry
from adaptive_vcm.v31.guard import fit_policy
from adaptive_vcm.v31.protocol import CODECS, QPS, canonical_hash
from tests.test_v31_guard import cfg, predictions, ar, rows as cal_rows
from tests.test_v31_metrics import ar_rows


def mod():
    assert importlib.util.find_spec('adaptive_vcm.v31.oracle') is not None, 'oracle gate missing'
    return importlib.import_module('adaptive_vcm.v31.oracle')


def static_fixture():
    registry = tuple(Action(n,'ar',None,'profile' if n.startswith('p') else 'control',n if n.startswith('p') else None,1)
                     for n in ['identity','p1','p2','p3','p4','area112','area96'])
    measured = []
    for codec in CODECS:
        for qp in QPS:
            for i in range(4):
                # p1,p2,p3 redundantly cover source0; p4 covers source1.
                amounts = [1000,100 if i==0 else 1000,110 if i==0 else 1000,120 if i==0 else 1000,
                           200 if i==1 else 1000,800,900]
                actions = [{'descriptor':asdict(a),'available':True,'total_bytes':amount,
                            'decoded_sha256':a.name,'predictions':predictions('ar',ar([.8,.2]))}
                           for a,amount in zip(registry,amounts)]
                measured.append({'task':'ar','source_id':str(i),'split':'fit','codec':codec,'qp':qp,
                                 'source_predictions':predictions('ar',ar([.8,.2])),
                                 'anchor_predictions':actions[0]['predictions'],'actions':actions,
                                 'source':{'duration':[1,1]},'ground_truth':{'label':0}})
    return registry, measured, fit_policy(cal_rows(),'ar',cfg('b'))


def test_greedy_marginal_coverage_and_fit_only_freeze():
    g = mod()
    registry, measured, policy = static_fixture()
    fitted = g.fit_static(measured,policy,registry,'ar')
    assert fitted['groups']['h264:30']['expanded_static_k3'] == ['p1','p4','area112']
    assert fitted['groups']['h264:30']['legacy_static_k3'] == ['p1','p4','p2']
    extra = copy.deepcopy(measured)
    for row in extra:
        row['split'] = 'dev'
        row['ground_truth']['label'] = 999
        row['actions'][2]['total_bytes'] = 1
    assert g.fit_static(measured+extra,policy,registry,'ar') == fitted


def test_portfolios_fixed_unguarded_and_guarded_and_full_bank():
    g = mod()
    registry,measured,policy = static_fixture()
    for row in measured:
        for action in row['actions']:
            action['predictions']['evaluators'] = {'r2plus1d_18':ar([.8,.2])}
        row['actions'][5]['predictions']['teachers'] = predictions('ar',ar([.1,.9]))['teachers']
    portfolio = g.fit_static(measured,policy,registry,'ar')
    selected = g.select_portfolios(measured,policy,portfolio,'ar')
    assert all(row['choices']['area112'] == 5 for row in selected)
    assert all(row['choices']['area112_guarded'] == 0 for row in selected)
    assert selected[0]['choices']['oracle'] == 1
    assert selected[1]['choices']['oracle'] == 4
    assert portfolio['fixed_spatial'] == 'area112'


def headroom_fixture():
    reference = {'qps':list(QPS),'rate':[1000,800,600,400],'quality':[90,85,80,75]}
    curves = {}
    for name,scale in [('anchor',1),('oracle',.8),('expanded_static_k3',.9),('fixed_spatial',.95)]:
        curves[name] = {codec:{**copy.deepcopy(reference),'rate':[r*scale for r in reference['rate']]} for codec in CODECS}
    return {'curves':curves}


def test_known_synthetic_headroom_qualifies_numerically():
    g = mod()
    result = g.assess_headroom(headroom_fixture())
    assert result['passed']
    assert result['codecs']['h264']['anchor']['pchip_bd_rate_pct'] == pytest.approx(-20)


@pytest.mark.parametrize('failure',['boundary','one_codec','static','quality','overlap'])
def test_headroom_failures_never_qualify(failure):
    g = mod()
    result = headroom_fixture()
    curves = result['curves']
    if failure in ('boundary','one_codec'):
        for codec in CODECS if failure=='boundary' else ['h265']:
            curves['oracle'][codec]['rate'] = [r*.85 for r in curves['anchor'][codec]['rate']]
    elif failure == 'static':
        curves['expanded_static_k3']['h264']['rate'] = [r*.7 for r in curves['anchor']['h264']['rate']]
    elif failure == 'quality':
        curves['oracle']['h264']['quality'][2] -= 1.1
    else:
        curves['oracle']['h264']['quality'] = [60,55,50,45]
    assert not g.assess_headroom(result)['passed']


def test_gate_hash_expected_identity_and_integrity_are_mandatory():
    g = mod()
    gate = {'version':'v31-oracle-gate-1','eligible':True,'task':'ar','arm':'b',
            'bindings':{key:'a'*64 for key in ('config_hash','policy_hash','registry_hash','fit_rows_hash','tune_rows_hash','code_manifest_hash')},
            'integrity':{'passed':True,'reasons':[], 'fit_sources':96,'cal_sources':32,'tune_sources':128,
                         'fit_conditions':768,'tune_conditions':1024},
            'results':headroom_fixture(), 'headroom':g.assess_headroom(headroom_fixture())}
    gate['gate_hash'] = canonical_hash(gate)
    g.validate_gate(gate,{'task':'ar','arm':'b','config_hash':'a'*64,'policy_hash':'a'*64})
    with pytest.raises(ValueError):
        g.validate_gate(gate,{'task':'od'})
    changed = copy.deepcopy(gate)
    changed['eligible'] = False
    with pytest.raises(ValueError):
        g.validate_gate(changed,{})
    changed['gate_hash'] = canonical_hash({k:v for k,v in changed.items() if k!='gate_hash'})
    with pytest.raises(ValueError,match='blocked'):
        g.validate_gate(changed,{})


def test_incomplete_oracle_reports_blocked_instead_of_shrinking_sources():
    g = mod()
    registry,measured,policy = static_fixture()
    fit = measured
    tune = copy.deepcopy(measured)
    for row in tune:
        row['split'] = 'tune'
        row['source_id'] = 't'+row['source_id']
    report = g.oracle_report(fit,tune,policy,registry,'ar',cfg('b'))
    assert not report['eligible']
    assert not report['integrity']['passed']
    assert any('count' in reason or 'registry' in reason for reason in report['integrity']['reasons'])
    with pytest.raises(ValueError,match='blocked'):
        g.validate_gate(report,{})


def test_od_size_freezes_once_and_oracle_scores_complete_set_coco():
    from adaptive_vcm.v31.metrics import curves
    from tests.test_v31_guard import det
    g = mod()
    registry = action_registry('od','b')
    policy = fit_policy(cal_rows('od'),'od',cfg('b'))
    measured = []
    for codec in CODECS:
        for j,qp in enumerate(QPS):
            for i in range(4):
                ground = {'image_id':i+1,'categories':[{'id':1,'name':'x'}],
                          'annotations':[{'id':i+1,'image_id':i+1,'category_id':1,
                                          'bbox':[0,0,10,10],'area':100,'iscrowd':0}]}
                actions = []
                for action in registry:
                    prediction = predictions('od',det())
                    prediction['evaluators'] = {'resnet50':det(box=(0,0,10,10) if i>=j else (20,20,30,30))}
                    actions.append({'descriptor':asdict(action),'available':True,'total_bytes':
                                    {'area256':700,'area224':500,'area192':600}.get(action.name,1000),
                                    'predictions':prediction,'decoded_sha256':action.name})
                measured.append({'task':'od','source_id':str(i+1),'split':'fit','codec':codec,'qp':qp,
                                 'ground_truth':ground,'source':{'original_shape':[10,10]},'actions':actions,
                                 'source_predictions':predictions('od',det()),'anchor_predictions':actions[0]['predictions']})
    portfolio = g.fit_static(measured,policy,registry,'od')
    assert portfolio['fixed_spatial'] == 'area224'
    selected = g.select_portfolios(measured,policy,portfolio,'od')
    assert {row['actions'][row['choices']['oracle']]['descriptor']['name'] for row in selected} == {'area224'}
    assert curves(selected,'od','oracle','h264')['quality'] == pytest.approx([100,56.43564356435643,25.24752475247524,6.435643564356436])
