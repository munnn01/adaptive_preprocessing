"""FIT-frozen portfolios and complete, oracle-first TUNE investment gates."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import re

import numpy as np

from .actions import action_registry
from .guard import annotate_row, choose_feasible, is_feasible
from .measure_store import validate_row
from .metrics import _grid, paired_comparisons, summarize_curves
from .protocol import CODECS, QPS, SOURCE_COUNTS, canonical_hash, code_manifest_v31, validate_config


VERSION = 'v31-oracle-gate-1'


def project_rows(rows,registry):
    """Transparent arm view of immutable union measurements; never remeasure."""
    result = []
    descriptors = [asdict(action) for action in registry]
    for row in rows:
        by_name = {a['descriptor']['name']:a for a in row['actions']}
        if len(by_name) != len(row['actions']):
            raise ValueError('duplicate action registry')
        selected = []
        for descriptor in descriptors:
            action = by_name.get(descriptor['name'])
            if action is None or action['descriptor'] != descriptor:
                raise ValueError('arm action registry missing or changed')
            selected.append(action)
        projected = dict(row,actions=selected)
        projected['arm_registry_hash'] = canonical_hash(descriptors)
        result.append(projected)
    return result


def _greedy(rows,policy,indices,k=3):
    annotations = [annotate_row(row,policy) for row in rows]
    anchor = np.asarray([row['actions'][0]['total_bytes'] for row in rows],float)
    costs = np.stack([[obs['total_bytes'] if is_feasible(obs,obs['guard_features'],policy) and obs['total_bytes'] <= a*(1-policy['min_savings']) else a
                       for obs in observations] for a,observations in zip(anchor,annotations)])
    best,chosen = anchor.copy(),[]
    for _ in range(min(k,len(indices))):
        options = [i for i in indices if i not in chosen]
        winner = min(options,key=lambda i:(-float((best-np.minimum(best,costs[:,i])).sum()),i))
        chosen.append(winner)
        best = np.minimum(best,costs[:,winner])
    return [rows[0]['actions'][i]['descriptor']['name'] for i in chosen]


def fit_static(fit_rows,policy,registry,task):
    rows = project_rows([row for row in fit_rows if row['split']=='fit'],registry)
    _grid(rows,task)
    if policy['task'] != task:
        raise ValueError('static task/policy mismatch')
    descriptors = [asdict(a) for a in registry]
    groups = {}
    for codec in CODECS:
        for qp in QPS:
            subset = [row for row in rows if row['codec']==codec and row['qp']==qp]
            groups[f'{codec}:{qp}'] = {
                'expanded_static_k3':_greedy(subset,policy,list(range(1,len(registry)))),
                'legacy_static_k3':_greedy(subset,policy,[i for i,a in enumerate(registry) if a.kind=='profile'])}
    fixed,spatial_objectives = 'area112',{}
    if task == 'od':
        names = [a.name for a in registry]
        for name in ('identity','area256','area224','area192'):
            index = names.index(name)
            grouped = []
            for codec in CODECS:
                for qp in QPS:
                    subset = [row for row in rows if row['codec']==codec and row['qp']==qp]
                    fractions = []
                    for row in subset:
                        observations = annotate_row(row,policy)
                        chosen = choose_feasible([observations[0],observations[index]],policy) if index else 0
                        selected = observations[index] if chosen else observations[0]
                        fractions.append(1-selected['total_bytes']/observations[0]['total_bytes'])
                    grouped.append(float(np.mean(fractions)))
            spatial_objectives[name] = float(np.mean(grouped))
        fixed = min(spatial_objectives,key=lambda n:(-spatial_objectives[n],names.index(n)))
    result = {'version':'v31-fit-static-1','task':task,'policy_hash':policy['policy_hash'],
              'registry_hash':canonical_hash(descriptors),'registry':descriptors,
              'fit_rows_hash':canonical_hash(rows),'fit_source_ids':sorted({row['source_id'] for row in rows}),
              'groups':groups,'fixed_spatial':fixed,'fixed_spatial_fit_objectives':spatial_objectives,
              'objective':'per-group greedy marginal total packet bytes; OD size equal-weight mean saving fractions across codec/QP groups'}
    result['portfolio_hash'] = canonical_hash(result)
    return result


def select_portfolios(rows,policy,portfolios,task):
    if portfolios['task'] != task or portfolios['policy_hash'] != policy['policy_hash'] or portfolios.get('portfolio_hash') != canonical_hash({k:v for k,v in portfolios.items() if k!='portfolio_hash'}):
        raise ValueError('frozen portfolio identity mismatch')
    result = []
    for row in rows:
        observations = annotate_row(row,policy)
        names = [a['descriptor']['name'] for a in observations]
        if canonical_hash([a['descriptor'] for a in observations]) != portfolios['registry_hash']:
            raise ValueError('portfolio registry differs from arm observations')
        def guarded(proposals):
            indices = [0]+[names.index(name) for name in proposals if name!='identity']
            return indices[choose_feasible([observations[i] for i in indices],policy)]
        group = portfolios['groups'][f"{row['codec']}:{row['qp']}"]
        choices = {'oracle':choose_feasible(observations,policy),
                   'expanded_static_k3':guarded(group['expanded_static_k3']),
                   'legacy_static_k3':guarded(group['legacy_static_k3']),
                   'controls':guarded([a['descriptor']['name'] for a in observations[1:] if a['descriptor']['kind']=='control']),
                   'fixed_spatial_guarded':guarded([portfolios['fixed_spatial']])}
        fixed_names = ['area112','area96'] if task=='ar' else ['identity','area256','area224','area192']
        for name in fixed_names:
            index = names.index(name)
            if not observations[index]['available']:
                raise ValueError('unavailable unguarded fixed spatial packet')
            choices[name] = index
            choices[name+'_guarded'] = guarded([name])
        choices['fixed_spatial'] = names.index(portfolios['fixed_spatial'])
        result.append(dict(row,choices=choices))
    return result


def evaluate_portfolios(rows,policy,portfolios,task,cfg):
    cfg = validate_config(cfg)
    selected = select_portfolios(rows,policy,portfolios,task)
    pairs = [('anchor','oracle'),('expanded_static_k3','oracle'),('fixed_spatial','oracle')]
    pairs.extend(('anchor',method) for method in selected[0]['choices'] if method not in ('oracle','fixed_spatial','expanded_static_k3'))
    result = paired_comparisons(selected,task,pairs,cfg['bootstrap_draws'],cfg['seed'])
    result['selection_counts'] = {method:{name:sum(row['actions'][row['choices'][method]]['descriptor']['name']==name for row in selected)
                                          for name in [a['descriptor']['name'] for a in selected[0]['actions']]}
                                  for method in selected[0]['choices']}
    result['packet_bytes'] = {method:sum(row['actions'][row['choices'][method]]['total_bytes'] for row in selected)
                              for method in selected[0]['choices']}
    result['probe_budget'] = {'oracle':len(selected[0]['actions']), 'expanded_static_k3':4,'legacy_static_k3':4,
                              'fixed_spatial':1,'fixed_spatial_guarded':2,
                              'controls':1+sum(a['descriptor']['kind']=='control' for a in selected[0]['actions'][1:])}
    return result


def assess_headroom(results):
    codecs,passed = {},True
    for codec in CODECS:
        reference = results['curves']['anchor'][codec]
        oracle = results['curves']['oracle'][codec]
        summaries = {name:summarize_curves(results['curves'][name][codec],oracle)
                     for name in ('anchor','expanded_static_k3','fixed_spatial')}
        reasons = []
        for name,summary in summaries.items():
            value = summary['pchip_bd_rate_pct']
            target = -15. if name=='anchor' else 0.
            if value is None or not value < target-1e-9:
                reasons.append(f'{name}: insufficient measured PCHIP headroom or overlap')
        if min(summaries['anchor']['same_qp_quality_gap_pp']) < -1.-1e-10:
            reasons.append('anchor: quality loss exceeds 1pp at a corresponding QP')
        codecs[codec] = {**summaries,'passed':not reasons,'reasons':reasons}
        passed = passed and not reasons
    return {'passed':bool(passed),'codecs':codecs,'rule':'both codecs; PCHIP<-15 versus anchor, <0 versus expandedK3/fixed, all quality gaps >=-1pp'}


def _integrity(fit_rows,tune_rows,policy,registry,task,cfg):
    reasons = []
    sets = [{row['source_id'] for row in rows} for rows in (fit_rows,tune_rows)]
    counts = SOURCE_COUNTS[task]
    info = {'fit_sources':len(sets[0]),'cal_sources':len(policy.get('cal_source_ids',[])),
            'tune_sources':len(sets[1]),'fit_conditions':len(fit_rows),'tune_conditions':len(tune_rows)}
    if any(info[key] != expected for key,expected in [('fit_sources',counts['fit']),('cal_sources',counts['cal']),
            ('tune_sources',counts['tune']),('fit_conditions',counts['fit']*8),('tune_conditions',counts['tune']*8)]):
        reasons.append('source/condition count differs from full frozen protocol')
    if sets[0]&sets[1] or (sets[0]|sets[1])&set(policy.get('cal_source_ids',[])):
        reasons.append('FIT/CAL/TUNE source IDs overlap')
    if registry != action_registry(task,cfg['v31_arm']):
        reasons.append('registry differs from frozen V31 arm bank')
    manifest = code_manifest_v31(Path(__file__).resolve().parents[2])
    measurement_cfg = dict(cfg,task=task,v31_arm='b',experiment='v31-b')
    identity,seen_pixels = None,{}
    for split,rows in [('fit',fit_rows),('tune',tune_rows)]:
        try:
            _grid(rows,task)
            for row in rows:
                validate_row(row)
                if row['split'] != split or row['config_hash'] != canonical_hash(measurement_cfg) or row['code_manifest_hash'] != manifest['manifest_hash']:
                    raise ValueError('split/config/code provenance differs from immutable measurements')
                if [a['descriptor'] for a in row['actions']] != [asdict(a) for a in action_registry(task,'b')]:
                    raise ValueError('full measurement union registry missing')
                models = row['model_hashes']
                if (models['teachers'] != policy.get('cal_model_hashes') or
                        row['code_manifest_hash'] != policy.get('cal_code_manifest_hash') or
                        row['config_hash'] != policy.get('cal_measurement_config_hash')):
                    raise ValueError('CAL/FIT/TUNE measurement provenance differs')
                expected_names = {'teachers':cfg['ar_teachers'] if task=='ar' else [cfg['od_teacher']],
                                  'evaluators':cfg['ar_evaluators'] if task=='ar' else [cfg['od_evaluator']]}
                if set(models) != set(expected_names) or any(set(models[role]) != set(names) or any(re.fullmatch('[0-9a-f]{64}',h) is None for h in models[role].values()) for role,names in expected_names.items()):
                    raise ValueError('named model identity mismatch')
                digest = canonical_hash(models)
                if identity is not None and digest != identity:
                    raise ValueError('model state changed across measurements')
                identity = digest
                pixel = row['source']['source_sha256']
                if pixel in seen_pixels and seen_pixels[pixel] != row['source_id'] or pixel in policy.get('cal_pixels_sha256',[]):
                    raise ValueError('duplicate source content across FIT/CAL/TUNE')
                seen_pixels[pixel] = row['source_id']
                if task=='ar' and (row['source']['timing_status']!='known_constant' or row['source']['duration'] is None):
                    raise ValueError('unavailable verified primary AR duration')
        except (ValueError,KeyError,TypeError) as exc:
            reasons.append(split+': '+str(exc))
    if policy.get('policy_hash') != canonical_hash({k:v for k,v in policy.items() if k!='policy_hash'}) or policy['config_hash'] != canonical_hash(cfg) or policy['task'] != task or policy['arm'] != cfg['v31_arm']:
        reasons.append('frozen CAL policy/config identity mismatch')
    return {**info,'passed':not reasons,'reasons':reasons},manifest


def oracle_report(fit_rows,tune_rows,policy,registry,task,cfg):
    cfg = validate_config(cfg)
    integrity,manifest = _integrity(fit_rows,tune_rows,policy,registry,task,cfg)
    fit_view,tune_view = project_rows(fit_rows,registry),project_rows(tune_rows,registry)
    bindings = {'config_hash':canonical_hash(cfg),'policy_hash':policy['policy_hash'],
                'registry_hash':canonical_hash([asdict(a) for a in registry]),
                'fit_rows_hash':canonical_hash(fit_view),'tune_rows_hash':canonical_hash(tune_view),
                'code_manifest_hash':manifest['manifest_hash']}
    result = {'version':VERSION,'task':task,'arm':cfg['v31_arm'],'bindings':bindings,'integrity':integrity,
              'eligible':False,'results':None,'headroom':None,'static':None,
              'scope':'TUNE oracle headroom; exploratory; no DEV/TEST access'}
    if integrity['passed']:
        portfolio = fit_static(fit_view,policy,registry,task)
        results = evaluate_portfolios(tune_view,policy,portfolio,task,cfg)
        headroom = assess_headroom(results)
        result.update(static=portfolio,results=results,headroom=headroom,eligible=headroom['passed'])
    result['gate_hash'] = canonical_hash(result)
    return result


def validate_gate(gate,expected):
    if not isinstance(gate,dict) or gate.get('gate_hash') != canonical_hash({k:v for k,v in gate.items() if k!='gate_hash'}):
        raise ValueError('oracle gate hash mismatch')
    if gate.get('version') != VERSION or gate.get('eligible') is not True or not gate.get('integrity',{}).get('passed'):
        raise ValueError('oracle blocked: no eligible complete measured headroom')
    task,arm = gate.get('task'),gate.get('arm')
    if task not in SOURCE_COUNTS or arm not in ('a','b','c'):
        raise ValueError('invalid gate task/arm')
    counts = SOURCE_COUNTS[task]
    integrity = gate['integrity']
    if integrity.get('reasons') != [] or any(integrity.get(key) != value for key,value in
            [('fit_sources',counts['fit']),('cal_sources',counts['cal']),('tune_sources',counts['tune']),
             ('fit_conditions',counts['fit']*8),('tune_conditions',counts['tune']*8)]):
        raise ValueError('oracle blocked: source/grid integrity incomplete')
    bindings = gate.get('bindings',{})
    required = ('config_hash','policy_hash','registry_hash','fit_rows_hash','tune_rows_hash','code_manifest_hash')
    if any(re.fullmatch('[0-9a-f]{64}',str(bindings.get(key,''))) is None for key in required):
        raise ValueError('oracle gate provenance bindings missing')
    try:
        actual = assess_headroom(gate['results'])
        if not actual['passed'] or actual != gate['headroom']:
            raise ValueError('oracle blocked: numerical headroom changed')
    except (KeyError,TypeError) as exc:
        raise ValueError('oracle blocked: headroom curves missing') from exc
    for key,value in expected.items():
        if (gate.get(key) if key in ('task','arm') else bindings.get(key)) != value:
            raise ValueError('oracle gate expected identity mismatch: '+key)
