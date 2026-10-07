"""Live matched-budget encoding and frozen DEV replay of verified packets."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re
import time

import numpy as np

from .actions import action_registry,execute_action,execute_actions,map_detections
from .codec import V31Codec
from .guard import annotate_row,choose_feasible
from .measure import CandidateEncodingError,collect,prepare_support
from .measure_models import ObservationCache,json_value,model_hashes
from .measure_store import atomic_json,load_measurements,read_json,sha
from .metrics import paired_comparisons
from .oracle import project_rows,select_portfolios
from .protocol import CODECS,QPS,SOURCE_COUNTS,canonical_hash,code_manifest_v31,validate_config
from .selector import RUNTIME_FIELDS,build_context
from .train import CHECKPOINT_SCHEMA,context_from_row,state_sha256
from .transport import pack_recipe


def source_availability(source,registry):
    return np.asarray([not (action.repeat_factor==2 and (source.get('padded',False) or source.get('source_fps') is None or source.get('duration') is None))
                       for action in registry],bool)


def _proposals(selector,context,availability,k=3):
    indices = selector.rank(context,k,availability)
    if not isinstance(indices,list) or len(indices)>k or len(set(indices))!=len(indices) or any(type(i) is not int or not 1<=i<len(availability) or not availability[i] for i in indices):
        raise ValueError('invalid learned/static action proposal')
    return indices


def choose_stream(sample,selector,policy,codec,teachers,cfg):
    # Fail on accidental evaluator/GT plumbing before calling a model.
    if not isinstance(sample,dict) or set(sample)-RUNTIME_FIELDS:
        raise ValueError('runtime sample contains unauthorized fields')
    task = sample.get('task')
    cfg = validate_config(cfg)
    if not isinstance(codec,tuple) or len(codec)!=2 or codec[0] not in CODECS or type(codec[1]) is not int or codec[1] not in QPS:
        raise ValueError('runtime codec must be (name,QP)')
    codec_name,qp = codec
    names = cfg['ar_teachers'] if task=='ar' else [cfg['od_teacher']]
    if set(teachers)!=set(names) or any(getattr(teachers[n],'name',None)!=n or getattr(teachers[n],'model_hash',None)!=policy.get('cal_model_hashes',{}).get(n) for n in names):
        raise ValueError('runtime teacher names/state differ from frozen CAL policy')
    if policy.get('policy_hash')!=canonical_hash({k:v for k,v in policy.items() if k!='policy_hash'}) or policy['task']!=task or policy['arm']!=cfg['v31_arm']:
        raise ValueError('runtime frozen guard identity mismatch')
    mode = cfg.get('selection_mode','learned_k3')
    if mode not in ('learned_k3','static_k3','diagnostic_union_controls'):
        raise ValueError('unknown runtime selection mode')
    registry = action_registry(task,cfg['v31_arm'])
    if selector.task!=task or tuple(selector.action_names)!=tuple(a.name for a in registry):
        raise ValueError('runtime selector registry mismatch')
    cache,packets,observations = ObservationCache(),{},[]
    start = time.perf_counter()
    rgb = sample['rgb']
    def predict(decoded):
        result = {}
        for name in names:
            value = cache.observe(teachers[name],decoded,task)
            if task=='od':
                value = json_value(map_detections(value,decoded.shape[1:3],sample['source_transform'],sample['original_shape']))
            result[name] = value
        return {'teachers':result}
    source_predictions = predict(rgb)
    controls,support = prepare_support(rgb,task,source_predictions['teachers'],teachers,cache,cfg['od_score_threshold'])
    conditioned = dict(sample,codec=codec_name,control_protection=controls)
    packet_for_slot,encode_seconds = {},0.
    def probe(index,execution=None):
        nonlocal encode_seconds
        action = registry[index]
        execution = execution if execution is not None else execute_action(conditioned,action,qp,support)
        obs = {'descriptor':asdict(action),'available':execution['available'],'reason':execution['reason'],
               'total_bytes':None,'elementary_bytes':None,'predictions':None,'decoded_sha256':None,
               'recipe':None,'packet_hash':None,'rgb_sha256':execution['rgb_sha256'],'error':None}
        if execution['available']:
            recipe = execution['recipe']
            key = (execution['rgb_sha256'],pack_recipe(recipe))
            if key not in packets:
                try:
                    packets[key] = V31Codec(codec_name,qp,cfg['preset'],recipe.fps).roundtrip(execution['rgb'],recipe)
                    encode_seconds += packets[key].seconds
                except CandidateEncodingError as exc:
                    if index==0: raise
                    packets[key] = str(exc)[:1000]
            packet = packets[key]
            if isinstance(packet,str):
                obs.update(available=False,reason='candidate_encoding_failure',error=packet)
            else:
                obs.update(total_bytes=packet.total_bytes,elementary_bytes=packet.elementary_bytes,
                           predictions=predict(packet.decoded),decoded_sha256=sha(packet.decoded.tobytes()),
                           recipe=asdict(recipe),packet_hash=packet.packet_hash)
                packet_for_slot[len(observations)] = packet
        observations.append(obs)
    probe(0)
    anchor = observations[0]
    context = build_context(conditioned,[source_predictions['teachers'][n] for n in names],
                            [anchor['predictions']['teachers'][n] for n in names],anchor,task,qp,codec_name,support)
    availability = source_availability(conditioned,registry)
    proposals = _proposals(selector,context,availability,cfg['proposal_k'])
    if mode=='diagnostic_union_controls':
        proposals = list(dict.fromkeys(proposals+[i for i,a in enumerate(registry) if i and a.kind=='control']))
    executions = execute_actions(conditioned,tuple(registry[i] for i in proposals),qp,support) if proposals else []
    for index,execution in zip(proposals,executions): probe(index,execution)
    row = {'task':task,'codec':codec_name,'qp':qp,'source_predictions':source_predictions,
           'anchor_predictions':anchor['predictions'],'actions':observations}
    annotated = annotate_row(row,policy)
    selected = choose_feasible(annotated,policy)
    teacher_observations = sum(key[1]==task for key in cache)
    return {'packet':packet_for_slot[selected],'action_index':([0]+proposals)[selected],
            'selection_index':selected,'action_name':annotated[selected]['descriptor']['name'],
            'proposals':proposals,'observations':annotated,'teacher_only_row':row,'scope':mode,
            'stats':{'slots':len(observations),'distinct_encodes':len(packets),
                     'aliases':sum(o['available'] for o in observations)-len({p.packet_hash for p in packet_for_slot.values()}),
                     'teacher_observations':teacher_observations,'saliency_calls':sum(key[1]=='saliency' for key in cache),
                     'codec_seconds':encode_seconds,'total_seconds':time.perf_counter()-start}}


def adaptive_claims(results):
    claims = {}
    for codec in CODECS:
        anchor = results['comparisons']['anchor->learned'][codec]
        baseline_pairs = [results['comparisons'][f'{method}->learned'][codec] for method in ('expanded_static_k3','fixed_spatial')]
        quality = min(anchor['same_qp_quality_gap_pp']) >= -1.-1e-10
        superiority = all(pair['pchip_bd_rate_pct'] is not None and pair['pchip_bd_rate_pct'] < -1e-9
                          and pair['pchip_ci']['hi'] is not None and pair['pchip_ci']['hi'] < 0
                          and pair['pchip_ci']['finite_fraction']>=.9 for pair in baseline_pairs)
        target = anchor['pchip_bd_rate_pct'] is not None and anchor['pchip_bd_rate_pct'] < -10.-1e-9
        claims[codec] = {'adaptive_evidence':bool(superiority and quality),
                         'learned_bd_lt_minus10':bool(target and quality),'quality_gap_ok':bool(quality),
                         'target_confirmed':False,'scope':'exploratory DEV, conditional on frozen weights/policy'}
    return claims


def _replay(rows,selector,policy,statics,task):
    selected = select_portfolios(rows,policy,statics,task)
    budgets,counts = [],{}
    registry = action_registry(task,policy['arm'])
    for row in selected:
        context = context_from_row(row,task)
        availability = source_availability(row['source'],registry)
        proposals = _proposals(selector,context,availability)
        annotated = annotate_row(row,policy)
        indices = [0]+proposals
        choice = indices[choose_feasible([annotated[i] for i in indices],policy)]
        row['choices']['learned'] = choice
        union = list(dict.fromkeys(indices+[i for i,a in enumerate(registry) if i and a.kind=='control']))
        row['choices']['diagnostic_union_controls'] = union[choose_feasible([annotated[i] for i in union],policy)]
        name = row['actions'][choice]['descriptor']['name']
        counts[name] = counts.get(name,0)+1
        def cost(probes):
            unique = {row['actions'][i]['packet_hash']:row['actions'][i] for i in probes if row['actions'][i]['available']}
            return {'slots':len(probes),'distinct_valid_packets':len(unique),
                    'codec_seconds':sum(a['seconds'] for a in unique.values()),
                    'teacher_decoded_inputs':len({(a['decoded_sha256'],tuple(a['coded_shape'])) for a in unique.values()})*len(policy['teacher_names'])}
        group = statics['groups'][f"{row['codec']}:{row['qp']}"]
        names = [a.name for a in registry]
        budgets.append({'source_id':row['source_id'],'codec':row['codec'],'qp':row['qp'],
                        'learned_proposals':[names[i] for i in proposals], 'learned':cost(indices),
                        'expanded_static_k3':cost([0]+[names.index(n) for n in group['expanded_static_k3']]),
                        'diagnostic_union_controls':cost(union)})
    return selected,budgets,counts


def evaluate_dev(plan,selector,policy,statics,cfg,out,models):
    metadata = getattr(selector,'metadata',{})
    if metadata.get('schema')!=CHECKPOINT_SCHEMA or metadata.get('checkpoint_role')!='LAST' or state_sha256(selector.state_dict())!=metadata.get('state_sha256'):
        raise ValueError('frozen verified LAST selector required before DEV access')
    cfg = validate_config(cfg)
    task = policy['task']
    manifest = code_manifest_v31(Path(__file__).resolve().parents[2])
    if metadata.get('config_hash')!=canonical_hash(cfg) or metadata.get('policy_hash')!=policy['policy_hash'] or metadata.get('code_manifest_hash')!=manifest['manifest_hash']:
        raise ValueError('frozen selector/config/policy/code mismatch')
    partitions,pixels = metadata.get('source_partitions',{}),metadata.get('source_pixels_sha256',{})
    if any(len(partitions.get(split,[]))!=SOURCE_COUNTS[task][split] or
           set(pixels.get(split,{}))!=set(partitions[split]) or
           any(re.fullmatch('[0-9a-f]{64}',str(value)) is None for value in pixels[split].values())
           for split in ('fit','cal','tune')):
        raise ValueError('frozen source partition/fingerprint provenance missing')
    if plan.get('measurement_store'):
        store = Path(plan['measurement_store'])
        expected = read_json(store/'expected.json')
        rows = load_measurements(store,expected)
    else:
        store = Path(out)/'measurements'
        measured = collect({'dev':plan['dev']},dict(cfg,task=task,v31_arm='b',experiment='v31-b'),store,models)
        expected,rows = measured['expected'],measured['rows']
    if (expected['code_provenance']['manifest_hash']!=manifest['manifest_hash'] or
            expected['model_hashes']!=model_hashes(models,task,cfg) or
            expected['config_hash']!=canonical_hash(dict(cfg,task=task,v31_arm='b',experiment='v31-b')) or
            expected['model_hashes']['teachers']!=policy['cal_model_hashes']):
        raise ValueError('DEV frozen analyzer/code mismatch')
    rows = [row for row in rows if row['split']=='dev']
    previous_ids = {s for sources in partitions.values() for s in sources}
    previous_pixels = {h for mapping in pixels.values() for h in mapping.values()}
    if not rows or any(row['source_id'] in previous_ids or row['source']['source_sha256'] in previous_pixels for row in rows):
        raise ValueError('DEV source partition missing/overlaps FIT/CAL/TUNE IDs or pixels')
    rows = project_rows(rows,action_registry(task,cfg['v31_arm']))
    rows = [dict(row,artifact_store=str(store)) for row in rows]
    selected,budgets,counts = _replay(rows,selector,policy,statics,task)
    comparisons = [('anchor','learned'),('expanded_static_k3','learned'),('fixed_spatial','learned'),
                   ('legacy_static_k3','learned'),('controls','learned'),('anchor','diagnostic_union_controls')]
    results = paired_comparisons(selected,task,comparisons,cfg['bootstrap_draws'],cfg['seed'])
    claims = adaptive_claims(results)
    complete = len({row['source_id'] for row in rows})==SOURCE_COUNTS[task]['dev']
    if not complete:
        for claim in claims.values(): claim.update(adaptive_evidence=False,learned_bd_lt_minus10=False)
    totals = {name:sum(row['actions'][row['choices'][name]]['total_bytes'] for row in selected) for name in selected[0]['choices']}
    result = {'version':'v31-frozen-dev-1','task':task,'arm':cfg['v31_arm'],'target_confirmed':False,
              'source_count':len({row['source_id'] for row in rows}),'complete_full_dev':complete,
              'gate_hash':metadata['gate_hash'],'checkpoint_state_sha256':metadata['state_sha256'],
              'policy_hash':policy['policy_hash'],'portfolio_hash':statics['portfolio_hash'],'measurement_expected_hash':canonical_hash(expected),
              'results':results,'claims':claims,'learned_selection_counts':counts,'packet_bytes':totals,
              'incremental_savings_bytes':{name:totals[name]-totals['learned'] for name in ('expanded_static_k3','fixed_spatial','controls')},
              'probe_budgets':budgets,'probe_cost_scope':'reconstructed from measured packets; offline full-grid audit costs are separate',
              'diagnostic_union_controls':'extra probes, excluded from primary adaptive claim',
              'scope':'exploratory DEV; no TEST; no refit on DEV'}
    result['report_hash'] = canonical_hash(result)
    atomic_json(Path(out)/'dev_report.json',result)
    return result
