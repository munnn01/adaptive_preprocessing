import importlib
import io
import json
import tarfile
from pathlib import Path

import pytest


def mod(): return importlib.import_module('scripts.run_v31_jobs')


class Client:
    def __init__(self, status='COMPLETE', hours=30, next_page=False, existing=False, fail_push=False):
        self.state,self.hours,self.next_page,self.existing,self.fail_push=status,hours,next_page,existing,fail_push
        self.calls=[]
    def call(self,account,argv,timeout=45):
        self.calls.append((account,argv))
        if argv[0]=='quota': output=json.dumps([{'resource':'GPU','remaining':str(self.hours)+'h'}])
        elif argv[1]=='list': output=json.dumps([{'ref':account+'/v31-fixture-oracle' if self.existing else account+'/old-job'}])
        elif argv[1]=='status': output='KernelWorkerStatus.'+self.state
        elif argv[1]=='push': output='Kernel version 1 successfully pushed'
        elif argv[1]=='logs': output='[v31-runtime] {"gpu":"Tesla T4"}\n'
        else: output=''
        return {'exit_code':1 if argv[1:2]==['push'] and self.fail_push else 0,'stdout':output,
                'stderr':'Next page token: token' if self.next_page and argv[1:2]==['list'] else '',
                'checked_utc':mod().now()}


def prepared(tmp_path,client,task='ar'):
    from tests.test_kaggle_v31 import args
    from scripts.kaggle_v31 import prepare_v31
    a=args(tmp_path,task); prepare_v31(a)
    registry=mod().Registry(tmp_path/'registry.json',client)
    job_id=registry.prepare(a.directory)
    return registry,job_id


@pytest.mark.parametrize('state,hours', [('RUNNING',30),('UNKNOWN',30),('COMPLETE',11),('COMPLETE',float('nan'))])
def test_active_unknown_or_quota_blocks_submit(tmp_path,state,hours):
    client=Client(state,hours); r,j=prepared(tmp_path,client)
    assert r.preflight(j)['eligible'] is False
    with pytest.raises(ValueError,match='preflight'): r.submit(j)
    assert not any(argv[1:2]==['push'] for _,argv in client.calls)


def test_duplicate_and_incomplete_inventory_block(tmp_path):
    client=Client(existing=True); r,j=prepared(tmp_path,client)
    assert not r.preflight(j)['eligible']
    client.existing=False; client.next_page=True
    assert not r.preflight(j)['eligible']


def test_push_failure_and_receipt_never_mean_running(tmp_path):
    client=Client(); r,j=prepared(tmp_path,client)
    assert r.preflight(j)['eligible']
    assert r.submit(j)['status']=='SUBMITTED_UNVERIFIED'
    with pytest.raises(ValueError,match='duplicate|intent'): r.submit(j)
    assert r.status(j)['status']=='COMPLETE'
    assert r.read()['jobs'][j]['runtime']['gpu']=='Tesla T4'


def test_failed_submission_keeps_durable_intent_without_retry(tmp_path):
    client=Client(fail_push=True); r,j=prepared(tmp_path,client)
    assert r.preflight(j)['eligible']
    assert r.submit(j)['status']=='SUBMISSION_FAILED'
    assert r.read()['jobs'][j]['submit_intent']
    with pytest.raises(ValueError,match='intent'): r.submit(j)


def test_stale_registry_snapshot_cannot_submit_twice(tmp_path,monkeypatch):
    import copy
    client=Client(); r,j=prepared(tmp_path,client); r.preflight(j)
    snapshot=r.read()
    assert r.submit(j)['status']=='SUBMITTED_UNVERIFIED'
    monkeypatch.setattr(r,'read',lambda:copy.deepcopy(snapshot))
    with pytest.raises(ValueError,match='intent|duplicate'): r.submit(j)
    assert sum(argv[1:2]==['push'] for _,argv in client.calls)==1


def test_verified_source_shards_merge_without_policy_fit(tmp_path,monkeypatch):
    import numpy as np
    from PIL import Image
    from tests.test_v31_measure import cfg,models,image_record
    from adaptive_vcm.v31.measure import collect
    from adaptive_vcm.v31.measure_store import atomic_json,load_measurements,read_json
    from adaptive_vcm.v31.protocol import code_manifest_v31,canonical_hash
    g=mod(); records=[image_record(tmp_path,str(i)) for i in (1,2)]
    Image.fromarray(np.full((100,201,3),32,np.uint8)).save(records[1]['path'])
    plans={'fit':[records[0]],'cal':[],'tune':[records[1]],'dev':[]}
    parent=tmp_path/'parent'; identity={'task':'od','plan_hash':canonical_hash(plans),
        'code_provenance':code_manifest_v31(Path(__file__).parents[1])}
    atomic_json(parent/'plan.json',{'plans':plans})
    atomic_json(parent/'run_state.json',{'identity':identity})
    directories=[]
    for index,split in enumerate(('fit','tune')):
        directory=tmp_path/('shard'+str(index)); directories.append(directory)
        atomic_json(directory/'run_state.json',{'identity':identity})
        collect({split:plans[split]},cfg(),directory/'measurements',models())
    result=g.merge_shards(directories,parent,tmp_path/'merged')
    assert result['sources']==2 and result['conditions']==16
    store=tmp_path/'merged'/'measurements'
    assert len(load_measurements(store,read_json(store/'expected.json')))==16
    assert not (tmp_path/'merged'/'arms').exists()
    with pytest.raises(ValueError,match='incomplete'): g.merge_shards(directories[:1],parent,tmp_path/'missing')
    with pytest.raises(ValueError,match='duplicate'): g.merge_shards([directories[0]]*2,parent,tmp_path/'repeated')


def test_od_requires_audited_same_release_ar(tmp_path):
    client=Client(); r,j=prepared(tmp_path,client,'od')
    assert not r.preflight(j)['eligible']
    assert 'AR' in ' '.join(r.read()['jobs'][j]['preflight']['reasons'])


def test_pool_preserves_order_and_redacts_all_credentials(tmp_path):
    g=mod(); pool=tmp_path/'pool.json'
    pool.write_text(json.dumps({'first':{'key':'SECRET_ONE','username':'first'},'second':'SECRET_TWO'}))
    visited=[]
    account=g.select_account(pool,lambda owner:visited.append(owner) or owner=='second')
    assert account=='second' and visited==['first','second']
    client=g.KaggleClient(pool)
    assert client.safe('SECRET_ONE SECRET_TWO')=='[REDACTED] [REDACTED]'


@pytest.mark.parametrize('name,kind', [('outputs/../escape','file'),('/absolute','file'),('outputs/link','symlink'),('outputs/device','device')])
def test_archive_traversal_links_and_special_files_rejected(tmp_path,name,kind):
    g=mod(); archive=tmp_path/'bad.tgz'
    with tarfile.open(archive,'w:gz') as tar:
        member=tarfile.TarInfo(name)
        if kind=='symlink': member.type=tarfile.SYMTYPE; member.linkname='../../escape'
        elif kind=='device': member.type=tarfile.CHRTYPE
        else: member.size=3
        tar.addfile(member,io.BytesIO(b'bad') if member.isfile() else None)
    with pytest.raises(ValueError,match='unsafe|archive'): g.extract_archive(archive,tmp_path/'extracted')
    assert not (tmp_path/'escape').exists()


def test_checksum_resume_rejects_corruption(tmp_path):
    g=mod(); source=tmp_path/'source'; source.mkdir(); (source/'one.json').write_text('{}')
    manifest=g.package_resume(source,tmp_path/'resume')
    assert manifest['files']['one.json']
    (tmp_path/'resume'/'one.json').write_text('corrupt')
    from adaptive_vcm.v31.run import import_resume
    with pytest.raises(ValueError,match='checksum'): import_resume(tmp_path/'resume',tmp_path/'out')
