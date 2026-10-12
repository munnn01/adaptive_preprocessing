import numpy as np
import torch
import pytest
from adaptive_vcm.v34.learning import targets, masked_loss, calibrate, select_action
from adaptive_vcm.v34.checkpoint import save_checkpoint, load_checkpoint
from adaptive_vcm.v34.model import Controller

def test_same_rate_quality_gain_is_not_discarded_and_missing_task_is_masked():
    damage=np.array([[0, np.nan, 0],[-.2,np.nan,0],[.9,np.nan,0]])
    choice=targets(np.array([1,1.002,.7]),damage,[1,0,1],[.05]*3,1.01)
    assert choice==1
    with pytest.raises(ValueError): targets([1],[[0,0,0]],[0,0,0],[.05]*3,1.01)
    pred=torch.tensor([[[1.,2.,3.]]],requires_grad=True)
    truth=torch.tensor([[[0.,float('nan'),0.]]])
    loss=masked_loss(pred,truth,torch.tensor([[True,False,True]]))
    loss.backward()
    assert loss.item()==5 and pred.grad[0,0,1].item()==0

def test_calibration_fallback_and_same_rate_gain():
    p=np.zeros((20,3,3)); actual=p.copy(); actual[:,2,0]=.3
    cal=calibrate(p,actual,np.ones((20,3)),np.ones((20,3)),np.ones((20,3),bool),.9)
    risk=np.array([[0,0,0],[-.1,0,0],[0,0,0]])
    assert select_action(risk,np.array([1,1.001,.8]),cal,[1,0,0],[.05]*3,1.01)==1
    assert select_action(risk,np.array([1,1.2,.8]),cal,[1,0,0],[.05]*3,1.01)==0

def test_checkpoint_is_weights_only_safe_and_detects_metadata_tampering(tmp_path):
    model=Controller(3); path=tmp_path/'model.pth'
    meta={'seed':303401,'source_commit':'a'*40,'protocol':'pilot','calibration':{'risk':[[0]*3]*3,'rate':[0]*3}}
    save_checkpoint(path,model,meta)
    loaded=load_checkpoint(path)
    assert loaded.metadata==meta
    value=torch.load(path,weights_only=True)
    value['metadata']['seed']=0; torch.save(value,path)
    with pytest.raises(ValueError,match='metadata'): load_checkpoint(path)

def test_three_seed_training_freezes_cal_and_roundtrips_native_metadata(tmp_path):
    from adaptive_vcm.v34.runner import train_models
    from adaptive_vcm.v34.protocol import digest,settings
    cfg=settings(); cfg['epochs']=1
    inputs={key:(torch.rand(1,3,12,12),torch.ones(1,3,12,12)/3) for key in ('fit','cal')}
    sources={k:{'split':s,'pixel_sha256':str(i)*64} for i,(k,s) in enumerate([('fit','train'),('cal','cal')])}
    rows=[{'id':k,'split':s,'condition':[0.,.7,0.,1.,0.,1.01],
        'mask':[0,1,0],'rate':[1.,.8,1.001],
        'damage':[[None,0.,None],[None,.01,None],[None,-.1,None]]} for k,s in [('fit','train'),('cal','cal')]]
    cache={'schema':'adaptive-vcm-v34-1','experiment':'ar_od','origin':{'origin_commit':'a'*40,'bundle_sha256':'b'*64},'sources':sources,'rows':rows}
    cache['sha256']=digest(cache)
    # Keep the same finite bank dimensions as the production model.
    for r in rows:
        r['rate'] += [1.]*10; r['damage'] += [[None,0.,None]]*10
    cache['sha256']=digest({k:v for k,v in cache.items() if k!='sha256'})
    models=train_models(cache,inputs,cfg,tmp_path,'cpu',{})
    assert len(models)==3 and len({m.checkpoint_sha256 for m in models})==3
    assert all(m.metadata['calibration']['count']==1 for m in models)
    assert all(len(m.metadata['training_history'])==1 for m in models)

def test_controller_trains_single_images_and_video_without_temporal_padding():
    from adaptive_vcm.v34.learning import train_controller
    cfg={'epochs':1,'lr':.001,'damage_tolerance':[.05]*3,'budget_ratio':1.01}
    inputs={k:(torch.rand(t,3,12,12),torch.ones(t,3,12,12)/3) for k,t in [('image',1),('clip',16)]}
    rows=[dict(id=k,condition=[0.,.7,float(k=='clip'),1.,0.,1.01],mask=[int(k=='clip'),1,0],
        rate=[1.,.9,1.],damage=[[0 if k=='clip' else None,0.,None],
        [0 if k=='clip' else None,0.,None],[0 if k=='clip' else None,0.,None]]) for k in inputs]
    history=train_controller(Controller(3),rows,inputs,cfg,'cpu')
    assert len(history)==1 and np.isfinite(history[0]['loss'])
