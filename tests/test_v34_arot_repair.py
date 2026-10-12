import numpy as np
import pytest
import torch
from adaptive_vcm.v34.protocol import settings
from adaptive_vcm.v34.runner import condition
from adaptive_vcm.v34.model import Controller
from adaptive_vcm.v34.checkpoint import save_checkpoint, load_checkpoint


def repair_cfg():
    cfg=settings()
    cfg.update(domain_condition=True, source_guard_limits=[.025,0,.05], source_shortlist=3,
               risk_scale=[.05]*3, detector='resnet50', ot_detector_size=960,
               detector_detections=300, detector_batch=2)
    return cfg


def test_tracking_quality_units_and_nonmonotone_fit_rejection():
    from adaptive_vcm.v34.evaluation import tracking_curve
    a=dict(bpp=[4,3,2,1],quality=[40,30,20,10])
    b=dict(bpp=[3.8,2.8,1.8,.8],quality=[41,31,21,11])
    result=tracking_curve(a,b)
    assert result['same_qp_gap_pp']==pytest.approx([1]*4)
    assert result['quality_unit']=='HOTA percentage points'
    assert result['bd_reliable']
    a['quality']=[40,10,20,15]
    result=tracking_curve(a,b)
    assert not result['bd_reliable'] and result['bd_rate_pct'] is None
    assert 'nonmonotone' in result['invalid_reason']


def test_controller_can_distinguish_label_and_pseudo_domains_and_roundtrip(tmp_path):
    cfg=repair_cfg()
    ar=condition('h264',35,[1,0,1],cfg,domain='ar')
    ot=condition('h264',35,[1,0,1],cfg,domain='ot')
    assert len(ar)==9 and ar!=ot
    model=Controller(3,conditions=9)
    pair=torch.rand(2,2,3,12,12)
    value=model(pair,pair,torch.tensor([ar,ot]))
    path=tmp_path/'model.pth'; save_checkpoint(path,model,{'seed':1})
    loaded=load_checkpoint(path)
    assert loaded.conditions==9
    assert torch.allclose(value['risk'],loaded(pair,pair,torch.tensor([ar,ot]))['risk'])


def test_calibration_does_not_mix_label_loss_with_source_consistency():
    from adaptive_vcm.v34.learning import calibrate_domains, domain_calibration
    predicted=[dict(risk=np.zeros((2,3)),rate=np.ones(2)) for _ in range(20)]
    rows=[dict(domain=d,damage=[[0,None,0],[v,None,0]],rate=[1,1],mask=[1,0,1])
          for d,v in [('ar',1.)]*10+[('ot',.01)]*10]
    cal=calibrate_domains(rows,predicted,.9)
    assert domain_calibration(cal,'ar')['risk'][1][0]==pytest.approx(1)
    assert domain_calibration(cal,'ot')['risk'][1][0]==pytest.approx(.01)
    with pytest.raises(ValueError): domain_calibration(cal,'od')


def test_real_source_guard_can_accept_when_predicted_bound_rejects_and_logs_work():
    from adaptive_vcm.v34.evaluation import choose_packet
    from adaptive_vcm.v32.codec import Packet
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(()))
            self.metadata={'calibration':{'risk':[[0]*3]*3,'rate':[0]*3}}
        def forward(self,x,m,c):
            return dict(logits=torch.tensor([[0.,2.,1.]]),risk=torch.tensor([[[0.,0.,0.],[1.,0.,1.],[1.,0.,1.]]]),
                        rate=torch.tensor([[1.,.8,.7]]))
    rgb=np.zeros((2,8,8,3),np.uint8)
    packets=[Packet(bytes([i])*size,rgb+i,rgb.shape,size-40,40,0.) for i,size in enumerate([100,90,80])]
    calls=[]
    def encode(action): calls.append(action); return packets[action]
    def score(decoded):
        action=int(decoded[0,0,0,0]); return [[.1,None,.2],[.11,None,.21],[.2,None,.21]][action]
    pair=(torch.zeros(2,3,8,8),torch.ones(2,3,8,8)/3)
    _,selected,packet,details=choose_packet(Model(),pair,'h264',35,[1,0,1],repair_cfg(),packets[0],encode,
                                            domain='ar',source_score=score)
    assert selected==1 and packet is packets[1] and details['calibrated_action']==0
    assert set(calls)=={1,2}
    assert details['source_guard']['scope']=='source consistency; no task accuracy guarantee'
    assert len(details['tried_actions'])==3
    assert details['source_guard']['rejected']==[2]
    assert details['source_guard']['limits']==[.025,0,.05]


def test_source_guard_rejects_missing_or_nonfinite_active_cost_and_paid_rate():
    from adaptive_vcm.v34.evaluation import source_guard_choice
    rates=[1,.8,.8,1.02]
    costs=[[0,None,0],[.01,None,None],[float('nan'),None,0],[0,None,0]]
    action,details=source_guard_choice(rates,costs,[1,0,1],[.025,0,.05],1.01)
    assert action==0 and details['rejected']==[1,2,3]


def test_ot_detector_transform_retains_resolution_and_detection_capacity():
    from torchvision.models.detection.transform import GeneralizedRCNNTransform
    from types import SimpleNamespace
    from adaptive_vcm.v34.tasks import Heads
    heads=Heads.__new__(Heads); heads.cfg=repair_cfg()
    transform=GeneralizedRCNNTransform(320,640,[0,0,0],[1,1,1])
    heads.detector=SimpleNamespace(model=SimpleNamespace(transform=transform,roi_heads=SimpleNamespace(detections_per_img=100)))
    heads.configure_detector()
    with heads.detector_resolution(np.zeros((1,960,960,3),np.uint8)):
        images,_=transform([torch.zeros(3,960,960)])
        assert images.image_sizes==[(960,960)]
    assert transform.min_size==(320,)
    assert heads.detector.model.roi_heads.detections_per_img==300


def test_source_only_ar_cost_never_requires_label_or_tracking_gt():
    # Missing label was previously indexed unconditionally by Heads.costs.
    # Exercise actual losses with a tiny injected frozen head, not a mock result.
    from adaptive_vcm.v34.tasks import Heads
    from adaptive_vcm.v33ap.sources import SourceRecord
    from types import SimpleNamespace
    class TinySwin(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.features=torch.nn.Sequential(torch.nn.Identity(),torch.nn.Identity())
        def forward(self,x):
            f=self.features(x.permute(0,2,3,4,1)); z=f.mean((1,2,3,4))
            return torch.stack((z,-z),1)
    model=TinySwin(); heads=Heads.__new__(Heads); heads.device='cpu'; heads.experiment='ar_ot'
    heads.action=SimpleNamespace(model=model,logits=model)
    rgb=np.zeros((2,8,8,3),np.uint8)
    r=SourceRecord('ar/unlabelled','ar',{},rgb,None,{})
    r.mask=[1,0,0]; r.source_predictions={'ar':np.array([.5,.5])}
    r.source_features={'ar':torch.zeros(1,96)}
    assert heads.costs(rgb,r,source_only=True)[0]==pytest.approx(.1)




@pytest.mark.codec
@pytest.mark.skipif(not __import__('os').environ.get('V34_TRACKEVAL'),reason='requires pinned TrackEval')
def test_repair_evaluates_shared_stream_and_clean_tracking_without_selection_gt(tmp_path,monkeypatch):
    import json,os
    from types import SimpleNamespace
    from tests.mot_fixture import fixture_sequence
    from adaptive_vcm.v32.tracking import plan_sequences,read_sequence
    from adaptive_vcm.v33ap.sources import SourceRecord
    from adaptive_vcm.v34.tasks import Heads
    from adaptive_vcm.v34 import evaluation
    from adaptive_vcm.v34.protocol import ROOT

    class TinySwin(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.features=torch.nn.Sequential(torch.nn.Identity(),torch.nn.Identity())
        def forward(self,x):
            f=self.features(x.permute(0,2,3,4,1)); z=f.mean((1,2,3,4))
            return torch.stack((z,-z),1)
    class TinyAction:
        def __init__(self,name,device): self.model=TinySwin(); self.name=name; self.device=device
        def logits(self,x): return self.model(x)
        def probabilities(self,rgb):
            from adaptive_vcm.v32vc.train import tensor
            return self.logits(tensor(rgb,self.device)).softmax(-1)[0].detach().numpy()
    class TinyDetector(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.transform=SimpleNamespace(min_size=(320,),max_size=640)
            self.roi_heads=SimpleNamespace(detections_per_img=100)
        def forward(self,images):
            # Actual per-image function; one persistent pedestrian in fixture.
            return [dict(boxes=torch.tensor([[10.,5.,30.,25.]]),scores=torch.tensor([.9]),labels=torch.tensor([1])) for x in images]

    cfg=settings(ROOT/'configs/v34_arot_r1.json'); cfg.update(qps=[30,40,50],codecs=['h264'],frames=3,seeds=[303401])
    heads=Heads.__new__(Heads); heads.cfg=cfg; heads.device='cpu'; heads.experiment='ar_ot'
    heads.action=TinyAction('swin3d_t','cpu'); heads.detector=SimpleNamespace(model=TinyDetector(),name='fixture')
    heads.configure_detector()
    monkeypatch.setattr(evaluation,'ActionAnalyzer',TinyAction)
    groups={'dev':[],'test':[]}; plans=[]
    for split in groups:
        root=tmp_path/split; fixture_sequence(root)
        plan=plan_sequences(root,['MOT20-01'],3)[0]; plans.append(plan)
        rgb,g=read_sequence(plan,80)
        # read_sequence letterboxes to 80x80, therefore use original pixels to
        # keep fixture detector coordinates aligned with the official GT.
        rgb=rgb[:,20:60].copy(); g=(1,1,0,0)
        record=SourceRecord('ot/'+split,'ot',dict(sequence='MOT20-01',width=80,height=40),rgb,g,{})
        record.targets={'must_not_read':'held-out GT'}
        groups[split]=[record]
    # Same sequence name exists under two temporary roots; only scoring may read it.
    meta={'ot':[plans[0]]}
    model=Controller(13,conditions=9).eval()
    model.metadata={'seed':303401,'calibration':{'risk':[[0]*3]*13,'rate':[0]*13}}
    output=tmp_path/'result'; output.mkdir()
    evaluation.evaluate(groups,heads,[model],cfg,meta,output,__import__('pathlib').Path(os.environ['V34_TRACKEVAL']))
    summary=json.loads((output/'test_summary.json').read_text())
    assert summary['clean_source_tracking']['combined']['HOTA']>90
    assert summary['ot_quality_unit'].startswith('raw curves: HOTA percent')
    rows=json.loads((output/'test_rows.json').read_text())
    selected=[r for r in rows if r['arm']=='selected_303401']
    assert len(selected)==3
    assert all(r['deployed_encode_count']==len(r['tried_actions']) for r in selected)
    assert all(r['deployed_source_guard_seconds']>0 and r['shared_packet'] for r in selected)
    assert all(r['source_guard']['scope'].startswith('source consistency') for r in selected)
    assert any(r['arm']=='calibrated_303401' for r in rows)
    for row in selected:
        work=row['deployed_candidate_work']
        assert {w['action'] for w in work}==set(row['tried_actions'])
        assert row['deployed_preprocess_seconds']==pytest.approx(sum(w['preprocess_seconds'] for w in work))
        assert row['deployed_probe_decode_seconds']==pytest.approx(sum(w['decode_seconds'] for w in work))
        assert row['deployed_sender_seconds']>=row['source_seconds']+row['deployed_encode_seconds']+row['deployed_probe_decode_seconds']
        assert row['controller_inference_seconds']>0
