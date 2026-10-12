import copy
import numpy as np
import pytest
from adaptive_vcm.v34.sources import split_training
from adaptive_vcm.v34.runner import measure_candidates, validate_cache
from adaptive_vcm.v34.protocol import codec_factory, settings
from adaptive_vcm.data import partition
import torch

def test_training_calibration_split_rejects_held_out_and_duplicates():
    items=[{'id':str(i)} for i in range(300) if partition('coco2017/'+str(i))=='train'][:64]
    fit,cal=split_training(items,'od',settings())
    assert len(fit)==48 and len(cal)==16
    assert not {r['id'] for r in fit}&{r['id'] for r in cal}
    bad=copy.deepcopy(items); bad[-1]=bad[0]
    with pytest.raises(ValueError): split_training(bad,'od',settings())
    bad[-1]={'id':next(str(i) for i in range(300) if partition('coco2017/'+str(i))=='test')}
    with pytest.raises(ValueError): split_training(bad,'od',settings())

@pytest.mark.codec
@pytest.mark.parametrize('codec',['h264','h265'])
def test_actual_codec_table_pays_header_and_shares_decoded_pixels(codec):
    rgb=np.random.default_rng(2).integers(0,256,(1,24,24,3),dtype=np.uint8)
    maps=np.zeros(rgb.shape,np.float32); maps[:,:,:,2]=1
    calls=[]
    def score(decoded):
        calls.append(decoded); return [None,float(decoded.mean()),None]
    table=measure_candidates(rgb,maps,[(0,)*6,(0,0,0,0,-.5,0)],
        codec_factory('od',codec,35,settings()),'od',score)
    assert len(calls)==2 and table['rate'][0]==1 and table['damage'][0]==[None,0.,None]
    assert table['packets'][0]['bytes']>table['packets'][0]['elementary_bytes']>0
    assert table['packets'][1]['q8_edits']>0
    assert all(p['decode_seconds']>0 for p in table['packets'])

def test_cache_validation_binds_source_pixels_and_prevents_test_supervision():
    from adaptive_vcm.v34.protocol import digest
    cache={'schema':'adaptive-vcm-v34-1','experiment':'ar_od','origin':{'commit':'a'*40},
           'sources':{'ar/a':{'split':'train','pixel_sha256':'b'*64}},
           'rows':[{'id':'ar/a','split':'train','rate':[1.],'damage':[[0.,None,None]],'mask':[1,0,0]}]}
    cache['sha256']=digest(cache)
    validate_cache(cache)
    cache['sources']['ar/a']['split']='test'
    with pytest.raises(ValueError): validate_cache(cache)

def test_selection_measures_real_bytes_and_falls_back_without_ground_truth():
    from adaptive_vcm.v34.evaluation import choose_packet
    from adaptive_vcm.v32.codec import Packet
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(()))
            self.metadata={'calibration':{'risk':[[0]*3]*2,'rate':[0,0]}}
        def forward(self,x,m,c):
            return dict(logits=torch.tensor([[0.,1.]]),risk=torch.tensor([[[0.,0.,0.],[0.,-.2,0.]]]),rate=torch.tensor([[1.,.8]]))
    rgb=np.zeros((1,8,8,3),np.uint8)
    anchor=Packet(b'a'*100,rgb,rgb.shape,60,40,0.)
    oversized=Packet(b'b'*102,rgb,rgb.shape,62,40,0.)
    pair=(torch.zeros(1,3,8,8),torch.ones(1,3,8,8)/3)
    forced,selected,packet,details=choose_packet(Model(),pair,'h264',35,[0,1,0],settings(),anchor,lambda a:oversized)
    assert forced==1 and selected==0 and packet is anchor
    assert details['rate_fallback'] and details['proposed_action']==1
