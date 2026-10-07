from fractions import Fraction
import importlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


def mod(): return importlib.import_module('adaptive_vcm.v31.gop')


@pytest.mark.parametrize('codec',['h264','h265'])
@pytest.mark.parametrize('keyint',[8,16,32])
def test_actual_64_frame_keyframes_and_packet_timing(codec,keyint,tmp_path):
    g=mod(); fps=Fraction(30000,1001)
    rgb=np.stack([np.full((16,16,3),(t*23)%256,np.uint8) for t in range(64)])
    settings=g.gop_settings(codec,keyint)
    assert settings=={'gop':keyint,'scenecut':False,'closed_gop':True,'bframes':0}
    packet,probe=g.encode_audit_packet(rgb,codec,40,fps,keyint,'medium',tmp_path)
    assert packet.recipe.duration==64/fps and packet.recipe.fps==fps
    assert packet.total_bytes==len(packet.encoded)+32
    assert packet.decoded.shape==(64,16,16,3)
    assert probe['keyframes']==list(range(0,64,keyint)) and probe['frames']==64
    assert 'B' not in probe['picture_types']
    windows=g.analyzer_windows(packet.decoded)
    assert len(windows)==3 and all(w.shape==(16,16,16,3) for w in windows)
    np.testing.assert_array_equal(windows[1],packet.decoded[16:48:2])


def test_gop_plan_rejects_od_and_invalid_or_short_windows(tmp_path):
    g=mod(); cfg=json.loads((Path(__file__).parents[1]/'configs/v31_gop.json').read_text())
    with pytest.raises(ValueError,match='AR|ar|OD'): g.gop_audit_plan([],dict(cfg,task='od'))
    for bad in (7,0,True):
        with pytest.raises(ValueError,match='GOP'): g.gop_settings('h264',bad)
    with pytest.raises(ValueError,match='64'): g.analyzer_windows(np.zeros((63,16,16,3),np.uint8))


def test_matched_gop_audit_quality_aggregates_windows_by_source(tmp_path):
    g=mod()
    from adaptive_vcm.codec import locate_ffmpeg
    from tests.test_v31_measure import models
    path=tmp_path/'source.mp4'
    rgb=np.zeros((64,128,128,3),np.uint8)
    subprocess.run([locate_ffmpeg(),'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','128x128',
        '-r','25','-i','pipe:0','-c:v','libx264','-bf','0','-threads','2','-pix_fmt','yuv420p',str(path)],input=rgb.tobytes(),check=True)
    cfg=json.loads((Path(__file__).parents[1]/'configs/v31_gop.json').read_text())
    cfg.update(qps=[40],codecs=['h264'],gops=[16],sizes=[112])
    plan=g.gop_audit_plan([{'id':'one','path':str(path),'label':1}],cfg)
    cfg['models']=models('ar')
    report=g.run_gop_audit(plan,cfg,tmp_path/'audit')
    assert report['schema']=='v31-gop-audit-1' and report['primary_oracle_eligible'] is False
    assert report['sources']==1 and report['primary_metric']=='source-averaged r2plus1d_18 Top1'
    assert report['latency']['same_gop_matched'] and report['latency']['cross_gop_relaxation_labelled']
    assert len(report['rows'])==2
    anchor,area=report['rows']
    assert anchor['preset']==area['preset']=='medium' and anchor['gop']==area['gop']==16
    assert anchor['duration']==area['duration']==[64,25]
    assert anchor['window_indices']==area['window_indices']==[list(range(s,s+32,2)) for s in (0,16,32)]
    assert all(r['aggregate_predictions']['r2plus1d_18']['top1']==1 for r in report['rows'])
    assert report['comparisons'][0]['reference']=='identity' and report['comparisons'][0]['gop']==16
    assert not (tmp_path/'audit'/'oracle_gate.json').exists()
