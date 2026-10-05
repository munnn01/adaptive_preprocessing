from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from adaptive_vcm.codec import StandardCodec
from adaptive_vcm.task_bank import build_task_bank
from adaptive_vcm.stabilized_bank import build_stabilization_actions,max_pixel_change

def texture(block):
 rng=np.random.default_rng(25)
 frame=rng.integers(40,210,(128//block,128//block,3),dtype=np.uint8)
 frame=np.repeat(np.repeat(frame,block,0),block,1)
 clip=np.stack([np.clip(frame.astype(float)+rng.normal(0,2,frame.shape),0,255).round().astype(np.uint8) for _ in range(16)])
 clip[:,48:80,48:80]=[232,48,16]
 mask=np.zeros((128,128),np.float32);mask[40:88,40:88]=1
 return clip,mask

def flicker():
 rng=np.random.default_rng(26)
 y,x=np.mgrid[:128,:128]
 field=115.+30.*np.sin(x/12.)+20.*np.cos(y/15.)
 frame=np.clip(field[...,None]+rng.normal(0,4,(128,128,3)),20,230)
 offsets=(0,10,-10,6,-6,10,0,-10,0,10,-10,6,-6,10,0,-10)
 clip=np.clip(np.stack([frame+b for b in offsets]),0,255).round().astype(np.uint8)
 mask=np.zeros((128,128),np.float32);mask[40:88,40:88]=1
 return clip,mask

parser=argparse.ArgumentParser(description='Actual-codec synthetic diagnostic; no AR teacher or quality result')
parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/stabilized_probe.json')
args=parser.parse_args()
rows=[]
for name,(clip,mask) in [('texture1',texture(1)),('texture2',texture(2)),('low_frequency_flicker',flicker())]:
 for codec in ['h264','h265']:
  for qp in [40,45,50]:
   encoder=StandardCodec(codec,qp);anchor=encoder.roundtrip(clip)
   record=dict(source=name,source_sha256=hashlib.sha256(clip.tobytes()).hexdigest(),codec=codec,qp=qp,anchor_bytes=anchor.coded_bytes)
   for label,bank in [('v25',build_task_bank(clip,mask,qp)[1:]),('stabilization',build_stabilization_actions(clip,mask,qp))]:
    results=[]
    for c in bank:
     resized=np.stack([cv2.resize(f,(128,128)) for f in c.clip]);mae=float(np.abs(resized.astype(float)-clip.astype(float)).mean())
     encoded=encoder.roundtrip(c.clip)
     results.append(dict(name=c.name,coded_bytes=encoded.coded_bytes,source_edit_mae=mae,rate_change_pct=100*(encoded.coded_bytes/anchor.coded_bytes-1),stream_sha256=hashlib.sha256(encoded.data).hexdigest()))
    record[label+'_actions']=results
    mild=[r for r in results if r['source_edit_mae']<=10]
    record[label+'_mild_saving_actions']=sum(r['coded_bytes']<=.99*anchor.coded_bytes for r in mild)
    record[label+'_best_mild_rate_change_pct']=min([0.]+[r['rate_change_pct'] for r in mild])
   rows.append(record)
   print(json.dumps({k:v for k,v in record.items() if not isinstance(v,list)}),flush=True)
out=args.out
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(scope='Synthetic only; actual medium H.264/H.265 QP40/45/50 on three16x128x128 sources,25fps. No teacher or ARquality.',teacher_feasibility=None,dev_score=None,bd_rate=None,top1=None,mild_definition='Source-space RGB MAE<=10 diagnostic; same slice for old and new actions',rows=rows),indent=2),encoding='utf-8')
