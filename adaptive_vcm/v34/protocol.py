import hashlib
import json
from fractions import Fraction
from pathlib import Path
from . import SCHEMA

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('ar', 'od', 'ot')
# Rows are [object H,M; context H,M; background H,M]. Row zero is identity.
ACTIONS = (
    (0,0,0,0,0,0),
    (0,0,0,0,-.35,0), (0,0,0,0,-.65,-.15),
    (0,0,-.25,0,-.65,-.15), (0,0,-.4,-.1,-.75,-.25),
    (-.15,0,-.25,0,-.5,-.1), (-.25,0,-.4,-.1,-.65,-.2),
    (.15,0,0,0,-.5,-.1), (.25,.1,0,0,-.65,-.15),
    (0,0,.15,0,-.5,-.1), (.1,0,.1,0,-.35,0),
    (-.35,-.1,-.35,-.1,-.35,-.1), (.15,0,.15,0,0,0))

def settings(path=None):
    cfg = json.loads((Path(path) if path else ROOT/'configs/v34.json').read_text())
    if (cfg['schema'] != SCHEMA or cfg['codecs'] != ['h264','h265']
        or cfg['qps'] != [30,35,40,45,50] or cfg['seeds'] != [303401,303402,303403]
        or cfg['frames'] != 16 or cfg['preset'] != 'medium'):
        raise ValueError('V34 recipe drift')
    return cfg

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def codec_factory(task, codec, qp, cfg):
    from ..v33ap.codec import MeasuredCodec
    if task not in TASKS or codec not in cfg['codecs'] or qp not in cfg['qps']:
        raise ValueError('invalid V34 codec condition')
    return MeasuredCodec(codec,qp,cfg['preset'],Fraction(cfg['fps']),gop_settings={
        'gop':1 if task=='od' else cfg['frames'], 'bframes':0,'closed_gop':True,'scenecut':False})
