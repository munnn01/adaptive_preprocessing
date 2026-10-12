"""Native weights_only-safe checkpoints, hashed state and JSON metadata."""
import hashlib
import json
from pathlib import Path
import torch
from ..v32vc.checkpoint import state_hash
from .model import Controller
from .protocol import digest
from . import SCHEMA

def save_checkpoint(path,model,metadata):
    # JSON roundtrip rejects non-native metadata (TorchVersion/NumPy scalars).
    meta=json.loads(json.dumps(metadata,allow_nan=False))
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(dict(schema=SCHEMA,actions=model.actions,conditions=model.conditions,metadata=meta,metadata_sha256=digest(meta),
                    state_sha256=state_hash(state),state=state),path)

def load_checkpoint(path,device='cpu'):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value.get('schema')!=SCHEMA or digest(value['metadata'])!=value['metadata_sha256']:
        raise ValueError('checkpoint metadata mismatch')
    if state_hash(value['state'])!=value['state_sha256']: raise ValueError('checkpoint state mismatch')
    model=Controller(value['actions'],value.get('conditions',6)); model.load_state_dict(value['state'],strict=True)
    model.to(device).eval(); model.metadata=value['metadata']
    model.checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return model
