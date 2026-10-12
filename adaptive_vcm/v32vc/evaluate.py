"""Explicit V32-VC arms, measured alongside unchanged historical V32 families."""
import argparse
import time
from pathlib import Path
from ..v32.selection import Candidate
from ..v32.roi import build_grids,retarget
from ..v32.evaluate import run

EXTRA_FAMILIES = {
    'vc_plain': {'controls','vc_plain'},
    'vc_joint_plain': {'controls','vc_joint_plain'},
    'vc_retarget_only': {'controls','vc_retarget_only'},
    'vc_joint': {'controls','vc_joint_plain','vc_joint_retarget'},
}


def extra_candidates(rgb,support,codec,models,cfg):
    plain,joint = [models[a].pixels(rgb,support,codec) for a in ('plain','joint')]
    candidates = [Candidate('vc_plain_rgb',plain,'vc_plain'),Candidate('vc_joint_rgb',joint,'vc_joint_plain')]
    for factor in cfg['roi']['factors']:
        grid = build_grids(rgb.shape,support,factor,cfg['roi']['margin'])
        if grid:
            candidates.extend((Candidate(f'vc_raw_retarget_f{factor}',retarget(rgb,grid),'vc_retarget_only',grid),
                               Candidate(f'vc_joint_retarget_f{factor}',retarget(joint,grid),'vc_joint_retarget',grid)))
    return candidates


def prepare_extension(rgb,support,codec,models,cfg):
    start = time.perf_counter()
    candidates = extra_candidates(rgb,support,codec,models,cfg)
    return dict(extra_candidates=candidates,extra_families=EXTRA_FAMILIES),time.perf_counter()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=['ar','od']); parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--annotations',type=Path); parser.add_argument('--config',type=Path)
    parser.add_argument('--split',choices=['dev','test'],default='dev'); parser.add_argument('--count',type=int,default=32)
    parser.add_argument('--bootstrap',type=int,default=0); parser.add_argument('--device',choices=['cpu','cuda'])
    parser.add_argument('--checkpoint',type=Path,required=True); parser.add_argument('--v30-runtime',type=Path,required=True)
    parser.add_argument('--plain-checkpoint',type=Path,required=True); parser.add_argument('--joint-checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__': main()
