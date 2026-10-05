import numpy as np
import pytest

from adaptive_vcm.codec import StandardCodec, locate_ffmpeg, reference_bpp
from adaptive_vcm.stabilized_bank import (ACTION_NAMES, CORE_ACTION_NAMES,
    STABILIZED_ACTION_NAMES, build_stabilization_actions, build_stabilized_bank,
    max_pixel_change)
from adaptive_vcm.task_bank import ACTION_NAMES as BASE_ACTION_NAMES, build_task_bank


def source():
    rng = np.random.default_rng(26)
    y, x = np.mgrid[:64, :96]
    field = 110.+30.*np.sin(x/12.)+20.*np.cos(y/15.)
    frame = np.clip(field[..., None]+rng.normal(0, 4, (64,96,3)), 20, 230)
    clip = np.clip(np.stack([frame+b for b in (0,10,-10,6,-6,10,0,-10)]),0,255).round().astype(np.uint8)
    mask = np.zeros((64,96), np.float32)
    mask[24:40,40:56] = 1.
    return clip, mask


@pytest.mark.parametrize('qp', [0,30,40,45,50,51])
def test_new_actions_obey_pixel_bounds_and_protected_exactness(qp):
    clip, mask = source()
    before = clip.copy()
    bank = build_stabilization_actions(clip,mask,qp)
    assert tuple(c.name for c in bank) == STABILIZED_ACTION_NAMES
    for c in bank:
        assert c.clip.shape == clip.shape and c.clip.dtype == np.uint8
        assert not np.shares_memory(c.clip,clip)
        level = c.name.rsplit('_',1)[1]
        delta = np.abs(c.clip.astype(int)-clip.astype(int))
        assert delta.max() <= max_pixel_change(qp,level)
        if c.name in CORE_ACTION_NAMES:
            np.testing.assert_array_equal(c.clip[:,mask==1],clip[:,mask==1])
    np.testing.assert_array_equal(clip,before)


def test_old_bank_prefix_unchanged_and_new_rendering_deterministic():
    clip,mask=source()
    original=build_task_bank(clip,mask,50)
    extended=build_stabilized_bank(clip,mask,50)
    repeat=build_stabilized_bank(clip,mask,50)
    assert len(extended)==34 and tuple(c.name for c in extended)==ACTION_NAMES
    assert ACTION_NAMES[:18]==BASE_ACTION_NAMES
    for old,new in zip(original,extended):
        assert old.name==new.name
        np.testing.assert_array_equal(old.clip,new.clip)
    for a,b in zip(extended,repeat):
        np.testing.assert_array_equal(a.clip,b.clip)


@pytest.mark.parametrize('qp',[0,30,40,45,50,51])
def test_tiny_actions_append_after_existing_twelve_and_bound_each_channel(qp):
    clip,mask=source()
    bank=build_stabilization_actions(clip,mask,qp)
    assert tuple(c.name for c in bank[-4:]) == (
        'uniform_dc_tiny','core_dc_tiny','uniform_exposure_tiny','core_exposure_tiny')
    assert len(bank)==16 and max_pixel_change(qp,'tiny')==1
    for c in bank[-4:]:
        assert c.clip.shape==clip.shape
        assert np.abs(c.clip.astype(int)-clip.astype(int)).max()<=1
        if c.name.startswith('core_'):
            np.testing.assert_array_equal(c.clip[:,mask==1],clip[:,mask==1])


def test_flicker_reduction_keeps_spatial_edges_and_color_differences():
    clip,mask=source()
    c=next(c for c in build_stabilization_actions(clip,mask,50) if c.name=='uniform_exposure_strong')
    assert np.std(c.clip.mean((1,2,3))) < np.std(clip.mean((1,2,3)))
    # Uniform grayscale shift retains chroma differences exactly unless clipping;
    # this fixture stays away from clipping, so contrast is source-exact.
    np.testing.assert_array_equal(c.clip[...,0].astype(int)-c.clip[...,1].astype(int),
                                  clip[...,0].astype(int)-clip[...,1].astype(int))
    np.testing.assert_array_equal(np.diff(c.clip.astype(int),axis=1),np.diff(clip.astype(int),axis=1))


def test_scene_cut_prevents_future_and_previous_shot_exposure_leakage():
    clip,mask=source()
    second=np.clip(clip[:4].astype(int)+90,0,255).astype(np.uint8)
    whole=build_stabilization_actions(np.concatenate([clip,second]),mask,50)
    left=build_stabilization_actions(clip,mask,50)
    right=build_stabilization_actions(second,mask,50)
    for a,b,c in zip(whole,left,right):
        np.testing.assert_array_equal(a.clip[:len(clip)],b.clip)
        np.testing.assert_array_equal(a.clip[len(clip):],c.clip)


def test_full_protection_makes_every_core_action_identity():
    clip,mask=source()
    for c in build_stabilization_actions(clip,np.ones_like(mask),50):
        if c.name in CORE_ACTION_NAMES:
            np.testing.assert_array_equal(c.clip,clip)


def test_motion_normalization_cancels_global_exposure_but_blocks_local_motion():
    from adaptive_vcm.stabilized_bank import _luma_reference
    source_clip=np.full((3,16,24,3),100,dtype=np.float32)
    source_clip[1]+=10
    source_clip[2]+=20
    source_clip[1:,4:8,4:8]+=30
    _,_,_,gate=_luma_reference(source_clip)
    assert gate[1,0,0] == 1 and gate[1,5,5] == 0


@pytest.mark.parametrize('qp',[True,-1,52,40.5])
def test_invalid_qp(qp):
    clip,mask=source()
    with pytest.raises(ValueError,match='QP'):
        build_stabilization_actions(clip,mask,qp)


@pytest.mark.parametrize('bad',[np.nan,-.01,1.01])
def test_invalid_mask(bad):
    clip,mask=source()
    mask[0,0]=bad
    with pytest.raises(ValueError,match='protection'):
        build_stabilization_actions(clip,mask,50)


@pytest.mark.codec
@pytest.mark.parametrize('codec',['h264','h265'])
@pytest.mark.parametrize('qp',[40,45,50])
def test_same_geometry_actual_stream_and_original_accounting(codec,qp):
    if locate_ffmpeg() is None:
        pytest.skip('FFmpeg unavailable')
    clip,mask=source()
    candidate=next(c for c in build_stabilization_actions(clip,mask,qp) if c.name=='core_exposure_medium')
    result=StandardCodec(codec,qp).roundtrip(candidate.clip)
    assert result.coded_bytes > 0 and result.decoded.shape==clip.shape
    assert reference_bpp(result.coded_bytes,clip.shape)==8*result.coded_bytes/np.prod(clip.shape[:3])
