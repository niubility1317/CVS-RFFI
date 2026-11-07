import torch
import pytest
import numpy as np
from experiments.phase1_dura_orthogonal.source_multiview_teacher import build_leave_one_view_teacher


def test_leaveout_teacher_has_correct_view_and_record_alignment():
    clean=torch.randn(4,2,256).repeat(3,1,1)
    anchor=torch.cat([torch.full((4,2,256),float(i)) for i in (1,3,9)])
    teacher=build_leave_one_view_teacher(anchor,clean,4)
    assert torch.equal(teacher[:4],torch.full((4,2,256),6.))
    assert torch.equal(teacher[4:8],torch.full((4,2,256),5.))
    assert torch.equal(teacher[8:],torch.full((4,2,256),2.))
    changed=anchor.clone()
    changed[:4]+=100
    assert torch.equal(build_leave_one_view_teacher(changed,clean,4)[:4],teacher[:4])
    clean[4,0,0]+=1
    with pytest.raises(ValueError,match='same clean'):
        build_leave_one_view_teacher(anchor,clean,4)


def test_ground_requires_matching_leaveout_source_contract(monkeypatch,tmp_path):
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
    model=DuraMultiscaleSpectral(width=32,taps=9)
    common=dict(commit='source-commit',epochs=5,architecture='dura_multiscale',
        objective='wide_multiview_dura_guard',TX_labels_read=0,F_weights_read=0,F_outputs_read=0,
        V_reads=0,target_reads=0,DURA_frozen=True,source_fit_size=32768,DURA_training_reads=1,
        source_leaveout_teacher_views=2,source_leaveout_teacher_weight=.5,
        source_leaveout_teacher='other_two_dura_mean',leaveout_teacher_inference_reads=0,
        adaptive_filter_taps=9,noise_gate_low=ground.GATE_LOG10_LOW,
        noise_gate_high=ground.GATE_LOG10_HIGH,spectral_freq_bins=17)
    summary=dict(status='SOURCE_TRAINED',fit_ids=32768,**common)
    state=dict(schema=ground.SCHEMA,fit_ids=np.arange(32768),width=32,model=model.state_dict(),**common)
    monkeypatch.setattr(ground.core,'read',lambda _:summary)
    monkeypatch.setattr(ground.torch,'load',lambda *args,**kwargs:state)
    monkeypatch.setattr(ground,'frozen_dura',lambda *args:model)
    monkeypatch.setattr(ground.core,'teacher',lambda *args:(model,None))
    loaded,_,_=ground.load_fixed(tmp_path,'source-run','source-commit')
    assert isinstance(loaded,DuraMultiscaleSpectral)
    state['source_leaveout_teacher_views']=3
    with pytest.raises(ValueError,match='leave-one-view'):
        ground.load_fixed(tmp_path,'source-run','source-commit')
