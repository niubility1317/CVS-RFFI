from pathlib import Path
import sys
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from experiments.pamf_reception_restore.models import (ComplexTFUNet, MultiscaleDiffusion,
    MultiscaleFlow, ReceptionUNet, mirror_spectrum)
from experiments.pamf_reception_restore.data import route_for,route_array,shared_pair

torch.set_num_threads(2)


def test_negative_frequency_conjugate_is_the_transform_of_time_conjugation():
    value=torch.complex(torch.randn(3,256),torch.randn(3,256))
    assert torch.allclose(mirror_spectrum(torch.fft.fft(value)),torch.fft.fft(value.conj()),atol=1e-5)


@pytest.mark.parametrize('factory',[ReceptionUNet,ComplexTFUNet])
def test_deterministic_initial_identity_preserves_unclipped_IQ_and_learns(factory):
    model=factory()
    value=torch.randn(2,2,256)*3
    assert torch.equal(model(value),value)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        loss=(model(value)-value*.8).square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert not torch.equal(model(value),value)


@pytest.mark.parametrize('factory',[MultiscaleDiffusion,MultiscaleFlow])
def test_complete_generation_reuses_current_multiscale_condition_and_preserves_state(factory,monkeypatch):
    model=factory().eval()
    encoder=model.conditioner if isinstance(model,MultiscaleDiffusion) else model.network.conditioner
    original=encoder.forward;calls=[]
    def count(value):calls.append(value.detach().clone());return original(value)
    monkeypatch.setattr(encoder,'forward',count)
    value,noise=torch.randn(2,2,256),torch.randn(2,2,256)
    before={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():result=model(value,noise)
    assert result.shape==value.shape and torch.isfinite(result).all()
    assert len(calls)==1 and torch.equal(calls[0],value)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def test_native_full_and_residual_use_same_physical_seed_without_composition():
    rng=np.random.default_rng(7)
    clean=rng.normal(size=(2,256)).astype(np.float32)
    r,f,meta=shared_pair(clean,'practical_mid','source-id','source-session',41,1,52)
    assert r.shape==f.shape==clean.shape and not np.allclose(r,f)
    r2,f2,meta2=shared_pair(clean,'practical_mid','source-id','source-session',41,1,52)
    assert np.array_equal(r,r2) and np.array_equal(f,f2) and meta==meta2
    r3,f3,_=shared_pair(clean,'practical_mid','source-id','source-session',41,2,52)
    assert not np.array_equal(r,r3) and not np.array_equal(f,f3)
    assert route_for('unet_residual',.9)=='residual'
    assert route_for('unet_full',.6)=='full'
    assert route_for('unet_mixed',.1)=='clean'
    assert route_for('unet_mixed',.7)=='residual'
    assert route_for('unet_mixed',.8)=='full'


@pytest.mark.parametrize('factory',[MultiscaleDiffusion,MultiscaleFlow])
def test_conditioner_receives_training_gradient_after_zero_projection_starts_learning(factory):
    model=factory()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    clean,received=torch.randn(2,2,256),torch.randn(2,2,256)
    for _ in range(2):
        optimizer.zero_grad()
        estimate,velocity=model.training_prediction(clean,received,torch.Generator().manual_seed(17))
        loss=velocity+.2*(estimate-clean).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    encoder=model.conditioner if isinstance(model,MultiscaleDiffusion) else model.network.conditioner
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in encoder.parameters())


def test_actual_training_composition_and_LR_match_registered_budget():
    from experiments.pamf_reception_restore.runner import learning_rate,pair_loss
    quantile=np.array([.01,.149,.15,.749,.75,.999])
    assert route_array('unet_mixed',quantile).tolist()==['clean','clean','residual','residual','full','full']
    assert route_array('unet_residual',quantile).tolist()==['clean','clean','residual','residual','residual','residual']
    assert route_array('unet_full',quantile).tolist()==['clean','clean','full','full','full','full']
    assert learning_rate(0)>0 and learning_rate(246)==pytest.approx(2e-4)
    assert learning_rate(4939)==pytest.approx(2e-5)
    estimate=torch.randn(2,2,256,requires_grad=True)
    loss,terms=pair_loss(estimate,torch.zeros_like(estimate))
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert torch.isfinite(estimate.grad).all()


def test_new_full_source_view_replays_existing_residual_namespace_exactly():
    from experiments.pamf_restoration.data import corrupt_one
    clean=np.random.default_rng(1).normal(size=(2,256)).astype(np.float32)
    scene,pid,session='practical_high','source-id','rx1:day1:eq1'
    old=corrupt_one((0,clean,scene,pid,session,'validation',2026101901))[1]
    residual,_,_=shared_pair(clean,scene,pid,session,2026101901,0,2026101902,'restoration/validation/draw0')
    assert np.array_equal(old,residual)


def test_waveform_selection_rejects_classification_fields_and_checkpoint_rejects_route_swap(tmp_path):
    from experiments.pamf_reception_restore.controller import waveform_choice
    from experiments.pamf_reception_restore.models import NAMES
    from experiments.pamf_reception_restore.runner import save_checkpoint,load_checkpoint
    rows=[dict(candidate=name,clean_nmse=.01,mean_residual_nmse=.2+index*.01) for index,name in enumerate(NAMES)]
    assert waveform_choice(rows)=='unet_residual'
    rows[0]['TX_accuracy']=100
    with pytest.raises(ValueError,match='TX/F'):waveform_choice(rows)
    model=ReceptionUNet();optimizer=torch.optim.AdamW(model.parameters())
    path=tmp_path/'real-checkpoint.pt'
    save_checkpoint(path,model,optimizer,1,1,'unet_residual')
    restored,_=load_checkpoint(path,'unet_residual','cpu')
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError,match='provenance/config'):load_checkpoint(path,'unet_full','cpu')


def test_new_checkpoint_loader_is_wired_through_original_seven_view_prediction(tmp_path,monkeypatch):
    from experiments.pamf_restoration import runner as old
    from experiments.pamf_restoration import frozen
    from experiments.pamf_restoration.data import SCENES
    class Reader(torch.nn.Module):
        def features(self,value):return value.mean((1,2))[:,None].expand(-1,160)
        def classify_features(self,feature):return feature[:,:6]
    monkeypatch.setattr(frozen,'load_readers',lambda *args,**kwargs:{'AsF':Reader()})
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    def forbidden(*args,**kwargs):raise AssertionError('Old-schema checkpoint loader must not run')
    monkeypatch.setattr(old,'load_checkpoint',forbidden)
    reference=tmp_path/'source_reference';reference.mkdir()
    folder=tmp_path/'row';folder.mkdir()
    for view in ('clean',*SCENES):
        np.save(reference/(view+'.npy'),np.ones((2,2,256),np.float32),allow_pickle=False)
    np.save(reference/'physical_ids.npy',np.array(['source-a','source-b']),allow_pickle=False)
    called=[]
    def loader(path,name,device):
        called.append((path,name,device));return ReceptionUNet().eval(),dict(epoch=20)
    old.predict_row('unet_mixed',reference,folder,'cpu',tmp_path,tmp_path,checkpoint_loader=loader)
    assert len(called)==1 and called[0][1]=='unet_mixed'
    import json
    done=json.loads((folder/'prediction/complete.json').read_text())
    assert done['count']==2 and len(done['views'])==7 and done['TX_truth_reads']==0
