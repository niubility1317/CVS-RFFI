import json
from types import SimpleNamespace

import numpy as np
import torch

from experiments.pamf_lora_main import gain_dura_screen as screen
from experiments.pamf_lora_main.gain_dura_screen import (
    COUNT, MODES, SCREEN_VIEWS, WiSigRealGain, score_panel, screen_rows,
)
from experiments.pamf_lora_main.gain_dura_fusion import fuse


class Identity:
    def __call__(self, x):
        return x

    adjoint = __call__


def test_fusion_uses_the_companion_in_a_new_direction_and_caps_edit():
    observed=torch.zeros(2,16,dtype=torch.complex64)
    anchor=observed.clone()
    anchor[0,0]=1+1j
    companion=anchor.clone()
    companion[:,0]+=3j
    companion[:,1]=4
    estimate=fuse(observed,anchor,companion,Identity(),torch.ones(2))
    delta=estimate-anchor
    assert 0<delta[0,1].real<4 and delta[0,0].abs()<1e-6
    assert delta[0].norm()<=.25*anchor[0].norm()+1e-6
    assert torch.equal(estimate[1],anchor[1])


def test_single_receiver_panel_is_stable_and_does_not_select_excluded_rows():
    ids=np.array([str(i) for i in range(3*COUNT)])
    rx=np.repeat((1,2,3),COUNT)
    rows=screen_rows(ids,rx,2,excluded=[COUNT],limit=COUNT-1)
    assert len(rows)==COUNT-1 and np.all(rx[rows]==2) and COUNT not in rows
    assert np.array_equal(rows,screen_rows(ids,rx,2,excluded=[COUNT],limit=COUNT-1))


def test_clean_noninferiority_is_required_even_when_paired_gain_is_clear():
    labels=np.zeros(COUNT,dtype=np.int16)
    predictions=np.zeros((len(SCREEN_VIEWS),len(MODES),COUNT),dtype=np.int16)
    predictions[1,:-1,:24]=1
    assert score_panel(predictions,labels)['promote']
    predictions[0,-1,0]=1
    assert not score_panel(predictions,labels)['promote']


def test_wisig_real_gain_uses_the_same_shape_as_its_observation():
    x=torch.randn(2,256,dtype=torch.complex64)
    model=WiSigRealGain()
    p,r=model(x)
    torch.testing.assert_close(r,p,rtol=0,atol=0)
    assert r.shape==x.shape and r.is_complex()


def test_recovery_scores_only_frozen_source_panels_with_separate_bindings(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    prior=tmp_path/screen.RUN
    root=tmp_path/screen.SCORE_RUN
    prior.mkdir()
    root.mkdir()
    screen.write(prior/'state.json',{'status':'FAILED'})
    lora_release=tmp_path/'ref'
    lora_run=tmp_path/'original'
    lora_source=tmp_path/'lora_source'
    lora_release.mkdir()
    lora_source.mkdir()
    (lora_run/'full_native').mkdir(parents=True)
    screen.write(lora_release/'runtime.json',{'source_root':str(lora_source),'run_root':str(lora_run)})
    lora_ids=np.array([f'lora{i}' for i in range(COUNT+300)])
    screen.write(lora_source/'contract.json',{'role_ids':{'V':lora_ids.tolist()}})
    screen.write(lora_run/'full_native'/'source-predictions.json',
                 dict(status='PREDICTIONS_FROZEN',source_truth_reads=0,target_reads=0,physical_ids=lora_ids[:300].tolist()))
    np.savez(lora_source/'V_metadata.npz',ids=lora_ids,rx_i=np.zeros(len(lora_ids),dtype=int),y=np.ones(len(lora_ids),dtype=int))
    wifi_source=tmp_path/'wifi_source'
    (wifi_source/'V').mkdir(parents=True)
    np.save(wifi_source/'V'/'labels.npy',np.zeros(COUNT,dtype=int))
    wifi_ids=np.array([f'wifi{i}' for i in range(COUNT)])
    handles=lambda _: (None,None,wifi_source,None,wifi_ids,np.ones(COUNT,dtype=int),np.arange(COUNT))
    monkeypatch.setattr(screen,'wifi_context',
        lambda rt:(SimpleNamespace(tri=SimpleNamespace(core=SimpleNamespace(handles=handles))),{}))
    for dataset,ids,desired in (('lora',lora_ids[300:],1),('wisig',wifi_ids,0)):
        folder=prior/dataset
        folder.mkdir()
        predictions={v:np.full((len(MODES),COUNT),desired,dtype=np.int16) for v in SCREEN_VIEWS}
        screen.freeze(folder,dataset,ids,predictions)
    screen.independent_score(dict(run_root=str(root),score_from=str(prior),lora_ref=str(lora_release)))
    results=json.loads((root/'scores.json').read_bytes())
    assert results['truth_last'] and not results['target_reads']
    for d in ('lora','wisig'):
        assert results['results'][d]['correct']['clean']['fusion']==COUNT
        assert not results['results'][d]['promote']
    assert not (prior/'scores.json').exists()
