"""Reapply the frozen response to isolated signal/noise on source V only."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

project=Path.home()/'2510044040/CV-SincNet'
run='20261125-phase1-lora-w24-native-wiener-pilot-s2026092702-r01'
release=project/'releases'/(run+'-release')
runtime=json.loads((release/'runtime.json').read_text())
sys.path.insert(0,runtime['w24_release_root'])
from experiments import pamf_lora_main
pamf_lora_main.__path__.append(str(release/'experiments/pamf_lora_main'))
from experiments.pamf_lora_main.w24_native_wiener import denoise, noise_profile, torch_iq, PAD, STRIDE
from experiments.pamf_lora_main.w24_factor_audit import load_classifier, trace_components
from experiments.pamf_lora_main.w24_native_full import SOURCE, VIEWS, classify
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.runner import activate

def apply_response(x,gain):
    extended=torch.cat((x[:,1:PAD+1].flip((-1,)),x,x[:,-PAD-1:-1].flip((-1,))),dim=1)
    f=(torch.fft.fftfreq(extended.shape[-1])*STRIDE).remainder(STRIDE)
    i=f.floor().long()
    response=torch.lerp(gain[:,i],gain[:,(i+1)%STRIDE],f-i)
    return torch.fft.ifft(torch.fft.fft(extended)*response)[:,PAD:-PAD]

def by_band(wave,diag):
    pct={}
    freq=1e6*torch.fft.fftfreq(wave.shape[-1]).abs()
    x=torch.fft.fft(wave,norm='ortho')
    z=torch.fft.fft(diag,norm='ortho')
    for start,end in ((0,100000),(100000,200000),(200000,350000),(350000,500001)):
        band=(freq>=start)&(freq<end)
        power=x[:,band].abs().square().sum(1).clamp_min(1e-12)
        retained=z[:,band].abs().square().sum(1)/power
        pct[f'{start//1000}-{end//1000}kHz']=float(retained.median())
    return pct

activate(runtime)
source=Path(runtime['source_root'])
contract=source_contract(source)
classifier=load_classifier(runtime,contract,'cpu')
with np.load(source/'V_metadata.npz',allow_pickle=False) as file:
    assert file['ids'].tolist()==contract['role_ids']['V']
    rx=file['rx_i'].copy()
rng=np.random.default_rng(2026112567)
ids=np.concatenate([np.sort(rng.choice(np.flatnonzero(rx==i),20,replace=False)) for i in range(len(SOURCE))])
original=native(source,'V',contract['counts']['V'])[ids]
threshold=json.loads((project/'runs'/run/'calibration.json').read_text())['threshold']
predictions={}
evidence={}
for scene in (VIEWS[1],VIEWS[-1]):
    items=[trace_components(frame,scene,SOURCE[rx[i]],int(i),run+'-component-v')
           for frame,i in zip(original,ids,strict=True)]
    y,s,n=(torch_iq(np.stack(part).astype(np.complex64),'cpu') for part in zip(*items,strict=True))
    ratio,gain=noise_profile(y)
    active=ratio>threshold
    gain[~active]=1
    gs,gn=apply_response(s,gain),apply_response(n,gain)
    checked,_=denoise(y,threshold)
    assert torch.allclose(checked,gs+gn,rtol=2e-5,atol=6e-6)
    hs=torch.lerp(s,gs,.5); hn=torch.lerp(n,gn,.5)
    variants=dict(raw=y,no_noise=s,half_actual=torch.lerp(y,checked,.5),
                  half_signal_only=hs+n,half_noise_only=s+hn,half_signal_no_noise=hs)
    edit=torch.fft.fft(hs-s,norm='ortho')
    f=1e6*torch.fft.fftfreq(y.shape[-1]).abs()
    for low,high in ((0,100000),(100000,200000),(200000,350000),(350000,500001)):
        band=(f>=low)&(f<high)
        change=torch.fft.ifft(edit*band,norm='ortho')
        variants[f'signal_edit_{low//1000}_{high//1000}kHz']=y+change
    predictions[scene]={name:np.concatenate([classify(classifier,np.from_dlpack(batch.contiguous()),'cpu')
                                              for batch in wave.split(32)]) for name,wave in variants.items()}
    info=dict(active=int(active.sum()),true_signal_power_retained=by_band(s,hs),noise_power_retained=by_band(n,hn))
    evidence[scene]=info
with np.load(source/'V_metadata.npz',allow_pickle=False) as file:
    y=file['y'][ids]
for scene in predictions:
    evidence[scene]['correct']={k:int((a==y).sum()) for k,a in predictions[scene].items()}
print(json.dumps(dict(n=len(ids),threshold_unchanged=True,target_reads=0,score_after_predictions=True,scenes=evidence)))
