import json
import os
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from experiments import pamf_lora_main
release=Path(os.environ['EXPERIMENT_RELEASE'])
pamf_lora_main.__path__.append(str(release/'experiments/pamf_lora_main'))
from experiments.pamf_lora_main import w24_native_innovation as ni
from experiments.pamf_lora_main.full_packet_views import source_role
from experiments.pamf_lora_main.w24_factor_audit import trace_components
from experiments.pamf_lora_main.w24_native_full import SOURCE, native_full_view

rt=ni.read(release/'runtime.json')
source=Path(rt['source_root'])
device='cpu'
tracker,_=ni.source_tools(rt,device)
contract=ni.source_contract(source)
raw,rx,_,_=source_role(source,contract,'U_s')
excluded=np.random.default_rng(ni.SEED).choice(len(raw),ni.FIT_PACKETS+ni.HOLD_PACKETS,replace=False)
rng=np.random.default_rng(2026112701)
ids=np.concatenate([rng.choice(np.setdiff1d(np.flatnonzero(rx==name),excluded),20,replace=False) for name in SOURCE])
taps=ni.high_pass_taps(device,torch.float32)
namespace='w24-native-innovation-noise-isolation-v1'

def as_array(x):
    return np.asarray(x.tolist(),dtype=np.float64)

def center_corr(a,b):
    a=a-a.mean((-1,-2),keepdim=True)
    b=b-b.mean((-1,-2),keepdim=True)
    num=(a*b).sum((-1,-2))
    return num/((a.square().sum((-1,-2))*b.square().sum((-1,-2))).sqrt().clamp_min(1e-12))

def pair_coherence(a,b):
    return (a.conj()*b).sum((-1,-2)).abs()/(a.abs().square().sum((-1,-2))*b.abs().square().sum((-1,-2))).sqrt().clamp_min(1e-12)

result={}
with torch.inference_mode():
    for view in ('practical_high','practical_mid_urban','practical_low_urban'):
        powers,spaces,directions=[],[],[]
        for i in range(0,len(ids),10):
            keys=ids[i:i+10]
            args=[(raw[j],view,rx[j],int(j),namespace) for j in keys]
            observed,_,noise=zip(*(trace_components(*a) for a in args),strict=True)
            if i==0:
                assert np.array_equal(observed[0].astype(np.complex64),native_full_view(*args[0]))
            waveform=ni.packet_tensor(np.stack(observed).astype(np.complex64),device)
            _,frame,_,fields,_=ni.observation(tracker,waveform,taps)
            truth=tracker.aperture(ni.packet_tensor(np.stack(noise).astype(np.complex64),device))[...,:ni.CENTER]/frame
            local_power=ni.local_mean(truth.abs().square())
            true_log=local_power.clamp_min(1e-8).log()
            measured=[torch.exp(x[...,2,:]*8) for x in fields]
            direction=[torch.complex(x[...,0,:],x[...,1,:]) for x in fields]
            powers.append(torch.stack([local_power.mean((-1,-2)),*(v.mean((-1,-2)) for v in measured)],dim=-1))
            spaces.append(torch.stack([center_corr(v.clamp_min(1e-8).log(),true_log) for v in measured],dim=-1))
            directions.append(torch.stack([pair_coherence(x,truth) for x in direction],dim=-1))
        ps=as_array(torch.cat(powers))
        by_packet=[float(spearmanr(ps[:,0],ps[:,i]).statistic) for i in (1,2)]
        result[view]={'packet_noise_power_spearman':by_packet,
                      'within_packet_log_noise_power_pearson_median':np.median(as_array(torch.cat(spaces)),axis=0).tolist(),
                      'within_packet_noise_complex_coherence_median':np.median(as_array(torch.cat(directions)),axis=0).tolist()}
print(json.dumps({'status':'ANALYZED','source_packets':len(ids),'fit_pool_overlap':0,'source_truth_reads':0,
                  'target_reads':0,'oracle_scope':'isolated source U simulator noise only',
                  'ordered_features':['aperture','native'],'scenes':result},indent=2))
