"""Measure frozen LoRa waveform error tails on the already-scored source panel."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
release=root.parent.parent/'releases'/(root.name+'-release')
sys.path.insert(0,str(release))

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import proof
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.denoise_source import replay
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_role, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.packet_physics import aligned_view
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.tail_pilot_source import MODES, RECIPES

assert read(root/'state.json')['status']=='SOURCE_SCORED'
routes=('full','residual')
frozen={route:read_frozen(root,route,MODES) for route in routes}
physical=frozen['full'][0]
assert frozen['residual'][0]==physical
rt=read(release/'runtime.json')
activate(rt)
torch.set_num_threads(2)
source=Path(rt['source_root'])
contract=source_contract(source)
wi,rx,ids,_=source_role(source,contract,'V')
idx={p:i for i,p in enumerate(ids.tolist())}
chosen=np.array([idx[p] for p in physical]).reshape(len(SOURCE),-1)[:,3::6].flatten()
rows=[(wi[i],rx[i],ids[i]) for i in chosen]
assert [sum(rx[chosen]==key) for key in SOURCE]==[30]*len(SOURCE)
cache=np.load(root.parent/DATA_RUN/'source_views'/'V.npy',mmap_mode='r',allow_pickle=False)
tracker=PacketRestorer('consensus').eval()
models={}
limits=[]
for name,recipe in RECIPES.items():
    saved=torch.load(root/f'{name}-model.pt',map_location='cpu',weights_only=True)
    assert saved['proof']==proof(contract,recipe)
    model=recipe.model().eval()
    model.load_state_dict(saved['weights'],strict=True)
    models[name]=model
    limits.append(saved['threshold'])
assert limits[0]==limits[1]
freq=torch.fft.fftfreq(256,d=1/256).abs()
bands=((1,8),(9,32),(33,64),(65,128))
select=(freq>0)
quant=torch.tensor((.50,.90,.95))

def fft(x):
    return torch.fft.fft(x,norm='ortho')

def power(x):
    return x.abs().square()

@torch.no_grad()
def readouts(wave, noise):
    original, rot, quality, rms=coordinates(tracker,wave)
    active=corruption_score(original)>limits[0]
    recovered={'direct':original}
    for name,model in models.items():
        coarse,fine=model(original,quality,rot)
        for key,x in ((name+'_coarse',coarse),(name,fine),(name+'_half',torch.lerp(original,fine,.5))):
            recovered[key]=torch.where(active[:,None,None],x,original)
    # The exact added noise is only used starting here. Both IQ restorers above
    # receive the original production arguments only.
    e=fft(complex_iq(noise)/(rot*rms))
    target=fft(original)-e
    return active, target, e, {key:fft(x) for key,x in recovered.items()}


def measure(parts):
    active=torch.cat([chunk[0] for chunk in parts])
    target=torch.cat([chunk[1] for chunk in parts])
    noise=torch.cat([chunk[2] for chunk in parts])
    candidates={name:torch.cat([chunk[3][name] for chunk in parts]) for name in parts[0][3]}
    m=len(active)
    noise_energy=power(noise[...,select]).sum(-1).clamp_min(1e-8)
    residual={name:power(z[...,select]-target[...,select]).sum(-1)/noise_energy
              for name,z in candidates.items()}
    over_active=active[:,None].expand_as(noise_energy)
    # Fix the worst tenth by the matching average-loss baseline on the active
    # chirps, then measure both arms on these exact same chirps.
    ref=residual['average'][over_active]
    tail=ref.topk((len(ref)+9)//10).indices if len(ref) else torch.empty(0,dtype=torch.long)
    outputs={}
    for mode,z in candidates.items():
        bounds={'all':residual[mode].flatten().quantile(quant).tolist()}
        district={}
        if active.any():
            active_error=residual[mode][over_active]
            bounds['active']=active_error.quantile(quant).tolist()
            bounds['fixed_mean_arm_tail']=active_error[tail].quantile(quant).tolist()
            bounds['matched_tail_delta_median']=(active_error[tail]-ref[tail]).median().item()
            for lo,hi in bands:
                k=(freq>=lo)&(freq<=hi)
                reference=target[active][...,k]
                judged=z[active][...,k]
                e=power(reference).sum(-1).clamp_min(1e-8)
                j=power(judged).sum(-1).clamp_min(1e-8)
                district[f'{lo}-{hi}']=dict(
                    complex_nmse=(power(judged-reference).sum(-1)/e).quantile(quant).tolist(),
                    magnitude_nmse=((judged.abs()-reference.abs()).square().sum(-1)/e).quantile(quant).tolist(),
                    real_correlation=((judged*reference.conj()).real.sum(-1)/(e*j).sqrt()).quantile(quant).tolist())
        outputs[mode]=dict(relative_to_added_noise=bounds, active_bands=district)
    return dict(n=m, active=int(active.sum()), fixed_tail_chirps=len(tail), arms=outputs)


report={}
for route in routes:
    report[route]={}
    for scene in ('practical_high','practical_low_urban'):
        with ProcessPoolExecutor(max_workers=4) as pool:
            values=list(pool.map(partial(replay,route,scene),rows,chunksize=8))
        iq=np.stack([pair[0] for pair in values])
        noise=np.stack([pair[1] for pair in values])
        assert iq.shape==noise.shape
        if route=='full':
            assert np.array_equal(iq,cache[VIEWS.index(scene),chosen])
        else:
            for k in range(0,len(rows),30):
                w,r,p=rows[k]
                assert np.array_equal(iq[k],aligned_view(w,scene,r,p,'source-V',processing_route=route))
        a,b=tensor(iq,'cpu'),tensor(noise,'cpu')
        chunks=[readouts(x,y) for x,y in zip(a.split(15),b.split(15),strict=True)]
        report[route][scene]=measure(chunks)

print(json.dumps(dict(status='VERIFIED',TX_reads=0,target_reads=0,source_isolated_noise=True,
                      paired_physical_count=len(rows),metrics=report),indent=2))
