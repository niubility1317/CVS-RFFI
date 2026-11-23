"""Audit frozen LoRa RF PSD confidence on source practical IQ with isolated noise reference."""
from pathlib import Path
import json
import sys

root=Path(sys.argv[1])
release=root.parent.parent/'releases'/(root.name+'-release')
sys.path.insert(0,str(release))

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.denoise_source import replay
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq
from experiments.pamf_lora_main.full_packet_publish import DATA_RUN
from experiments.pamf_lora_main.full_packet_views import source_role, source_contract
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.packet_physics import aligned_view
from experiments.pamf_lora_main.rf_noise_confidence import RFNoiseConfidence
from experiments.pamf_lora_main.rf_noise_transport import RFNoiseTransport
from experiments.pamf_lora_main.rf_noise_transport_source import CONFIDENCE_MODES
from experiments.pamf_lora_main.runner import activate

assert read(root/'state.json')['status']=='SOURCE_SCORED'
frozen={route:read_frozen(root,route,CONFIDENCE_MODES) for route in ('full','residual')}
ids=frozen['full'][0]
assert frozen['residual'][0]==ids
rt=read(release/'runtime.json')
activate(rt)
source=Path(rt['source_root'])
reference, rx, source_ids, _=source_role(source,source_contract(source),'V')
position={label:i for i,label in enumerate(source_ids.tolist())}
indices=np.array([position[label] for label in ids])
sample=indices.reshape(len(SOURCE),-1)[:,8::12].flatten()
cache=np.load(root.parent/DATA_RUN/'source_views'/'V.npy',mmap_mode='r',allow_pickle=False)
raw=[(reference[i],rx[i],source_ids[i]) for i in sample]
gate=read(root/'confidence-complete.json')['threshold']
tracker=PacketRestorer('consensus').eval()
kernel=RFNoiseConfidence(False).eval()
confidence=RFNoiseConfidence(True).eval()
flat=RFNoiseTransport(False).eval()
grid=torch.tensor([.10,.50,.90])
summary={}
torch.set_num_threads(2)

def summarize(y,epsilon):
    observed, rotation, _, rms=coordinates(tracker,y)
    truth=kernel.analyze(complex_iq(epsilon)/(rotation*rms)).abs().square()
    # Pool six simultaneous frequency grids before shape comparison.
    samples=truth.mean(dim=1,keepdim=True)
    samples=F.avg_pool2d(samples, (3,3),stride=1,padding=1,count_include_pad=False)
    actual=samples/samples.mean(dim=-1,keepdim=True).clamp_min(1e-8)
    m0=flat.estimate_noise(observed,rotation)
    m1=kernel.estimate_noise(observed,rotation)
    m2, q=confidence.estimate_with_confidence(observed,rotation)
    active=corruption_score(observed)>gate

    def norm(p):
        return p/p.mean(dim=-1,keepdim=True).clamp_min(1e-8)

    all_groups={}
    for label,selected in [('all',torch.ones_like(active)),('active',active)]:
        if not selected.any():
            continue
        estimates={}
        for name,estimate in [('flat',m0),('full_color',m1),('confidence',m2)]:
            mean=estimate.mean(1,keepdim=True)
            error=(norm(mean)-actual).square().mean((1,2,3))
            estimates[name]=error[selected].quantile(grid).tolist()
        scores=q[selected]
        all_groups[label]=dict(n=len(scores), no_color=int((scores==0).sum()),
                               confidence=scores.quantile(grid).tolist(),
                               estimated_to_actual_power=(m0.mean((1,2,3))/truth.mean((1,2,3)))[selected].quantile(grid).tolist(),
                               shape_error=estimates)
    return all_groups

with torch.no_grad():
    for route in ('full','residual'):
        studies={}
        for scene in ('practical_high','practical_low_urban'):
            items=[replay(route,scene,row) for row in raw]
            received=np.stack([item[0] for item in items])
            noise=np.stack([item[1] for item in items])
            if route=='full':
                assert np.array_equal(received,cache[VIEWS.index(scene),sample])
            else:
                for k in range(0,len(raw),len(raw)//len(SOURCE)):
                    signal,rx_id,pid=raw[k]
                    expected=aligned_view(signal,scene,rx_id,pid,'source-V',processing_route=route)
                    assert np.array_equal(received[k],expected)
            assert noise.shape==received.shape
            studies[scene]=summarize(tensor(received,'cpu'),tensor(noise,'cpu'))
        summary[route]=studies

print(json.dumps(dict(status='VERIFIED',sample_size=len(sample),
                      isolated_reference=True,TX_reads=0,target_reads=0, metrics=summary),indent=2))
