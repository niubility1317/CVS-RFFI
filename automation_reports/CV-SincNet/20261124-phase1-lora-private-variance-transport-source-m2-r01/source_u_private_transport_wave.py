
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import numpy as np
import torch

from experiments.cric_dura_comparison.io import read,tensor
from experiments.pamf_lora_main.channel_blind import corruption_score
from experiments.pamf_lora_main.channel_blind_source import partition
from experiments.pamf_lora_main.data import SOURCE,VIEWS
from experiments.pamf_lora_main.dura_repair import PacketRestorer,complex_iq
from experiments.pamf_lora_main.full_packet_views import source_contract,source_role
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.offband_noise_source import replay
from experiments.pamf_lora_main.private_transport_source import MODES,proposed_reads
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_lora_main.weak_guard_source import parent_model


def project(value):
    spec=torch.fft.fft(value,norm='ortho')
    return spec-spec.mean(1,keepdim=True)


def summarize(predicted,expected,rms,mask):
    s=predicted[...,mask]
    t=expected[...,mask]
    weight=rms.square()
    e=(t.abs().square()*weight).sum((1,2))
    d=((s-t).abs().square()*weight).sum((1,2))
    output=(s.abs().square()*weight).sum((1,2))
    good=e>1e-12
    transfer=(s*t.conj()*weight).sum((1,2))[good]/e[good]
    return dict(relative_error=float(d.sum()/e.sum().clamp_min(1e-12)),
                output_power_to_reference=float(output.sum()/e.sum().clamp_min(1e-12)),
                median_real_gain=float(transfer.real.median()),
                median_absolute_phase=float(transfer.angle().abs().median()))


@torch.no_grad()
def main():
    rt=read(Path('runtime.json'))
    activate(rt)
    source=Path(rt['source_root'])
    contract=source_contract(source)
    raw,rx,ids,_=source_role(source,contract,'U_s')
    _,_,held=partition(rx)
    selected=np.concatenate([subset[:30] for subset in np.split(held,len(SOURCE))])
    model,threshold=parent_model(Path(rt['run_root']),contract,'cpu')
    tracker=PacketRestorer('consensus').eval()
    index=(256*torch.fft.fftfreq(256)).abs()
    bands={'1_to_8':(index>0)&(index<=8),
           '9_to_32':(index>8)&(index<=32),
           '33_to_64':(index>32)&(index<=64)}
    result={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for route in ('full','residual'):
            rows={}
            for view in (VIEWS[3],VIEWS[-1]):
                jobs=((route,view,raw[i],rx[i],ids[i]) for i in selected)
                capture,noise=(tensor(np.stack(v),'cpu')
                               for v in zip(*pool.map(replay,jobs,chunksize=6),strict=True))
                observed,rot,q,rms=tuple(torch.cat(parts) for parts in
                    zip(*(coordinates(tracker,chunk) for chunk in capture.split(32)),strict=True))
                active=corruption_score(observed)>threshold
                prepared={name:observed.clone() for name in MODES[2:]}
                for picks in torch.where(active)[0].split(18):
                    if len(picks):
                        legacy,rest=proposed_reads(observed[picks],q[picks],model)
                        for name,new in zip(MODES[2:],(legacy,*(legacy+change for change in rest)),strict=True):
                            prepared[name][picks]=new
                target=project(observed-complex_iq(noise)*rot.conj()/rms)
                output={'direct':observed,**prepared}
                rows[view]=dict(n=len(selected),active=int(active.sum()),
                    by_mode={name:{b:summarize(project(value),target,rms,sel) for b,sel in bands.items()}
                             for name,value in output.items()})
            result[route]=rows
    print(json.dumps(dict(n=len(selected),role='held_U_s',target_reads=0,TX_reads=0,F_reads=0,V_reads=0,
                          exact_noise_isolated_after_inference=True,by_route=result)),flush=True)


if __name__=='__main__':
    main()
