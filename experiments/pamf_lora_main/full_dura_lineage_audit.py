"""Audit frozen LoRa DURA stages on a fixed, TX-blind source validation panel."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.g_lineage import STAGES, load
from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.full_dura_lineage_data import aligned_teacher
from experiments.pamf_lora_main.full_supervised import handles
from experiments.pamf_lora_main.runner import activate


PER_RECEIVER=24


def fixed_panel(source, c):
    with np.load(source/'V_metadata.npz',allow_pickle=False) as a:
        if a['ids'].tolist()!=c['role_ids']['V']:
            raise ValueError('Source V IDs differ from the dataset contract')
        receiver=a['rx_i']
    rng=np.random.default_rng(2026112210)
    indices=np.stack([np.sort(rng.choice(np.flatnonzero(receiver==j),
                                      PER_RECEIVER,replace=False)) for j in range(len(SOURCE))])
    return indices


def complex_iq(iq):
    return torch.complex(iq[:,0],iq[:,1])


def waveform_error(x,y):
    """Return registered-pair NMSE and residual after removal of one complex gain."""
    ref=complex_iq(x)
    received=complex_iq(y)
    power=ref.abs().square().mean(-1).clamp_min(1e-8)
    gain=(received*ref.conj()).mean(-1)/power
    received_power=received.abs().square().mean(-1)
    residual=(received-gain[:,None]*ref).abs().square().mean(-1)
    unexplained=torch.where(received_power>1e-8,residual/received_power.clamp_min(1e-8),
                            torch.ones_like(received_power))
    return {
        'iq_nmse': (received-ref).abs().square().mean(-1)/power,
        'gain_removed_nmse': residual/power,
        'relative_power': received_power/power,
        'unexplained_energy_fraction': unexplained,
    }


def summarize(metrics):
    out={}
    for name,values in metrics.items():
        a=values.detach().cpu().tolist() if isinstance(values,torch.Tensor) else values
        a=np.asarray(a,dtype=np.float64)
        if not np.isfinite(a).all():
            raise FloatingPointError(f'Nonfinite waveform statistic {name}')
        q=np.quantile(a,[.5,.95]).tolist()
        out[name]=dict(mean=float(a.mean()),median=q[0],p95=q[1])
    return out


def frozen_stage(root, c, stage, seed, device):
    path=root/'G'/'lineage'/stage
    proof=read(path/'complete.json')
    if (proof['status']!='SOURCE_FROZEN' or proof['stage']!=stage or proof['updates']!=4940
            or any(proof[k] for k in ('F_reads','TX_labels_read','target_reads'))):
        raise ValueError('Require a fully frozen source-only stage')
    g,item=load(path/'final.pt',c,'LU',device)
    if item['stage']!=stage or item['model_seed']!=seed:
        raise ValueError('Stage identity mismatch')
    return g.eval().requires_grad_(False)


def proxy_action_error(estimator, received, expected, clean):
    from experiments.pamf_action_restore.evaluation import probe_error
    kernel,coef,noise=[tensor(np.stack(y),received.device) for y in zip(*expected,strict=True)]
    true=Action(complex_iq(kernel),torch.complex(coef[:,0],coef[:,1]),
                torch.complex(coef[:,2],coef[:,3]))
    _,estimate,variance=estimator.estimate(received)
    generator=torch.Generator(device=received.device).manual_seed(2026112217)
    probes=[torch.complex(*torch.randn((2,len(clean),256),generator=generator,device=clean.device))/2**.5
            for _ in range(2)]
    measured=(complex_iq(received)-true(complex_iq(clean))).abs().square().mean(-1)*.5
    variance_gap=10.*(variance.clamp_min(1e-8).log10()-noise.log10())
    return dict(probe_nmse=probe_error(estimate,true,probes),noise_estimate=variance,
                noise_truth=noise,observed_noise=measured,
                noise_error_db=variance_gap,absolute_noise_error_db=variance_gap.abs())


@torch.no_grad()
def audit(rt, stage, seed, device):
    source,cache,c=handles(rt)
    ids=fixed_panel(source,c)
    v=np.load(cache/'V.npy',mmap_mode='r',allow_pickle=False)
    if v.shape!=(len(VIEWS),len(c['role_ids']['V']),6,2,256) or v.dtype!=np.float32:
        raise ValueError('Source V waveform dimensions differ')
    root=Path(rt['run_root'])
    g=frozen_stage(root,c,stage,seed,device) if stage else None
    observations={}
    for j,view in enumerate(VIEWS):
        records={}
        for rx,indices in zip(SOURCE,ids,strict=True):
            # Rotating the selected chirp covers all six without weighting any packet twice.
            chirps=np.arange(PER_RECEIVER)%6
            windows=v[0,indices,chirps]
            x=tensor(windows,device)
            y=tensor(v[j,indices,chirps],device)
            ref=summarize(waveform_error(x,y))
            current=dict(full=ref)
            if g is not None:
                current['restored_full']=summarize(waveform_error(x,g(y)))
            if j:
                items=[aligned_teacher((w,view,rx,c['role_ids']['V'][i],19))
                       for i,w in zip(indices.tolist(),windows,strict=True)]
                proxy=[item for item,_ in items]
                p=tensor(np.stack([item[0] for item in proxy]),device)
                current['residual_proxy']=summarize(waveform_error(x,p))
                current['removed_fir_energy']=summarize({'fraction':[value for _,value in items]})['fraction']
                if g is not None:
                    current['restored_proxy']=summarize(waveform_error(x,g(p)))
                    if stage!='coarse':
                        estimator=g if stage=='action' else g.frozen
                        current['proxy_action']=summarize(proxy_action_error(
                            estimator,p,[item[1:] for item in proxy],x))
            records[rx]=current
        observations[view]=records
    return dict(status='SOURCE_AUDIT_COMPLETE',stage=stage or 'input',n=int(ids.size),
                repeats_per_receiver=PER_RECEIVER,split='V',F_reads=0,TX_reads=0,target_reads=0,
                receiver_records=observations)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('runtime',type=Path)
    p.add_argument('--stage',choices=STAGES)
    p.add_argument('--seed',type=int,default=2026092702)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    rt=read(a.runtime)
    if set(rt)&{'truth','target_views','target_truth','F_checkpoint'}:
        raise ValueError('Pass the source worker configuration without target or F')
    activate(rt)
    print(json.dumps(audit(rt,a.stage,a.seed,a.device),allow_nan=False),flush=True)
