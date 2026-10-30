"""Calibrate spectral styles on source L/U; stress fixed G without labels or F guidance."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.pamf_dura_direct_train import runner as training,no_variance
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values,action_from_arrays
from experiments.pamf_restoration.models import complex_iq,pair
from .spectral_style import bank_from_psd,paired_spectral

RUN='20261030-diagnostic-DURA-source-spectral-stress-s2026092701-r01'
G_COMMIT='cf40edea90667d9645a724b1e8f333027cdf7869'
BASELINE='20261030-diagnostic-DURA-paired-acquisition-stress-s2026092701-r01'
BASELINE_COMMIT='6328fe60db02e99c1ea806f0c27b6fd70c6b5e3b'
SEED=2026103015
RULE=('Both source cached views must have styled/original common-power MSE ratio>=1.10, '
      'per-packet power relative error max<=2e-6 and exact-A noise-replay error/common power max<=1e-10. '
      'Only then separately preregister scratch E20 paired spectral augmentation p.5 with one G forward. '
      'Otherwise no training from this candidate. No F metrics, TX/V/target or parameter selection.')


@torch.no_grad()
def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable source-only CPU output required')
    no_variance.configure();p=training.paths(args.root)
    done=training.read_json(p['output']/'scratch_joint/training_complete.json')
    expected=dict(status='SOURCE_TRAINED',epoch=20,updates=4940,training_TX_labels_read=0,training_F_access=False,
        V_reads=0,target_reads=0,initialization_checkpoint_loads=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed original noVar source-final required')
    model=training.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=G_COMMIT).eval().requires_grad_(False)
    before={key:v.clone() for key,v in model.state_dict().items()}
    clean,roles=training.source_training(p);indices=source_indices(roles)
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False);day=np.load(p['cache']/'train/day.npy',allow_pickle=False)
    if rx.shape!=(63000,) or day.shape!=(63000,) or set(rx.tolist())!={1,3,4,6,8} or set(day.tolist())!={1,2,3}:
        raise ValueError('Existing original source RX/day metadata required')
    sessions=[(r,d) for r in (1,3,4,6,8) for d in (1,2,3)]
    lookup={v:i for i,v in enumerate(sessions)}
    session=np.asarray([lookup[int(r),int(d)] for r,d in zip(rx,day)],np.int64)
    counts=np.bincount(session,minlength=15)
    if not np.array_equal(counts,np.full(15,4200)):raise ValueError('Original balanced source sessions required')
    baseline=args.root/'runs'/BASELINE
    previous=training.read_json(baseline/'complete.json');launch=training.read_json(baseline/'launch.json')
    if (previous['status'],previous['optimizer_updates'],previous['predictions'],launch['commit'])!=('ANALYZED_SOURCE_ONLY',0,0,BASELINE_COMMIT):
        raise ValueError('Completed fixed original baseline required for exact source reuse')
    args.output.mkdir();training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,CPU_threads=2,GPU_workers=0))
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    state={key:v.clone() for key,v in reader.state_dict().items()}
    logits=reader(model(tensor(clean[indices[:2]],'cpu')))
    if logits.shape!=(2,6) or not torch.isfinite(logits).all() or any(not torch.equal(v,state[key]) for key,v in reader.state_dict().items()):
        raise ValueError('Actual source2Ls G/F no-query smoke failed')
    del reader,state,logits
    print(json.dumps(dict(event='SOURCE_SPECTRAL_SMOKE_PASS',real_Ls=2,F_inputs_scope='noquery_smoke_only',TX_labels_read=0,target_reads=0)),flush=True)
    started=time.perf_counter();sums=torch.zeros(15,256,dtype=torch.float64)
    for offset in range(0,63000,512):
        x=tensor(clean[offset:offset+512],'cpu')
        power=torch.fft.fft(complex_iq(x),norm='ortho').abs().square().double()
        sums.index_add_(0,torch.tensor(session[offset:offset+512].tolist()),power)
    mean_psd=sums/torch.tensor(counts.tolist())[:,None]
    gain,calibration=bank_from_psd(mean_psd);gain=gain.float()
    rng=np.random.default_rng(SEED)
    origins=session[indices]
    destinations=np.asarray([rng.choice([j for j,(r,d) in enumerate(sessions) if r!=rx[i]]) for i in indices],np.int64)
    if np.any(np.asarray(sessions)[origins,0]==np.asarray(sessions)[destinations,0]):raise ValueError('Cross-source-RX style draw required')
    calibration.update(source_records=63000,source_L=6300,source_U=56700,sessions=sessions,session_counts=counts.tolist(),
        seconds=time.perf_counter()-started,scope='Empirical source RX/day PSD contrast, not an identified causal receiver transfer function')
    np.savez(args.output/'spectral_bank.npz',mean_psd=np.from_dlpack(mean_psd).copy(),gain=numpy_values(gain),
        sessions=np.asarray(sessions,np.int64),session_counts=counts)
    print(json.dumps(dict(event='SOURCE_SPECTRAL_BANK_COMPLETE',source_records=63000,seconds=calibration['seconds'],gain_min=calibration['gain_min'],gain_max=calibration['gain_max'])),flush=True)
    summaries={};decisions=[]
    for epoch in (1,20):
        folder=p['cache']/'epochs'/f'{epoch:02d}'
        arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('received','kernel','coefficients')}
        with np.load(folder/'groups.npz',allow_pickle=False) as groups:labels={key:groups[key][indices] for key in ('scene','state','lock')}
        # Selective NPZ access never opens the old diagnostic's F metric fields.
        with np.load(baseline/f'epoch{epoch:02d}.npz',allow_pickle=False) as old:
            if not np.array_equal(old['source_positions'],indices) or not np.array_equal(old['roles'],roles[indices]) or any(not np.array_equal(old[key],v) for key,v in labels.items()):
                raise ValueError('Reused original IQ baseline has different source cohort/views')
            original=old['original_common_nmse'].copy()
        if original.shape!=(512,) or not np.isfinite(original).all():raise ValueError('Complete original IQ metrics required')
        accumulated={}
        for offset in range(0,512,32):
            take=indices[offset:offset+32];x=tensor(clean[take],'cpu');y=tensor(arrays['received'][take],'cpu')
            action=action_from_arrays(tensor(arrays['kernel'][take],'cpu'),tensor(arrays['coefficients'][take],'cpu'))
            weights=gain[torch.tensor(origins[offset:offset+32].tolist()),torch.tensor(destinations[offset:offset+32].tolist())]
            shifted,reference=paired_spectral(y,x,action,weights)
            restored=model(shifted);power=x.square().mean((1,2)).clamp_min(1e-4)
            residual=y-pair(action(complex_iq(x)));changed=shifted-pair(action(complex_iq(reference)))
            metrics=dict(styled_common_nmse=(restored-reference).square().mean((1,2))/power,
                power_relative_error=(reference.square().mean((1,2))-power).abs()/power,
                noise_replay_error=(changed-residual).square().mean((1,2))/power,
                style_delta_energy=(reference-x).square().mean((1,2))/power)
            for key,v in metrics.items():
                if v.shape!=(len(take),) or not torch.isfinite(v).all():raise FloatingPointError('Nonfinite source spectral stress')
                accumulated.setdefault(key,[]).append(numpy_values(v))
            print(json.dumps(dict(event='SOURCE_SPECTRAL_BATCH',epoch=epoch,count=offset+32,seconds=time.perf_counter()-started)),flush=True)
        values=dict(original_common_nmse=original,**{key:np.concatenate(v) for key,v in accumulated.items()})
        np.savez(args.output/f'epoch{epoch:02d}.npz',source_positions=indices,roles=roles[indices],source_rx=rx[indices],source_day=day[indices],
            origin_session=origins,destination_session=destinations,**labels,**values)
        masks={'all':np.ones(512,dtype=bool)}
        for key,group in dict(role=roles[indices],source_rx=rx[indices],source_day=day[indices],**labels).items():
            for label in np.unique(group):masks[f'{key}:{label}']=group==label
        summary={key:dict(count=int(mask.sum()),metrics={name:dict(mean=float(v[mask].mean()),median=float(np.median(v[mask])),max=float(v[mask].max())) for name,v in values.items()}) for key,mask in masks.items()}
        summaries[str(epoch)]=summary;overall=summary['all']['metrics']
        ratio=overall['styled_common_nmse']['mean']/overall['original_common_nmse']['mean']
        decisions.append(dict(epoch=epoch,common_MSE_ratio=ratio,power_error_max=overall['power_relative_error']['max'],
            noise_replay_max=overall['noise_replay_error']['max'],passed=ratio>=1.1 and overall['power_relative_error']['max']<=2e-6 and overall['noise_replay_error']['max']<=1e-10))
    if any(not torch.equal(v,before[key]) for key,v in model.state_dict().items()) or any(v.grad is not None for v in model.parameters()):raise ValueError('Frozen G changed')
    result=dict(status='ANALYZED_SOURCE_ONLY',calibration=calibration,summary=summaries,decisions=decisions,
        training_contrast_eligible=all(v['passed'] for v in decisions),source_rule=RULE,source_records=512,source_L=128,source_U=384,epochs=[1,20],
        seed=SEED,G_unchanged=True,F_smoke_unchanged=True,F_smoke_inputs=2,F_optimization_inputs=0,F_feature_analysis_inputs=0,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,baseline_reused=BASELINE,baseline_G_commit=G_COMMIT,
        seconds=time.perf_counter()-started,scope='Synthetic source spectral robustness;not causal RX proof or TX fidelity;no trained version/target prediction')
    training.write_json(args.output/'results.json',result)
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',optimizer_updates=0,predictions=0,
        training_contrast_eligible=result['training_contrast_eligible'],goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
