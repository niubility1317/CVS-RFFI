"""Matched source channel reassignment: original clean reference never changes."""
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
from .channel_reassign import matched_donors,reassign,qualifies

RUN='20261030-diagnostic-DURA-source-channel-reassign-s2026092701-r01'
G_COMMIT='cf40edea90667d9645a724b1e8f333027cdf7869'
BASELINE='20261030-diagnostic-DURA-paired-acquisition-stress-s2026092701-r01'
BASELINE_COMMIT='6328fe60db02e99c1ea806f0c27b6fd70c6b5e3b'
SEED=2026103017
PROBE_SEED=2026103018
RULE=('Both source cached views need>=480/512 metadata-eligible records, cross/within mean IQ error>=1.10 '
      'AND cross/within mean action-probe error>=1.10, clean reference unchanged, noise replay error<=1e-10. '
      'Within donor matches RX/day/scene/state/lock and differs from self; cross matches scene/state/lock '
      'and differs in RX. Only then may a separate scratch E20 whole-realization reassignment contrast '
      'with p.5/oneGforward be preregistered. Otherwise no training from this candidate;no F/TX/V/target metric.')


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
        raise ValueError('Original source RX/day handles required')
    previous=args.root/'runs'/BASELINE
    complete=training.read_json(previous/'complete.json');launch=training.read_json(previous/'launch.json')
    if (complete['status'],complete['optimizer_updates'],launch['commit'])!=('ANALYZED_SOURCE_ONLY',0,BASELINE_COMMIT):
        raise ValueError('Completed exact original IQ baseline required')
    args.output.mkdir();training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,CPU_threads=2,GPU_workers=0))
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    state={key:v.clone() for key,v in reader.state_dict().items()}
    scores=reader(model(tensor(clean[indices[:2]],'cpu')))
    if scores.shape!=(2,6) or not torch.isfinite(scores).all() or any(not torch.equal(v,state[key]) for key,v in reader.state_dict().items()):
        raise ValueError('Actual source2Ls G/F no-query smoke failed')
    del reader,state,scores
    print(json.dumps(dict(event='SOURCE_REASSIGN_SMOKE_PASS',real_Ls=2,F_scope='smoke_only',TX_labels_read=0,target_reads=0)),flush=True)
    started=time.perf_counter();summary={};decisions=[];coverage={}
    for epoch in (1,20):
        folder=p['cache']/'epochs'/f'{epoch:02d}'
        arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('received','kernel','coefficients')}
        with np.load(folder/'groups.npz',allow_pickle=False) as groups:labels={key:groups[key] for key in ('scene','state','lock')}
        with np.load(previous/f'epoch{epoch:02d}.npz',allow_pickle=False) as old:
            if not np.array_equal(old['source_positions'],indices) or not np.array_equal(old['roles'],roles[indices]) or any(not np.array_equal(old[k],v[indices]) for k,v in labels.items()):
                raise ValueError('Original baseline cohort/views differ')
            original=old['original_common_nmse'].copy()
        plan=matched_donors(indices,rx,day,labels,np.random.default_rng(SEED+epoch))
        positions=plan['positions'];count=len(positions)
        if not count:raise ValueError('No matched source counterfactuals available')
        coverage[str(epoch)]=dict(count=count,excluded_count=len(plan['excluded']))
        values={'original/iq_nmse':original[np.isin(indices,positions)]};accumulated={}
        generator=torch.Generator().manual_seed(PROBE_SEED)
        for offset in range(0,count,32):
            take=positions[offset:offset+32];x=tensor(clean[take],'cpu');reference_before=x.clone()
            power=x.square().mean((1,2)).clamp_min(1e-4)
            probes=[torch.complex(torch.randn(len(take),256,generator=generator),torch.randn(len(take),256,generator=generator)) for _ in range(2)]
            for case in ('within','cross'):
                donor=plan[case][offset:offset+32]
                actual=action_from_arrays(tensor(arrays['kernel'][donor],'cpu'),tensor(arrays['coefficients'][donor],'cpu'))
                observed,noise,signal=reassign(x,tensor(clean[donor],'cpu'),tensor(arrays['received'][donor],'cpu'),actual)
                context=model.estimate_context(observed);estimated=context[1]
                restored=model.stage_outputs(observed,context)[0][-1]
                if epoch==1 and offset==0 and case=='within' and not torch.equal(restored,model(observed)):
                    raise ValueError('Reused context changed native final output')
                errors=[]
                for probe in probes:
                    response=actual(probe)
                    errors.append((estimated(probe)-response).abs().square().mean(1)/response.abs().square().mean(1).clamp_min(1e-4))
                metrics=dict(iq_nmse=(restored-x).square().mean((1,2))/power,
                    action_probe_nmse=torch.stack(errors).mean(0),
                    noise_replay_nmse=((observed-signal)-noise).square().mean((1,2))/power,
                    signal_power=signal.square().mean((1,2))/power,noise_power=noise.square().mean((1,2))/power)
                if not torch.equal(x,reference_before):raise ValueError('Clean reference changed')
                for key,v in metrics.items():
                    if v.shape!=(len(take),) or not torch.isfinite(v).all():raise FloatingPointError('Nonfinite reassignment metric')
                    accumulated.setdefault(case+'/'+key,[]).append(numpy_values(v))
            print(json.dumps(dict(event='SOURCE_REASSIGN_BATCH',epoch=epoch,count=offset+len(take),seconds=time.perf_counter()-started)),flush=True)
        values.update({key:np.concatenate(v) for key,v in accumulated.items()})
        np.savez(args.output/f'epoch{epoch:02d}.npz',source_positions=positions,donor_within=plan['within'],donor_cross=plan['cross'],excluded=plan['excluded'],
            roles=roles[positions],source_rx=rx[positions],source_day=day[positions],
            **{key:value[positions] for key,value in labels.items()},**values)
        masks={'all':np.ones(count,dtype=bool)}
        for key,group in dict(role=roles[positions],source_rx=rx[positions],source_day=day[positions],**{k:v[positions] for k,v in labels.items()}).items():
            for label in np.unique(group):masks[f'{key}:{label}']=group==label
        summary[str(epoch)]={key:dict(count=int(mask.sum()),metrics={name:dict(mean=float(v[mask].mean()),median=float(np.median(v[mask])),p95=float(np.quantile(v[mask],.95))) for name,v in values.items()}) for key,mask in masks.items()}
        metrics=summary[str(epoch)]['all']['metrics']
        decisions.append(dict(epoch=epoch,count=count,IQ_ratio=metrics['cross/iq_nmse']['mean']/metrics['within/iq_nmse']['mean'],
            action_ratio=metrics['cross/action_probe_nmse']['mean']/metrics['within/action_probe_nmse']['mean'],reference_unchanged=True,
            noise_error_max=max(float(values[case+'/noise_replay_nmse'].max()) for case in ('within','cross'))))
    if any(not torch.equal(v,before[key]) for key,v in model.state_dict().items()) or any(v.grad is not None for v in model.parameters()):raise ValueError('Frozen G changed')
    eligible=qualifies(decisions)
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',summary=summary,decisions=decisions,coverage=coverage,
        training_contrast_eligible=eligible,source_rule=RULE,requested_source_records=512,requested_source_L=128,requested_source_U=384,
        donor_seed_base=SEED,probe_seed=PROBE_SEED,G_unchanged=True,F_smoke_unchanged=True,F_smoke_inputs=2,F_optimization_inputs=0,F_feature_analysis_inputs=0,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,baseline_reused=BASELINE,baseline_G_commit=G_COMMIT,
        seconds=time.perf_counter()-started,scope='Source matched operator/noise counterfactual,not a fresh native AGC draw or causal RX proof;no trained version/target prediction'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',optimizer_updates=0,predictions=0,training_contrast_eligible=eligible,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
