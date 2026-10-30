"""Source teacher fidelity along fixed DURA stages and counterfactual solves."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_dura_direct_train import runner as training,f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values,last_prior,action_from_arrays
from experiments.pamf_restoration.models import complex_iq,pair
from experiments.pamf_uncertainty_restore.models import weighted_map
from .task_geometry import decision_metrics
from .head_features import head_feature_basis,projected_feature_error

RUN='20261030-diagnostic-DURA-source-stage-fidelity-s2026092701-r01'
CG_COUNTS=(4,8,16)
RULE=('A fixed-last-prior learned CG4/8/16 diagnostic path qualifies for separate formal-version registration only '
      'if BOTH cached source views teacher agreement improves >=1pp overall, cosine feature error <=.9 native, '
      'and no source scene loses any clean-F-choice agreements; choose smallest qualifying CG count, otherwise no solver release. '
      'Counterfactual true action/noise paths never qualify; no target or TX labels, no updated weights.')


def solver_decision(summary):
    decisions={}
    for steps in CG_COUNTS:
        checks=[]
        for epoch in ('1','20'):
            current=summary[epoch];native=current['all']['cases']['native'];candidate=current['all']['cases']['CG'+str(steps)]
            no_scene_damage=all(group['cases']['CG'+str(steps)]['teacher_correct']>=group['cases']['native']['teacher_correct']
                for key,group in current.items() if key.startswith('scene:'))
            checks.append(candidate['teacher_correct']-native['teacher_correct']>=6
                and candidate['metrics']['feature_cosine_error']['mean']<=.9*native['metrics']['feature_cosine_error']['mean']
                and no_scene_damage)
        decisions[str(steps)]=dict(cached_view_passed=checks,qualified=all(checks))
    eligible=next((steps for steps in CG_COUNTS if decisions[str(steps)]['qualified']),None)
    return dict(candidates=decisions,smallest_qualified_CG=eligible)


@torch.no_grad()
def candidates(model,received,actual_action,actual_variance):
    context=model.estimate_context(received);base,action,variance=context
    stages=model.stage_outputs(received,context)[0]
    prior=last_prior(model,received,context,stages[0]);observed=complex_iq(received);start=complex_iq(stages[0])
    check=pair(weighted_map(action,observed,complex_iq(prior),variance,steps=2,initial=start))
    if not torch.equal(check,stages[-1]):raise ValueError('Native final stage must reconstruct exactly')
    values=dict(received=received,coarse=base,stage1=stages[0],prior=prior,native=stages[-1])
    for steps in CG_COUNTS:
        values['CG'+str(steps)]=pair(weighted_map(action,observed,complex_iq(prior),variance,steps=steps,initial=start))
    values['oracle_action_CG16']=pair(weighted_map(actual_action,observed,complex_iq(prior),variance,steps=16,initial=start))
    values['oracle_both_CG16']=pair(weighted_map(actual_action,observed,complex_iq(prior),actual_variance,steps=16,initial=start))
    return values


def aggregate(values,selected):
    result={}
    native=values['native/teacher_argmax_agreement'].astype(bool)
    for case in sorted({key.split('/')[0] for key in values}):
        metrics={key.split('/',1)[1]:dict(mean=float(v[selected].mean()),median=float(np.median(v[selected])),p95=float(np.quantile(v[selected],.95)))
            for key,v in values.items() if key.startswith(case+'/')}
        correct=values[case+'/teacher_argmax_agreement'].astype(bool)
        result[case]=dict(metrics=metrics,teacher_correct=int(correct[selected].sum()),
            teacher_repaired_vs_native=int((correct&~native&selected).sum()),
            teacher_damaged_vs_native=int((~correct&native&selected).sum()))
    return dict(count=int(selected.sum()),cases=result)


@torch.no_grad()
def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU source diagnostic root required')
    p=training.paths(args.root)
    from experiments.pamf_r65_four_route.runtime import load_checkpoint
    model,payload=load_checkpoint(p['parent'],'cpu')
    if (payload['arm'],payload['epoch'],payload['updates'])!=('stage2',20,4940):raise ValueError('Default source-final DURA required')
    model.eval().requires_grad_(False)
    clean,roles=training.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    cache=args.root/'runs'/f_guided.RUN/'scratch_joint/source_feature_cache'
    meta=training.read_json(cache/'complete.json')
    if (meta['status'],meta['count'],meta['dimension'],meta['F_teacher'],meta['TX_labels_read'],meta['target_reads'])!=(
            'SOURCE_F_FEATURES_COMPLETE',63000,160,f_guided.teacher_reference(),0,0):raise ValueError('Legal source F cache required')
    if not np.array_equal(np.load(cache/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Source F physical order differs')
    reference_features=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    if reference_features.shape!=(63000,160) or reference_features.dtype!=np.float32:raise ValueError('Original FP32 source F cache required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    snapshots=[{key:v.clone() for key,v in item.state_dict().items()} for item in (model,reader)]
    basis=head_feature_basis(reader).float();args.output.mkdir()
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,CPU_threads=2,GPU_workers=0))
    scores=reader(model(tensor(clean[indices[:2]],'cpu')))
    if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Actual source2Ls G/F smoke failed')
    print(json.dumps(dict(event='SOURCE_STAGE_FIDELITY_SMOKE_PASS',real_Ls=2,TX_labels_read=0,target_reads=0)),flush=True)
    summary={};started=time.perf_counter()
    for epoch in (1,20):
        folder=p['cache']/'epochs'/f'{epoch:02d}'
        arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('received','kernel','coefficients','variance')}
        with np.load(folder/'groups.npz',allow_pickle=False) as groups:
            labels={key:groups[key][indices] for key in ('scene','state','lock')}
        accumulated={}
        for offset in range(0,512,32):
            take=indices[offset:offset+32];x=tensor(clean[take],'cpu');reference=tensor(reference_features[take],'cpu')
            values={key:tensor(v[take],'cpu') for key,v in arrays.items()};teacher=reader.classify_features(reference)
            energy=x.square().mean((1,2)).clamp_min(1e-4)
            variants=candidates(model,values['received'],action_from_arrays(values['kernel'],values['coefficients']),values['variance'])
            for case,restored in variants.items():
                feature=reader.features(restored);actual=reader.classify_features(feature)
                metrics=dict(iq_nmse=(restored-x).square().mean((1,2))/energy,
                    feature_cosine_error=1-F.cosine_similarity(feature,reference,dim=1,eps=1e-8),
                    head_projected_feature_error=projected_feature_error(feature,reference,basis),
                    **decision_metrics(actual,teacher))
                for key,v in metrics.items():
                    if v.shape!=(len(take),) or not torch.isfinite(v).all():raise FloatingPointError('Nonfinite or misaligned source fidelity')
                    accumulated.setdefault(case+'/'+key,[]).append(numpy_values(v))
            print(json.dumps(dict(event='SOURCE_STAGE_FIDELITY_BATCH',epoch=epoch,count=offset+32,seconds=time.perf_counter()-started)),flush=True)
        values={key:np.concatenate(v) for key,v in accumulated.items()}
        if any(v.shape!=(512,) for v in values.values()):raise ValueError('Complete paired source results required')
        np.savez(args.output/f'epoch{epoch:02d}.npz',source_positions=indices,roles=roles[indices],**labels,**values)
        masks={'all':np.ones(512,dtype=bool)}
        for key,labels_for_group in dict(role=roles[indices],**labels).items():
            for label in np.unique(labels_for_group):masks[f'{key}:{label}']=labels_for_group==label
        summary[str(epoch)]={key:aggregate(values,mask) for key,mask in masks.items()}
    for item,snapshot in zip((model,reader),snapshots):
        if any(not torch.equal(v,snapshot[k]) for k,v in item.state_dict().items()) or any(p.grad is not None for p in item.parameters()):
            raise ValueError('Source diagnostic changed frozen state or accumulated gradient')
    decision=solver_decision(summary)
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',summary=summary,solver_decision=decision,
        source_records=512,source_L=128,source_U=384,epochs=[1,20],seconds=time.perf_counter()-started,
        model_unchanged=True,F_unchanged=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        source_rule=RULE,F_teacher=f_guided.teacher_reference(),G_source_default_unchanged=True,
        scope='Source clean-F choice/feature fidelity, not TX accuracy or target evidence; oracle paths are source diagnostics only, never deployment or rigorous accuracy bounds'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',optimizer_updates=0,predictions=0,
        smallest_qualified_CG=decision['smallest_qualified_CG'],goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
