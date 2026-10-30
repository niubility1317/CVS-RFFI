"""Matched source-stage fidelity of the fixed noVar and head-guided finals."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_dura_direct_train import runner as training,no_variance,f_guided,head_guided_capacity3
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values,action_from_arrays
from .stage_fidelity import candidates,aggregate,RUN as DEFAULT_DIAGNOSTIC
from .task_geometry import decision_metrics,GLOBALS
from .head_features import head_feature_basis,projected_feature_error

RUN='20261030-diagnostic-DURA-source-stage-compare-s2026092701-m2-r01'
MODELS=(('noVarAux',no_variance,'cf40edea90667d9645a724b1e8f333027cdf7869',False),
        ('headFeature',head_guided_capacity3,'81dfc80a2d4687f9b9ab7bc9c28ede6b092f7c4a',True))
RULE=('Only headFeature prior-only runtime may qualify, not the noVar control: BOTH cached source views '
      'prior teacher-choice agreement >=native+1pp (>=6/512 correct), cosine feature error <=.9 native, '
      'and no source scene loses any clean-F-choice agreement. Otherwise no prior runtime from this diagnostic. '
      'Oracle action/noise never deployable; no training or target/TX truth.')


def prior_decision(summary):
    checks=[]
    for epoch in ('1','20'):
        groups=summary['headFeature'][epoch];native=groups['all']['cases']['native'];prior=groups['all']['cases']['prior']
        no_scene_damage=all(v['cases']['prior']['teacher_correct']>=v['cases']['native']['teacher_correct']
            for key,v in groups.items() if key.startswith('scene:'))
        checks.append(prior['teacher_correct']-native['teacher_correct']>=6
            and prior['metrics']['feature_cosine_error']['mean']<=.9*native['metrics']['feature_cosine_error']['mean']
            and no_scene_damage)
    return dict(cached_view_passed=checks,head_prior_runtime_eligible=all(checks))


@torch.no_grad()
def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable source-only CPU output required')
    original={key:getattr(training,key) for key in GLOBALS}
    p=training.paths(args.root);clean,roles=training.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    cache=args.root/'runs'/f_guided.RUN/'scratch_joint/source_feature_cache';meta=training.read_json(cache/'complete.json')
    if (meta['status'],meta['count'],meta['dimension'],meta['F_teacher'],meta['TX_labels_read'],meta['target_reads'])!=(
            'SOURCE_F_FEATURES_COMPLETE',63000,160,f_guided.teacher_reference(),0,0):raise ValueError('Matching authorized source F cache required')
    if not np.array_equal(np.load(cache/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Source clean F order differs')
    features=np.load(cache/'features.npy',mmap_mode='r',allow_pickle=False)
    if features.shape!=(63000,160) or features.dtype!=np.float32:raise ValueError('Original FP32 source features required')
    baseline=args.root/'runs'/DEFAULT_DIAGNOSTIC;default=training.read_json(baseline/'results.json')
    if (default['status'],default['F_teacher'],default['model_unchanged'],default['F_unchanged'])!=(
            'ANALYZED_SOURCE_ONLY',f_guided.teacher_reference(),True,True) or any(default[k]!=0 for k in ('optimizer_updates','TX_labels_read','V_reads','target_reads')):
        raise ValueError('Completed same-F default source diagnostic required for exact reuse')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    frozen={key:v.clone() for key,v in reader.state_dict().items()};basis=head_feature_basis(reader).float()
    args.output.mkdir();training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        commit=args.commit,CPU_threads=2,GPU_workers=0))
    summary={'DURA_default':default['summary']};started=time.perf_counter()
    for name,module,commit,F_access in MODELS:
        for key,value in original.items():setattr(training,key,value)
        module.configure();paths=training.paths(args.root)
        done=training.read_json(paths['output']/'scratch_joint/training_complete.json')
        expected=dict(status='SOURCE_TRAINED',epoch=20,updates=4940,training_TX_labels_read=0,training_F_access=F_access,
            V_reads=0,target_reads=0,initialization_checkpoint_loads=0)
        if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Only fixed completed source finals allowed')
        if F_access and (done.get('F_updates')!=0 or done.get('F_unchanged') is not True or done.get('F_teacher')!=f_guided.teacher_reference()):
            raise ValueError('Authorized same frozen F source provenance required')
        model=training.load_model(paths['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=commit)
        model.eval().requires_grad_(False);before={key:v.clone() for key,v in model.state_dict().items()}
        smoke=reader(model(tensor(clean[indices[:2]],'cpu')))
        if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Actual source2Ls G/F smoke failed')
        print(json.dumps(dict(event='SOURCE_STAGE_COMPARE_SMOKE_PASS',model=name,real_Ls=2,TX_labels_read=0,target_reads=0)),flush=True)
        cells={}
        for epoch in (1,20):
            folder=p['cache']/'epochs'/f'{epoch:02d}'
            arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('received','kernel','coefficients','variance')}
            with np.load(folder/'groups.npz',allow_pickle=False) as groups:
                labels={key:groups[key][indices] for key in ('scene','state','lock')}
            with np.load(baseline/f'epoch{epoch:02d}.npz',allow_pickle=False) as old:
                if not np.array_equal(old['source_positions'],indices) or not np.array_equal(old['roles'],roles[indices]) or any(not np.array_equal(old[key],v) for key,v in labels.items()):
                    raise ValueError('Reused default has different source positions/roles/views')
            accumulated={}
            for offset in range(0,512,32):
                take=indices[offset:offset+32];x=tensor(clean[take],'cpu');reference=tensor(features[take],'cpu')
                values={key:tensor(v[take],'cpu') for key,v in arrays.items()};teacher=reader.classify_features(reference)
                energy=x.square().mean((1,2)).clamp_min(1e-4)
                outputs=candidates(model,values['received'],action_from_arrays(values['kernel'],values['coefficients']),values['variance'])
                for case,restored in outputs.items():
                    embedding=reader.features(restored)
                    metrics=dict(iq_nmse=(restored-x).square().mean((1,2))/energy,
                        feature_cosine_error=1-F.cosine_similarity(embedding,reference,dim=1,eps=1e-8),
                        head_projected_feature_error=projected_feature_error(embedding,reference,basis),
                        **decision_metrics(reader.classify_features(embedding),teacher))
                    for key,value in metrics.items():
                        if value.shape!=(len(take),) or not torch.isfinite(value).all():raise FloatingPointError('Nonfinite or misaligned source comparison')
                        accumulated.setdefault(case+'/'+key,[]).append(numpy_values(value))
                print(json.dumps(dict(event='SOURCE_STAGE_COMPARE_BATCH',model=name,epoch=epoch,count=offset+32,seconds=time.perf_counter()-started)),flush=True)
            values={key:np.concatenate(v) for key,v in accumulated.items()}
            if any(v.shape!=(512,) for v in values.values()):raise ValueError('Complete source comparison required')
            np.savez(args.output/f'{name}-epoch{epoch:02d}.npz',source_positions=indices,roles=roles[indices],**labels,**values)
            masks={'all':np.ones(512,dtype=bool)}
            for key,group in dict(role=roles[indices],**labels).items():
                for label in np.unique(group):masks[f'{key}:{label}']=group==label
            cells[str(epoch)]={key:aggregate(values,mask) for key,mask in masks.items()}
        if any(not torch.equal(v,before[key]) for key,v in model.state_dict().items()) or any(v.grad is not None for v in model.parameters()):
            raise ValueError('Source comparison changed frozen G')
        summary[name]=cells;del model,before;gc.collect()
    for key,value in original.items():setattr(training,key,value)
    if any(not torch.equal(v,frozen[key]) for key,v in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):
        raise ValueError('Source comparison changed frozen F')
    decision=prior_decision(summary)
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',summary=summary,prior_decision=decision,
        default_reused_from=DEFAULT_DIAGNOSTIC,default_reused_exact_source_positions=True,source_records=512,source_L=128,source_U=384,
        epochs=[1,20],batches=64,seconds=time.perf_counter()-started,F_unchanged=True,G_unchanged=True,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,F_teacher=f_guided.teacher_reference(),source_rule=RULE,
        scope='Source clean-F feature/choice fidelity only, not actual TX accuracy or target bound; noVar and DURA are controls, not selectable parents; oracle paths nondeployable'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',optimizer_updates=0,predictions=0,
        head_prior_runtime_eligible=decision['head_prior_runtime_eligible'],goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
