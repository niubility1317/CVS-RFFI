"""Source-only gradient diagnosis of coarse/action roles in joint training."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from experiments.pamf_dura_direct_train import no_variance
from experiments.pamf_dura_efficiency.model import direct_objective,supervised_action
from experiments.pamf_dura_diagnostics.runner import tensor,source_indices
from .runner import geometry

base=no_variance.base
RUN='20261031-diagnostic-DURA-source-module-gradients-s2026092701-r01'
SOURCE='20261030-phase1-DURA-direct-noVarAux-s2026092701-r01'
SOURCE_COMMIT='cf40edea90667d9645a724b1e8f333027cdf7869'
PREFIXES={'coarse':'frozen.coarse.','action':'frozen.estimator.','prior':'prior.'}


def validate_source(done):
    expected=dict(status='SOURCE_TRAINED',epoch=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        training_TX_labels_read=0,training_F_access=False,V_reads=0,target_reads=0,teacher_updates=0,
        learned_teacher_inputs=0,initialization_checkpoint_loads=0,ancestor_epochs=0,head_fit_updates=0,parameters=266951)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Original complete scratch source-only E20 noVar checkpoint required')


def module_groups(names):
    groups={key:[i for i,name in enumerate(names) if name.startswith(prefix)] for key,prefix in PREFIXES.items()}
    if any(not indices for indices in groups.values()) or sorted(i for indices in groups.values() for i in indices)!=list(range(len(names))):
        raise ValueError('Disjoint complete original coarse/action/prior parameter groups required')
    return groups


def source_losses(model,received,clean,action,variance,probes):
    _,terms,restored,coarse=direct_objective(model,received,clean,action,variance,probes,return_coarse=True)
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    return dict(final_IQ=terms['iq_nmse'],prior_IQ=.1*terms['prior_nmse'],spectral=.05*terms['spectral_error'],
        action_probe=.05*terms['action_probe_nmse'],
        coarse_clean_IQ=(coarse-clean.detach()).square().mean((1,2))/energy)


def summarize(records):
    result={}
    for group in PREFIXES:
        for pair in records[0]['geometry'][group]:
            rows=[r['geometry'][group][pair] for r in records]
            values=[r['cosine'] for r in rows if r['cosine'] is not None]
            result[group+'/'+pair]=dict(batches=len(rows),defined_cosines=len(values),negative_cosines=sum(v<0 for v in values),
                mean_cosine=float(np.mean(values)) if values else None,
                median_right_to_left_norm=float(np.median([r['right_to_left_norm'] for r in rows if r['right_to_left_norm'] is not None])) if values else None)
    return result


def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source diagnostic root required')
    no_variance.configure();p=base.paths(args.root)
    if p['output'].name!=SOURCE:raise ValueError('Predesignated source E20 noVar required')
    validate_source(base.read_json(p['output']/'scratch_joint/training_complete.json'))
    model=base.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=SOURCE_COMMIT)
    model.eval().requires_grad_(True);before={k:v.detach().clone() for k,v in model.state_dict().items()}
    names,parameters=zip(*model.named_parameters());groups=module_groups(names)
    clean,roles=base.source_training(p);indices=source_indices(roles)
    if len(indices)!=512 or (roles[indices]=='L_s').sum()!=128 or (roles[indices]=='U_s').sum()!=384:
        raise ValueError('Fixed existing source probe required')
    args.output.mkdir(parents=True)
    base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        source_run=SOURCE,source_commit=SOURCE_COMMIT,CPU_threads=2,GPU_workers=0))
    with torch.no_grad():output=model(tensor(clean[indices[:2]],'cpu'))
    if output.shape!=(2,2,256) or not torch.isfinite(output).all():raise ValueError('Real source checkpoint no-query smoke failed')
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls=2,G_inputs=2,F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0))
    print('{"event":"REAL_SOURCE_MODULE_GRADIENT_CHECKPOINT_SMOKE_PASS"}',flush=True)
    records=[];started=time.perf_counter();generator=torch.Generator().manual_seed(2026103005)
    for view in (1,20):
        folder=p['cache']/'epochs'/f'{view:02d}'
        arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('received','kernel','coefficients','variance')}
        expected=dict(received=(63000,2,256),kernel=(63000,2,64),coefficients=(63000,4,256),variance=(63000,))
        if any(v.shape!=expected[k] or v.dtype!=np.float32 for k,v in arrays.items()):raise ValueError('Matched source views/actions required')
        for offset in range(0,512,32):
            take=indices[offset:offset+32];values={k:tensor(v[take],'cpu') for k,v in arrays.items()}
            probes=[torch.complex(torch.randn(32,256,generator=generator),torch.randn(32,256,generator=generator)) for _ in range(2)]
            losses=source_losses(model,values['received'],tensor(clean[take],'cpu'),
                supervised_action(values['kernel'],values['coefficients']),values['variance'],probes)
            gradients={}
            for key,value in losses.items():
                if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite source module loss')
                gradient=torch.autograd.grad(value.mean(),parameters,retain_graph=True,allow_unused=True)
                gradients[key]=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,gradient)]
                if any(not torch.isfinite(g).all() for g in gradients[key]):raise FloatingPointError('Nonfinite source module gradients')
            graphs={}
            for group,selected in groups.items():
                iq=[gradients['final_IQ'][i] for i in selected]
                graphs[group]={name:geometry(iq,[gradients[name][i] for i in selected]) for name in ('coarse_clean_IQ','action_probe','prior_IQ','spectral')}
            records.append(dict(source_view=view,offset=offset,count=32,L_s=int((roles[take]=='L_s').sum()),U_s=int((roles[take]=='U_s').sum()),
                source_positions=take.tolist(),losses={k:float(v.detach().mean()) for k,v in losses.items()},geometry=graphs))
            print(json.dumps(dict(event='SOURCE_MODULE_GRADIENT_BATCH',source_view=view,offset=offset,seconds=time.perf_counter()-started)),flush=True)
    if any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()) or any(p.grad is not None for p in parameters):
        raise ValueError('Source diagnostic changed model or accumulated gradients')
    stats=summarize(records);coarse=stats['coarse/coarse_clean_IQ'];action=stats['action/action_probe']
    decision=dict(coarse_gradient_conflict=coarse['defined_cosines']==32 and coarse['negative_cosines']>=24,
        action_gradient_conflict=action['defined_cosines']==32 and action['negative_cosines']>=24,
        rule='Atleast24of32negativecosines inpredesignatedgroup;sourcehypothesisonly/notidentityorformalqualification')
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',source_run=SOURCE,source_commit=SOURCE_COMMIT,
        records=records,summary=stats,decision=decision,source_records=512,source_L=128,source_U=384,source_views=[1,20],batch_size=32,
        source_G_inputs=1024,smoke_G_inputs=2,new_F_inputs=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        model_unchanged=True,persistent_gradients=False,seconds=time.perf_counter()-started,
        scope='Fixedsource training-pool batchmean gradients/online localgeometry;notTXaccuracy/targeteffect orproofofalltrainingstages'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',source_G_inputs=1024,optimizer_updates=0,predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
