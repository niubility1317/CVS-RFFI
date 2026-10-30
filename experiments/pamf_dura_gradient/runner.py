"""Measure competing source gradients without F, labels or optimizer updates."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.pamf_dura_direct_train import fast_lr
from experiments.pamf_dura_direct_train import runner as training
from experiments.pamf_dura_efficiency.model import direct_objective,supervised_action
from experiments.pamf_dura_diagnostics.runner import tensor,source_indices

RUN='20261030-diagnostic-DURA-joint-gradient-s2026092701-r01'
SOURCE='20261030-phase1-DURA-direct-fastlr-s2026092701-r01'
SOURCE_COMMIT='a0acfbe0c237e38d0460a9f1ea29c46b22cf84a6'
WEIGHTS={'iq_nmse':1.,'prior_nmse':.1,'spectral_error':.05,'action_probe_nmse':.05,'log_variance_mse':.001}


def geometry(left,right):
    a=sum(float(value.detach().double().square().sum()) for value in left)
    b=sum(float(value.detach().double().square().sum()) for value in right)
    dot=sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(left,right))
    cosine=dot/math.sqrt(a*b) if a>0 and b>0 else None
    return dict(left_norm=math.sqrt(a),right_norm=math.sqrt(b),dot=dot,cosine=cosine,
                right_to_left_norm=math.sqrt(b/a) if a>0 else None)


def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable source-only CPU diagnostic root required')
    fast_lr.configure()
    p=training.paths(args.root)
    if p['output'].name!=SOURCE:raise ValueError('Only predesignated single-stage source model allowed')
    done=training.read_json(p['output']/'scratch_joint/training_complete.json')
    expected=dict(status='SOURCE_TRAINED',epoch=20,updates=4940,L_exposures=126000,U_exposures=1134000,
                  training_TX_labels_read=0,training_F_access=False,V_reads=0,target_reads=0,
                  learned_teacher_inputs=0,initialization_checkpoint_loads=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Complete original source-only E20 model required')
    model=training.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=SOURCE_COMMIT)
    model.eval().requires_grad_(True)
    clean,roles=training.source_training(p);indices=source_indices(roles)
    before={key:value.detach().clone() for key,value in model.state_dict().items()}
    args.output.mkdir(parents=True)
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        source_run=SOURCE,source_commit=SOURCE_COMMIT,CPU_threads=2,GPU_workers=0,owner='codex/root/dura-source-gradients'))
    with torch.no_grad():
        output=model(tensor(clean[indices[:2]],'cpu'))
        if output.shape!=(2,2,256) or not torch.isfinite(output).all():raise ValueError('Actual 2Ls checkpoint noquery smoke failed')
    print(json.dumps(dict(event='SOURCE_CHECKPOINT_SMOKE_PASS',real_Ls=2,F_reads=0,target_reads=0)),flush=True)
    names,parameters=zip(*model.named_parameters())
    groups={'all':list(range(len(names)))}
    for prefix in ('frozen.coarse.','frozen.estimator.','prior.'):
        groups[prefix]=[i for i,name in enumerate(names) if name.startswith(prefix)]
    records=[];started=time.perf_counter();generator=torch.Generator().manual_seed(2026103005)
    for epoch in (1,20):
        folder=p['cache']/'epochs'/f'{epoch:02d}'
        arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('received','kernel','coefficients','variance')}
        for offset in range(0,len(indices),32):
            take=indices[offset:offset+32]
            values={key:tensor(value[take],'cpu') for key,value in arrays.items()}
            actual=supervised_action(values['kernel'],values['coefficients'])
            probes=[torch.complex(torch.randn(len(take),256,generator=generator),torch.randn(len(take),256,generator=generator)) for _ in range(2)]
            loss,terms=direct_objective(model,values['received'],tensor(clean[take],'cpu'),actual,values['variance'],probes)
            gradients={}
            for name,weight in WEIGHTS.items():
                value=terms[name].mean()*weight
                gradient=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
                gradients[name]=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,gradient)]
                if any(not torch.isfinite(g).all() for g in gradients[name]):raise FloatingPointError('Nonfinite source component gradient')
            auxiliary=[sum(gradients[key][i] for key in WEIGHTS if key!='iq_nmse') for i in range(len(parameters))]
            geometry_groups={}
            for group,selected in groups.items():
                iq=[gradients['iq_nmse'][i] for i in selected]
                comparisons={key:geometry(iq,[value[i] for i in selected]) for key,value in gradients.items() if key!='iq_nmse'}
                comparisons['combined_auxiliary']=geometry(iq,[auxiliary[i] for i in selected])
                geometry_groups[group]=comparisons
            record=dict(epoch=epoch,offset=offset,count=len(take),L_s=int((roles[take]=='L_s').sum()),
                U_s=int((roles[take]=='U_s').sum()),source_positions=take.tolist(),
                losses={key:float(value.detach().mean()) for key,value in terms.items()},geometry=geometry_groups)
            records.append(record)
            print(json.dumps(dict(event='SOURCE_GRADIENT_BATCH',epoch=epoch,count=offset+len(take),seconds=time.perf_counter()-started)),flush=True)
    if any(not torch.equal(value,before[key]) for key,value in model.state_dict().items()):raise ValueError('Diagnostic changed model state')
    if any(p.grad is not None for p in model.parameters()):raise ValueError('Diagnostic accumulated persistent gradients')
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',source_run=SOURCE,weights=WEIGHTS,
        records=records,batch_size=32,source_records=512,source_L=128,source_U=384,epochs=[1,20],
        model_unchanged=True,optimizer_updates=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,
        seconds=time.perf_counter()-started,scope='Batch-mean weighted gradient geometry, not per-example gradients or target/identity evidence'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',optimizer_updates=0,predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned code required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
