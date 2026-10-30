"""Frozen source counterfactuals for the matched direct-training variance ablation."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.pamf_dura_direct_train import runner as training
from experiments.pamf_dura_direct_train import fast_lr,no_variance
from experiments.pamf_dura_diagnostics.runner import diagnose_batch,source_indices,summarize,tensor
from .model import supervised_action

RUN='20261030-diagnostic-DURA-direct-physics-s2026092701-m2-r01'
MODELS=(('fastLR',fast_lr,'a0acfbe0c237e38d0460a9f1ea29c46b22cf84a6'),
        ('noVarAux',no_variance,'cf40edea90667d9645a724b1e8f333027cdf7869'))
GLOBALS=('RUN','configuration','command','learning_rate','scratch_model','direct_objective')


def load_source(root,module,commit,original):
    for key,value in original.items():setattr(training,key,value)
    module.configure()
    p=training.paths(root)
    done=training.read_json(p['output']/'scratch_joint/training_complete.json')
    expected=dict(status='SOURCE_TRAINED',epoch=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        training_TX_labels_read=0,training_F_access=False,V_reads=0,target_reads=0,
        learned_teacher_inputs=0,initialization_checkpoint_loads=0)
    if p['output'].name!=module.RUN or any(done.get(k)!=v for k,v in expected.items()):
        raise ValueError('Complete matching source-only single-stage E20 required')
    return training.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=commit),p


@torch.no_grad()
def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable source-only CPU diagnostic root required')
    original={key:getattr(training,key) for key in GLOBALS}
    p=training.paths(args.root)
    clean,roles=training.source_training(p);indices=source_indices(roles)
    if len(indices)!=512 or (roles[indices]=='L_s').sum()!=128 or (roles[indices]=='U_s').sum()!=384:
        raise ValueError('Original fixed 128Ls/384Us source panel required')
    args.output.mkdir()
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        commit=args.commit,CPU_threads=2,GPU_workers=0))
    np.save(args.output/'source_positions.npy',indices,allow_pickle=False)
    raw={};cells=[];started=time.perf_counter()
    for name,module,commit in MODELS:
        model,p=load_source(args.root,module,commit,original)
        before={key:value.detach().clone() for key,value in model.state_dict().items()}
        output=model(tensor(clean[indices[:2]],'cpu'))
        if output.shape!=(2,2,256) or not torch.isfinite(output).all():raise ValueError('Actual source checkpoint smoke failed')
        print(json.dumps(dict(event='SOURCE_CHECKPOINT_SMOKE_PASS',model=name,real_Ls=2,F_reads=0,target_reads=0)),flush=True)
        generator=torch.Generator().manual_seed(2026103007)
        for epoch in (1,20):
            folder=p['cache']/'epochs'/f'{epoch:02d}'
            arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)
                    for key in ('received','kernel','coefficients','variance')}
            batches=[]
            for offset in range(0,512,64):
                take=indices[offset:offset+64]
                values={key:tensor(value[take],'cpu') for key,value in arrays.items()}
                batches.append(diagnose_batch(model,tensor(clean[take],'cpu'),values['received'],
                    supervised_action(values['kernel'],values['coefficients']),values['variance'],generator))
            values={key:np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
            for key,value in values.items():raw[f'{name}/{epoch:02d}/{key}']=value
            for role,mask in (('all',np.ones(512,dtype=bool)),('L_s',roles[indices]=='L_s'),('U_s',roles[indices]=='U_s')):
                cells.append(dict(model=name,epoch=epoch,role=role,count=int(mask.sum()),metrics=summarize(values,mask)))
            print(json.dumps(dict(event='SOURCE_PHYSICS_COMPLETE',model=name,epoch=epoch,count=512,
                seconds=time.perf_counter()-started)),flush=True)
        if any(not torch.equal(value,before[key]) for key,value in model.state_dict().items()):
            raise ValueError('Diagnostic changed frozen model state')
        del model,before;gc.collect()
    for key,value in original.items():setattr(training,key,value)
    np.savez(args.output/'per_example.npz',**raw)
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',models=[dict(name=n,run=m.RUN,commit=c) for n,m,c in MODELS],
        source_records=512,source_L=128,source_U=384,epochs=[1,20],cells=cells,seconds=time.perf_counter()-started,
        model_unchanged=True,optimizer_updates=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,
        source_rule='Action bottleneck supported only if oracle-action-CG16 lowers IQ error >=10% versus learned-CG16 in both epochs and probe0 error >1%;not a target or deployment claim',
        scope='Fixed final prior counterfactuals; oracle operator/noise/clean prior are source diagnostics only; not accuracy bounds or legal deployment predictions'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',predictions=0,optimizer_updates=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
