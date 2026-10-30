"""Source-only initial gradient conditioning; no optimizer or F access."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments.pamf_dura_direct_train import fast_lr
from experiments.pamf_dura_direct_train import runner as training
from experiments.pamf_dura_direct_train.identity_init import initialize_identity
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor
from experiments.pamf_dura_gradient.runner import geometry,WEIGHTS,SOURCE_COMMIT
from .model import scratch_model,direct_objective,supervised_action

RUN='20261030-diagnostic-DURA-initial-condition-s2026092701-m2-r01'
POLICIES={'base':dict(WEIGHTS),'strong_action':dict(WEIGHTS,action_probe_nmse=.2),
          'no_variance':dict(WEIGHTS,log_variance_mse=0.),
          'no_variance_strong_action':dict(WEIGHTS,log_variance_mse=0.,action_probe_nmse=.2)}


def run(args):
    if args.output.name!=RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable source-only CPU diagnostic root required')
    fast_lr.configure();p=training.paths(args.root)
    done=training.read_json(p['output']/'scratch_joint/training_complete.json')
    expected=dict(status='SOURCE_TRAINED',epoch=20,updates=4940,training_TX_labels_read=0,
        training_F_access=False,V_reads=0,target_reads=0,learned_teacher_inputs=0,initialization_checkpoint_loads=0)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Legal source checkpoint required')
    clean,roles=training.source_training(p);indices=source_indices(roles)
    real=training.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=SOURCE_COMMIT)
    with torch.no_grad():
        value=real(tensor(clean[indices[:2]],'cpu'))
        if value.shape!=(2,2,256) or not torch.isfinite(value).all():raise ValueError('Real source checkpoint noquery smoke failed')
    del real;gc.collect()
    args.output.mkdir()
    training.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        commit=args.commit,CPU_threads=2,GPU_workers=0))
    print(json.dumps(dict(event='REAL_SOURCE_CHECKPOINT_NOQUERY_SMOKE_PASS',real_Ls=2,F_reads=0,target_reads=0)),flush=True)
    records=[];started=time.perf_counter()
    for kind in ('ordinary','identity'):
        model=scratch_model(2026092701)
        if kind=='identity':initialize_identity(model)
        model.eval().requires_grad_(True)
        before={key:value.detach().clone() for key,value in model.state_dict().items()}
        parameters=list(model.parameters());generator=torch.Generator().manual_seed(2026103005)
        for epoch in (1,20):
            folder=p['cache']/'epochs'/f'{epoch:02d}'
            arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)
                    for key in ('received','kernel','coefficients','variance')}
            for offset in range(0,512,32):
                take=indices[offset:offset+32]
                values={key:tensor(value[take],'cpu') for key,value in arrays.items()}
                probes=[torch.complex(torch.randn(32,256,generator=generator),torch.randn(32,256,generator=generator)) for _ in range(2)]
                _,terms=direct_objective(model,values['received'],tensor(clean[take],'cpu'),
                    supervised_action(values['kernel'],values['coefficients']),values['variance'],probes)
                gradients={}
                for key in WEIGHTS:
                    derivative=torch.autograd.grad(terms[key].mean(),parameters,retain_graph=True,allow_unused=True)
                    gradients[key]=[torch.zeros_like(p) if v is None else v.detach() for p,v in zip(parameters,derivative)]
                    if any(not torch.isfinite(v).all() for v in gradients[key]):raise FloatingPointError('Nonfinite source gradient')
                policies={}
                for name,weights in POLICIES.items():
                    combined=[sum(weights[key]*gradients[key][i] for key in weights) for i in range(len(parameters))]
                    stat=geometry(gradients['iq_nmse'],combined)
                    policies[name]=dict(raw_norm=stat['right_norm'],cosine_to_IQ=stat['cosine'],
                        clipping_would_apply=stat['right_norm']>1.,clip_scale=min(1.,1./(stat['right_norm']+1e-6)),
                        loss=sum(weights[key]*float(terms[key].detach().mean()) for key in weights))
                records.append(dict(initialization=kind,epoch=epoch,offset=offset,count=len(take),
                    L_s=int((roles[take]=='L_s').sum()),U_s=int((roles[take]=='U_s').sum()),source_positions=take.tolist(),
                    losses={key:float(v.detach().mean()) for key,v in terms.items()},policies=policies))
                print(json.dumps(dict(event='INITIAL_GRADIENT_BATCH',initialization=kind,epoch=epoch,
                    count=offset+32,seconds=time.perf_counter()-started)),flush=True)
        if any(not torch.equal(value,before[key]) for key,value in model.state_dict().items()):raise ValueError('Initial model changed')
        if any(p.grad is not None for p in parameters):raise ValueError('Persistent gradient accumulated')
        del model,before,parameters;gc.collect()
    training.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',records=records,policies=POLICIES,
        source_records=512,source_L=128,source_U=384,epochs=[1,20],model_seed=2026092701,seconds=time.perf_counter()-started,
        model_unchanged=True,optimizer_updates=0,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,
        source_rule='Identity conditioning supported only if median no_variance raw-gradient norm is <= half ordinary and mean no_variance loss is lower at both cached epochs;otherwise no combined initializer candidate from this diagnostic',
        scope='Untrained initial states, not epoch-trained states; no-gradient-step nor convergence/identity proof; clipping estimate uses FP64 norm before any optimizer'))
    training.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',optimizer_updates=0,predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
