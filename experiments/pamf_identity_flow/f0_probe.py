"""Two-record source-only feasibility check of the actual frozen Phase3 F0.

The original F0 and the random flow are loaded before reading allowed source
IQ. This is not a training run, a target evaluation, or a tuned loss weight.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import IdentityConditionalFlow, source_flow_loss, frozen_identity_consistency
from experiments.pamf_dura_direct_train import runner as source
from experiments.pamf_dura_direct_train.f_guided import source_ids
from experiments.pamf_phase3_open_world import star_relative_source as original_f0

RUN='20261101-diagnostic-identity-flow-F0-source-s2026092701-r01'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference-runtime',type=Path,required=True)
    parser.add_argument('--code-commit',required=True)
    args=parser.parse_args();root=args.root.resolve()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.code_commit:
        raise ValueError('Pinned diagnostic release required')
    out=root/'runs'/RUN
    if out.exists():raise FileExistsError('Diagnostic output already exists')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtime=json.loads(args.reference_runtime.read_text(encoding='utf-8'))
    base=runtime['eval_config']['reference_config']['source_control_config']['reference_config']
    native=base['train_config']['native_config']
    spec=dict(reader_config={k:native[k] for k in ('base_release','frozen_sources','source_output','model_seed')},
              source_contract=native['source_contract'])
    # The original loader verifies F0 architecture, seed, source contract and
    # source checkpoint ancestry. No DURA or target model is loaded.
    loaded=original_f0._backend(spec,'cpu',with_dura=False)
    teacher=loaded['reader'].eval().requires_grad_(False)
    identity=loaded['reader_identity']
    if (identity['parameters'],identity['epoch'],identity['updates']) != (232993,50,11100):
        raise ValueError('Actual original Phase3 F0 E50 required')
    p=source.paths(root);clean,roles=source.source_training(p);ids=source_ids(p,roles)
    contract=original_f0._contract(spec)
    for role in ('L_s','U_s'):
        if set(ids[roles==role].tolist())!=set(contract['role_ids'][role]):
            raise ValueError('Restorer and F0 source physical roles differ')
    cache=p['cache']/'epochs/01'
    variance=np.load(cache/'variance.npy',mmap_mode='r',allow_pickle=False)
    selected=np.flatnonzero((roles=='L_s')&(variance>np.float32(1e-8)))[:2]
    if len(selected)!=2:raise ValueError('Two fixed nonclean source L_s records required')
    x=source.tensor(clean[selected],'cpu')
    y=source.tensor(np.load(cache/'received.npy',mmap_mode='r',allow_pickle=False)[selected],'cpu')
    torch.manual_seed(2026092701);model=IdentityConditionalFlow()
    g=torch.Generator().manual_seed(2026110102)
    noise_a=torch.randn(y.shape,generator=g);noise_b=torch.randn(y.shape,generator=g)
    t=torch.tensor([.25,.75])
    loss_a,end_a=source_flow_loss(model,x,y,t,noise_a)
    loss_b,end_b=source_flow_loss(model,x,y,t,noise_b)
    before={k:v.detach().clone() for k,v in teacher.state_dict().items()}
    anchor,agreement=frozen_identity_consistency(teacher,x,end_a,end_b)
    gradients=torch.autograd.grad((anchor+agreement).sum(),(end_a,end_b),retain_graph=True)
    field_grad=torch.autograd.grad((loss_a+loss_b+anchor+agreement).sum(),model.output.weight)[0]
    if (not all(torch.isfinite(v).all() and v.abs().sum()>0 for v in gradients)
            or not torch.isfinite(field_grad).all() or not field_grad.abs().sum()>0):
        raise ValueError('Actual F0 guidance and source flow gradients must be finite and nonzero')
    if any(v.grad is not None for v in teacher.parameters()) or any(not torch.equal(v,before[k]) for k,v in teacher.state_dict().items()):
        raise ValueError('Frozen F0 parameters and buffers must remain unchanged')
    out.mkdir()
    result=dict(status='REAL_F0_SOURCE_GRADIENT_PASS',run_id=RUN,code_commit=args.code_commit,
        source_count=2,source_roles=['L_s'],source_views='original R07 source view01; not G10',
        F0=identity,F0_source_roles_match=True,F0_frozen=True,F0_updates=0,optimizer_updates=0,
        model_initialization='random scratch, zero output',parameters=sum(p.numel() for p in model.parameters()),
        source_flow_loss=[float(loss_a.detach().mean()),float(loss_b.detach().mean())],
        feature_anchor=float(anchor.detach().mean()),feature_agreement=float(agreement.detach().mean()),
        endpoint_gradient_norms=[float(v.detach().norm()) for v in gradients],
        field_gradient_norm=float(field_grad.detach().norm()),
        F_forward_rows=6,G_forward_rows=4,TX_labels_read=0,V_IQ_reads=0,OE_IQ_reads=0,target_IQ_reads=0,
        caveat='Two-source numerical compatibility only; no performance or loss-weight claim')
    with (out/'complete.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
