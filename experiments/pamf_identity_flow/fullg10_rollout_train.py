"""Source-trained F0-guided full-rollout flow; no target access."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .model import IdentityConditionalFlow
from .training import FixedAntitheticFlow
from .fullg10_data import validate_manifest, mix_full_g10
from .rollout_objective import rollout_objective
from experiments.pamf_dura_direct_train import runner as source
from experiments.pamf_phase3_open_world import star_relative_source as f0

RUN='20261101-phase3-identity-rollout-fullG10-E30-s2026092701-r02'
ARM='identity_rollout_fullg10'
EPOCHS=30
BATCH=128
STEPS=EPOCHS*math.ceil(63000/BATCH)
G10_RUN='20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01'


def make_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026092701)
        return IdentityConditionalFlow()


def config():
    return dict(run=RUN,epochs=EPOCHS,batch=BATCH,updates=STEPS,seed_model=2026092701,
        seed_order=2026110104,seed_flow=2026110105,seed_inference=2026110103,
        source_examples_per_epoch=63000,source_mix=dict(g10=47250,clean=15750),
        g10_sample_rule='one permutation of all L_s/U_s; first 75% G10, remaining 25% clean',
        G10_view='one fixed MIX01 source L_s/U_s view, 2.45 GHz, repeated; not independent observations',
        objective='mean(two full-rollout IQ NMSE)+F0 feature anchor+0.25*F0 path agreement',
        time='four fixed Euler steps through both training and inference',noise='one Gaussian and its negative, scale 0.02',
        optimizer='AdamW',peak_lr=.0003,final_lr=.00002,warmup=493,weight_decay=.0001,
        gradient_clip=1.,final_checkpoint='fixed E30; no target-dependent selection',
        inference='mean of two antithetic fixed common noise templates, scale 0.02, 4 Euler steps each',
        teacher='original Phase3 source F0 E50/11100 frozen; gradients only through input',
        source_scope='L_s/U_s only, no TX labels, V/OE/novel/target IQ or threshold updates')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def teacher_and_data(root, runtime, device):
    native=_read(runtime)['eval_config']['reference_config']['source_control_config']['reference_config']['train_config']['native_config']
    spec=dict(reader_config={k:native[k] for k in ('base_release','frozen_sources','source_output','model_seed')},source_contract=native['source_contract'])
    backend=f0._backend(spec,device,with_dura=False);teacher=backend['reader'].eval().requires_grad_(False);identity=backend['reader_identity']
    if (identity['parameters'],identity['epoch'],identity['updates'])!=(232993,50,11100):raise ValueError('Original Phase3 F0 E50 required')
    parent=root/'runs'/G10_RUN;cache=parent/'cache';done=_read(parent/'complete.json');manifest=_read(cache/'manifest.json')
    validate_manifest(manifest)
    if done.get('status')!='P3_RESTORE_SOURCE_CACHE_COMPLETE' or done.get('run_id')!=G10_RUN or done.get('target_inputs')!=0:
        raise ValueError('Completed full G10 source cache required')
    if any(manifest.get('F0_identity',{}).get(k)!=v for k,v in identity.items()):raise ValueError('Cache and reader F0 identities differ')
    ids=np.load(cache/'ids.npy',allow_pickle=False);roles=np.load(cache/'roles.npy',allow_pickle=False)
    if ids.shape!=(63000,) or len(set(ids.tolist()))!=63000 or roles.shape!=(63000,):raise ValueError('Distinct complete source identities required')
    contract=f0._contract(spec)
    for role,count in (('L_s',6300),('U_s',56700)):
        if int((roles==role).sum())!=count or ids[roles==role].tolist()!=contract['role_ids'][role]:raise ValueError('Exact source role and physical ID order required')
    clean=np.load(cache/'clean.npy',mmap_mode='r',allow_pickle=False);received=np.load(cache/'received.npy',mmap_mode='r',allow_pickle=False)
    if any(a.shape!=(63000,2,256) or a.dtype!=np.float32 for a in (clean,received)):raise ValueError('Complete FP32 G10 source pairs required')
    return {'cache':cache},clean,roles,clean,received,teacher,identity


def learning_rate(update):
    if not 0<=update<STEPS:raise ValueError('Fixed update range required')
    if update<493:return .0003*(update+1)/493
    fraction=(update-493)/(STEPS-493-1)
    return .00002+(.0003-.00002)*.5*(1+math.cos(math.pi*fraction))


def save(path,model,commit,epoch,updates):
    payload=dict(schema='identity_fullg10_rollout_source_v1',run_id=RUN,code_commit=commit,config=config(),
        epoch=epoch,updates=updates,model=model.state_dict(),TX_labels_read=0,V_IQ_reads=0,
        OE_IQ_reads=0,target_IQ_reads=0,teacher='frozen_original_Phase3_F0')
    with Path(path).open('xb') as stream:torch.save(payload,stream)


def load(path,device,commit,epoch=EPOCHS):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    expected=dict(schema='identity_fullg10_rollout_source_v1',run_id=RUN,code_commit=commit,config=config(),
        epoch=epoch,updates=epoch*math.ceil(63000/BATCH),TX_labels_read=0,V_IQ_reads=0,
        OE_IQ_reads=0,target_IQ_reads=0,teacher='frozen_original_Phase3_F0')
    if any(payload.get(k)!=v for k,v in expected.items()):raise ValueError('Exact fixed source conditional-flow checkpoint required')
    model=make_model();model.load_state_dict(payload['model'],strict=True)
    return FixedAntitheticFlow(model).to(device).eval().requires_grad_(False)


def smoke(root,runtime,commit,output):
    p,clean,roles,g_clean,g_received,teacher,identity=teacher_and_data(root,runtime,'cpu')
    model=make_model();x=source.tensor(g_clean[:2],'cpu');y=source.tensor(g_received[:2],'cpu')
    generator=torch.Generator().manual_seed(2026110105)
    noise=torch.randn(y.shape,generator=generator)
    snapshot={k:v.detach().clone() for k,v in teacher.state_dict().items()}
    loss,parts=rollout_objective(model,teacher,x,y,noise)
    identity_grad=torch.autograd.grad(parts['identity'].sum(),model.output.weight,retain_graph=True)[0]
    flow_grad=torch.autograd.grad(parts['iq'].sum(),model.output.weight,retain_graph=True)[0]
    if any(not torch.isfinite(g).all() or not g.abs().sum()>0 for g in (identity_grad,flow_grad)):
        raise ValueError('Two independent nonzero source gradients required')
    loss.mean().backward();torch.optim.AdamW(model.parameters(),lr=.0003).step()
    if any(v.grad is not None for v in teacher.parameters()) or any(not torch.equal(v,snapshot[k]) for k,v in teacher.state_dict().items()):
        raise ValueError('Teacher must stay frozen')
    path=output/'smoke.pt';save(path,model,commit,0,0)
    restored=load(path,'cpu',commit,epoch=0)
    with torch.no_grad():
        actual=restored(y);expected=FixedAntitheticFlow(model).eval()(y)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        features=teacher.features(actual)
    if not torch.isfinite(actual).all() or not torch.isfinite(features).all():raise ValueError('Finite actual F0/G10 checkpoint smoke required')
    _write(output/'smoke.json',dict(status='PASS',actual_F0=True,actual_G10_L_s=2,checkpoint_reload=True,
        identity_gradient_norm=float(identity_grad.norm()),rollout_IQ_gradient_norm=float(flow_grad.norm()),
        source_only=True,TX_labels_read=0,V_IQ_reads=0,OE_IQ_reads=0,target_IQ_reads=0))


def train(root,runtime,commit,output):
    p,clean,roles,g_clean,g_received,teacher,identity=teacher_and_data(root,runtime,'cuda:0')
    before={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
    model=make_model().cuda().train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    torch.manual_seed(2026110105);torch.cuda.manual_seed_all(2026110105)
    rng=np.random.default_rng(2026110104);start=time.perf_counter();updates=0
    row=output/ARM;row.mkdir();_write(row/'config.json',config())
    torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,EPOCHS+1):
            order=rng.permutation(63000)
            x,y,counts=mix_full_g10(clean,g_received,order)
            if int((roles[order]=='L_s').sum())!=6300 or int((roles[order]=='U_s').sum())!=56700:raise ValueError('Full source roles per epoch required')
            totals=dict(loss=0.,iq=0.,identity=0.,anchor=0.,agreement=0.);clipped=0
            for offset in range(0,63000,BATCH):
                a=source.tensor(x[offset:offset+BATCH],'cuda:0');b=source.tensor(y[offset:offset+BATCH],'cuda:0')
                noise=torch.randn_like(a)
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(updates)
                loss,terms=rollout_objective(model,teacher,a,b,noise)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source objective')
                loss.mean().backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates+=1;clipped+=int(norm>1)
                for key,value in dict(loss=loss,**{k:terms[k] for k in ('iq','identity','anchor','agreement')}).items():totals[key]+=float(value.detach().sum())
            record=dict(epoch=epoch,source_cache_view='fixed_G10_MIX01',updates=updates,source_mix=counts,source_roles=dict(L_s=6300,U_s=56700),losses={k:v/63000 for k,v in totals.items()},
                clipped_fraction=clipped/math.ceil(63000/BATCH),seconds=time.perf_counter()-start)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();print(json.dumps(record),flush=True)
    if updates!=STEPS:raise ValueError('Fixed E30 source budget incomplete')
    if any(v.grad is not None for v in teacher.parameters()) or any(not torch.equal(v.detach().cpu(),before[k]) for k,v in teacher.state_dict().items()):
        raise ValueError('F0 must remain frozen throughout training')
    save(row/'final.pt',model,commit,EPOCHS,updates)
    _write(row/'training_complete.json',dict(status='SOURCE_TRAINED',epoch=EPOCHS,updates=updates,
        source_mix_per_epoch=counts,F0=identity,F0_frozen=True,F0_updates=0,TX_labels_read=0,
        V_IQ_reads=0,OE_IQ_reads=0,target_IQ_reads=0,parameters=sum(v.numel() for v in model.parameters()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-start))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('smoke','train'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--reference-runtime',type=Path,required=True)
    parser.add_argument('--code-commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=args.code_commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output=args.root/'runs'/RUN
    if not output.is_dir():raise ValueError('Controller must create the unique registered run root')
    if args.mode=='smoke':smoke(args.root,args.reference_runtime,args.code_commit,output)
    else:train(args.root,args.reference_runtime,args.code_commit,output)


if __name__=='__main__':main()
