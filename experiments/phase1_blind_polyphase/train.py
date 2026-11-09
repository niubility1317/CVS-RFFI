"""F-free and DURA-free source training for blind unrolled IQ recovery."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.train_source import load_source_training, prepare
from experiments.phase1_blind_unrolled.model import WIDTH, STEPS, TAPS
from experiments.phase1_blind_polyphase.model import BlindPolyphaseRecovery

SCHEMA='phase1_blind_polyphase_source_v1'
EPOCHS=20
BATCH=32
MODEL_SEED=2026110903
ORDER_SEED=2026110904
PHYSICS_WEIGHT=.02


def source_data(root, smoke):
    pilot,_,_,_,_,_=source_inputs(root)
    cache,fresh,clean,fit=load_source_training(root,pilot,fit_size=32768)
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if smoke:
        selected=np.concatenate((fit[roles[fit]=='L_s'][:32],fit[roles[fit]=='U_s'][:32]))
        if len(selected)!=64:raise ValueError('Mixed legal source roles required')
    else:
        selected=fit
    data,counts=prepare(root,None,clean,fresh,selected,'cuda:0',32,
                        architecture='direct_spectral_student',source_epochs=(21,22,23))
    return data[0].to('cuda:0'),data[4].to('cuda:0'),fit,selected,counts


def loss_terms(restored,projected,received,clean):
    clean_energy=clean.square().mean((1,2)).clamp_min(1e-8)
    input_energy=received.square().mean((1,2)).clamp_min(1e-8)
    iq=((restored-clean).square().mean((1,2))/clean_energy).mean()
    physics=((projected-received).square().mean((1,2))/input_energy).mean()
    return iq,physics


def run(root,out,commit,smoke=False):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    training_environments('cuda:0',os.environ)
    if out.exists() or not torch.cuda.is_available():raise ValueError('Unique output and selected GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(MODEL_SEED)
    received,clean,fit,selected,counts=source_data(root,smoke)
    model=BlindPolyphaseRecovery().to('cuda:0').train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    rng=np.random.default_rng(ORDER_SEED)
    out.mkdir(parents=True)
    start_time=time.monotonic();losses=[]
    for epoch in range(1,2 if smoke else EPOCHS+1):
        order=rng.permutation(len(received))
        total=torch.zeros((),device='cuda:0');physics_total=torch.zeros((),device='cuda:0')
        for start in range(0,len(order),BATCH):
            idx=torch.tensor(order[start:start+BATCH].tolist(),dtype=torch.long,device='cuda:0')
            y,target=received[idx],clean[idx]
            restored,projected=model(y,return_physics=True)
            iq,physics=loss_terms(restored,projected,y,target)
            loss=iq+PHYSICS_WEIGHT*physics
            if not torch.isfinite(loss):raise FloatingPointError('Finite source loss required')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if start==0 and (model.priors[0][-1].weight.grad is None or
                            model.priors[0][-1].weight.grad.abs().sum()<=0):
                raise ValueError('Finite nonzero recovery gradient required')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();total+=loss.detach()*len(idx);physics_total+=physics.detach()*len(idx)
        torch.cuda.synchronize();losses.append(float(total.cpu())/len(order))
        print('BLIND_POLYPHASE_SOURCE_EPOCH',epoch,'loss',losses[-1],'physics',float(physics_total.cpu())/len(order),
              'elapsed',round(time.monotonic()-start_time,1),flush=True)
    common=dict(schema=SCHEMA,commit=commit,epochs=1 if smoke else EPOCHS,fit_ids=len(fit),
        actual_fit_ids=len(selected),actual_source_view_count=len(received),views_per_record=3,
        width=WIDTH,steps=STEPS,taps=TAPS,physics_weight=PHYSICS_WEIGHT,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        model_seed=MODEL_SEED,order_seed=ORDER_SEED,
        source_loss='normalized paired IQ + .02 normalized blind-channel reprojection',
        inference_received_IQ_only=True,DURA_training_reads=0,TX_labels_read=0,F_weights_read=0,
        F_outputs_read=0,V_reads=0,target_reads=0,source_counts=counts,
        elapsed_seconds=time.monotonic()-start_time,losses=losses)
    with (out/('smoke.pt' if smoke else 'final.pt')).open('xb') as f:
        torch.save(dict(**common,model=model.eval().cpu().state_dict()),f)
    core.write(out/('smoke.json' if smoke else 'source_summary.json'),dict(
        status='SOURCE_TRAINING_SMOKE_PASS' if smoke else 'SOURCE_TRAINED',**common))
    print('BLIND_POLYPHASE_SOURCE_DONE',commit,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--device',choices=('cuda:0',),default='cuda:0')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.root,a.out,a.commit,a.smoke)

if __name__=='__main__':main()
