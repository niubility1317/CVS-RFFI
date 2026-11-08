"""Train DURA/direct mixing gate on source records unseen by direct parent."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch

from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.signal_confidence_gate import source_gate_loss
from experiments.phase1_dura_orthogonal.cross_expert_refiner import CrossExpertRefiner
from experiments.phase1_dura_orthogonal.polyphase_cross_expert import PolyphaseCrossExpertRefiner
from experiments.phase1_dura_orthogonal.signal_gate_train import BATCH, EPOCHS, MODEL_SEED, ORDER_SEED
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import SOURCE as PARENT, SOURCE_COMMIT as PARENT_COMMIT, load_direct
from experiments.phase1_dura_orthogonal.train_source import load_source_training, prepare, extract_context

OOF_SIZE=17632
SCHEMA='phase1_cross_expert_refiner_v1'
MODEL_SEED=2026110830
ORDER_SEED=2026110831


def select_oof_ids(rx,day,roles,parent,pilot,*,expected_size=OOF_SIZE):
    if rx.shape!=day.shape or rx.shape!=roles.shape:
        raise ValueError('Matching source receiver, day and role arrays required')
    eligible=np.flatnonzero(~(np.isin(rx,(4,6,8)) & (day==3)))
    selected=np.setdiff1d(np.setdiff1d(eligible,pilot),parent)
    if (len(selected)!=expected_size or not np.isin(roles[selected],('L_s','U_s')).all()
            or np.intersect1d(selected,parent).size or np.intersect1d(selected,pilot).size):
        raise ValueError('Disjoint legally permitted source fit records required')
    return selected


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--implementation',choices=('native','polyphase'),default='native')
    args=p.parse_args()
    training_environments('cuda:0',os.environ)
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and one selected GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    pilot,_,_,_,_,_=source_inputs(args.root)
    state=torch.load(args.root/'runs'/PARENT/'final.pt',map_location='cpu',weights_only=False)
    parent=load_direct(args.root,pilot)
    parent_fit=np.asarray(state['fit_ids'],dtype=np.int64)
    cache,fresh,clean,expected_parent=load_source_training(args.root,pilot,fit_size=32768)
    if not np.array_equal(parent_fit,expected_parent):raise ValueError('Matching source parent fit required')
    rx=np.load(cache/'train/rx.npy',allow_pickle=False)
    day=np.load(cache/'train/day.npy',allow_pickle=False)
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    fit=select_oof_ids(rx,day,roles,parent_fit,pilot)
    parent=parent.to('cuda:0').eval().requires_grad_(False)
    del state
    dura=frozen_dura(args.root,'cuda:0')
    probe,_=prepare(args.root,dura,clean,fresh,fit[:32],'cuda:0',32,source_epochs=(21,))
    x=probe[0].to('cuda:0')
    with torch.no_grad():
        actual=extract_context(dura,x)
        for got,expected in zip(actual,probe[1:4]):
            torch.testing.assert_close(got,expected.to('cuda:0'),rtol=1e-4,atol=1e-5)
        first=parent(x,validate=False)
    if not torch.isfinite(first).all():raise FloatingPointError('Finite frozen source parent output required')
    torch.manual_seed(MODEL_SEED)
    model_class=PolyphaseCrossExpertRefiner if args.implementation=='polyphase' else CrossExpertRefiner
    smoke=model_class().to('cuda:0').train()
    smoke_output=smoke(x,*actual[:2],first,actual[2])
    torch.testing.assert_close(smoke_output,(actual[0]+first)*.5,rtol=1e-5,atol=3e-6)
    smoke_loss=source_gate_loss(smoke_output,actual[0],probe[4].to('cuda:0'))
    smoke_loss.backward()
    if not torch.isfinite(smoke_loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in (smoke.time_out.weight,smoke.freq_out.weight)):
        raise FloatingPointError('Finite real-source time/frequency forward and gradient required')
    del smoke,smoke_output,smoke_loss,probe,x,actual,first
    print('CROSS_EXPERT_SOURCE_SMOKE_PASS 32',flush=True)
    args.out.mkdir(parents=True)
    started=time.monotonic()
    data,counts=prepare(args.root,dura,clean,fresh,fit,'cuda:0',32,source_epochs=(21,22,23))
    del dura
    prepared=[]
    with torch.no_grad():
        for i in range(0,len(data[0]),32):
            received=data[0][i:i+32].to('cuda:0')
            value=parent(received,validate=False)
            if not torch.isfinite(value).all():raise FloatingPointError('Finite frozen source parent predictions required')
            prepared.append(value.cpu())
            if (i+len(value))%16384==0:print('CROSS_EXPERT_PARENT_PROGRESS',i+len(value),flush=True)
    proposal=torch.cat(prepared)
    del parent,prepared
    torch.cuda.empty_cache()
    torch.manual_seed(MODEL_SEED)
    gate=model_class().to('cuda:0').train()
    optimizer=torch.optim.AdamW(gate.parameters(),lr=.0003,weight_decay=.001)
    generator=torch.Generator().manual_seed(ORDER_SEED)
    losses=[]
    for epoch in range(EPOCHS):
        order=torch.randperm(len(data[0]),generator=generator)
        total=torch.zeros((),device='cuda:0')
        for begin in range(0,len(order),BATCH):
            idx=order[begin:begin+BATCH]
            received,anchor,innovation,variance,clean_iq=[v[idx].to('cuda:0') for v in data]
            target=proposal[idx].to('cuda:0')
            output=gate(received,anchor,innovation,target,variance)
            loss=source_gate_loss(output,anchor,clean_iq)
            if not torch.isfinite(loss):raise FloatingPointError('Finite source gate loss required')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            total+=loss.detach()*len(idx)
        torch.cuda.synchronize()
        losses.append(float(total.cpu())/len(order))
        print('CROSS_EXPERT_SOURCE_EPOCH',epoch+1,'loss',losses[-1],'elapsed',round(time.monotonic()-started,1),flush=True)
    result=dict(schema=SCHEMA,commit=args.commit,implementation=args.implementation,parent=PARENT,parent_commit=PARENT_COMMIT,
        fit_ids=fit.tolist(),epochs=EPOCHS,model_seed=MODEL_SEED,order_seed=ORDER_SEED,batch=BATCH,
        parent_fit_disjoint=True,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,DURA_frozen=True,
        source_epochs=[21,22,23],model=gate.eval().cpu().state_dict())
    with (args.out/'final.pt').open('xb') as f:torch.save(result,f)
    summary={k:v for k,v in result.items() if k not in ('model','fit_ids')}
    summary.update(status='SOURCE_TRAINED',fit_ids=len(fit),counts=counts,losses=losses)
    (args.out/'source_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print('CROSS_EXPERT_SOURCE_TRAINED',args.commit,flush=True)


if __name__=='__main__':main()
