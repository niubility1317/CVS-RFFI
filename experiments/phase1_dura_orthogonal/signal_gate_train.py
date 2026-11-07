"""Train a confidence gate solely from paired source IQ and frozen DURA context."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch

from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.large_context_stable import StableLargeContextDura
from experiments.phase1_dura_orthogonal.signal_confidence_gate import SignalConfidenceGate, source_gate_loss
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.train_source import extract_context

PARENT = '20261108-phase1-dura-large-context-stable-source-s2026092701-r01'
PARENT_COMMIT = 'b873acec23758690e5484bd1c5f82162739c6655'
MODEL_SEED = 2026110826
ORDER_SEED = 2026110827
EPOCHS = 20
BATCH = 128


def validate_parent(state, pilot):
    required = dict(schema='phase1_dura_orthogonal_source_v1', commit=PARENT_COMMIT,
        architecture='large_context_stable', objective='wide_large_context_stable', epochs=20,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True)
    fit = np.asarray(state.get('fit_ids', []), dtype=np.int64)
    if (any(state.get(k) != v for k,v in required.items()) or len(fit) != 32768
            or len(np.unique(fit)) != len(fit) or np.intersect1d(fit, pilot).size):
        raise ValueError('Matching F-free source parent and pilot-disjoint fit required')
    return fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    training_environments('cuda:0', os.environ)
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and one selected GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    pilot, _, _, _, _, _ = source_inputs(args.root)
    state=torch.load(args.root/'runs'/PARENT/'final.pt',map_location='cpu',weights_only=False)
    fit=validate_parent(state,pilot)
    data,counts=load_cache(args.cache,fit,32,'cuda:0',deterministic=True)
    parent=StableLargeContextDura(width=state['width'],layers=state['large_context_layers'])
    parent.load_state_dict(state['model'],strict=True)
    parent=parent.to('cuda:0').eval().requires_grad_(False)
    del state
    dura=frozen_dura(args.root,'cuda:0')
    x=data[0][:32].to('cuda:0')
    with torch.no_grad():
        actual=extract_context(dura,x)
        for got,expected in zip(actual,data[1:4]):
            torch.testing.assert_close(got,expected[:32].to('cuda:0'),rtol=1e-4,atol=1e-5)
        first=parent(x,*[v[:32].to('cuda:0') for v in data[1:4]])
    if not torch.isfinite(first).all():
        raise FloatingPointError('Finite source parent output required')
    del dura,x,actual,first
    print('SIGNAL_GATE_SOURCE_SMOKE_PASS 32',flush=True)
    args.out.mkdir(parents=True)
    prepared=[]
    start=time.monotonic()
    with torch.no_grad():
        for i in range(0,len(data[0]),32):
            part=[v[i:i+32].to('cuda:0') for v in data[:4]]
            value=parent(*part)
            if not torch.isfinite(value).all():
                raise FloatingPointError('Finite source parent predictions required')
            prepared.append(value.cpu())
            if (i+len(value))%16384==0:
                print('SIGNAL_GATE_PARENT_PROGRESS',i+len(value),flush=True)
    proposal=torch.cat(prepared)
    del parent,prepared
    torch.cuda.empty_cache()
    torch.manual_seed(MODEL_SEED)
    gate=SignalConfidenceGate().to('cuda:0').train()
    optimizer=torch.optim.AdamW(gate.parameters(),lr=.0003,weight_decay=.001)
    generator=torch.Generator().manual_seed(ORDER_SEED)
    losses=[]
    for epoch in range(EPOCHS):
        order=torch.randperm(len(data[0]),generator=generator)
        total=torch.zeros((),device='cuda:0')
        for begin in range(0,len(order),BATCH):
            idx=order[begin:begin+BATCH]
            received,anchor,innovation,variance,clean=[v[idx].to('cuda:0') for v in data]
            target=proposal[idx].to('cuda:0')
            output=gate(received,anchor,innovation,target,variance)
            loss=source_gate_loss(output,anchor,clean)
            if not torch.isfinite(loss):
                raise FloatingPointError('Finite source gate loss required')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach()*len(idx)
        torch.cuda.synchronize()
        losses.append(float(total.cpu())/len(order))
        print('SIGNAL_GATE_SOURCE_EPOCH',epoch+1,'loss',losses[-1],'elapsed',round(time.monotonic()-start,1),flush=True)
    result=dict(schema='phase1_signal_confidence_gate_v1',commit=args.commit,parent=PARENT,parent_commit=PARENT_COMMIT,
        fit_ids=fit.tolist(),epochs=EPOCHS,model_seed=MODEL_SEED,order_seed=ORDER_SEED,batch=BATCH,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,DURA_frozen=True,
        source_context_cache=str(args.cache),model=gate.eval().cpu().state_dict())
    with (args.out/'final.pt').open('xb') as f:torch.save(result,f)
    summary={k:v for k,v in result.items() if k not in ('model','fit_ids')}
    summary.update(status='SOURCE_TRAINED',fit_ids=len(fit),counts=counts,losses=losses)
    (args.out/'source_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print('SIGNAL_GATE_SOURCE_TRAINED',args.commit,flush=True)


if __name__=='__main__':main()
