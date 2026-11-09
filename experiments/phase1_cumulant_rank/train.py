"""F-free source training with same-record geometry; no TX labels."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_joint_particle.train import _data,_loss
from experiments.phase1_cumulant_context.model import CumulantContextRecovery,CUMULANT_WEIGHT,LAGS
from experiments.phase1_cumulant_context.train import EPOCHS,BATCH,WIDTH,LAYERS,MODEL_SEED,ORDER_SEED
from experiments.phase1_cumulant_rank.objective import raw_features,normalize_features,rank_loss,TEMPERATURE,RANK_WEIGHT,NORMALIZATION_IDS

SCHEMA='phase1_cumulant_record_rank_source_v1'


@torch.no_grad()
def source_reference(values,smoke):
    physical=len(values[4]) if smoke else len(values[4])//3
    if not smoke and (physical!=32768 or not torch.equal(values[4][:physical],values[4][physical:2*physical])
                      or not torch.equal(values[4][:physical],values[4][2*physical:])):
        raise ValueError('Three fixed source views must share clean physical records')
    pieces=[[],[]]
    for start in range(0,physical,256):
        blocks=raw_features(values[4][start:start+256])
        for dst,x in zip(pieces,blocks):dst.append(x)
    raw=[torch.cat(x,0) for x in pieces]
    n=min(NORMALIZATION_IDS,physical) if smoke else NORMALIZATION_IDS
    stats=[(x[:n].mean(0),x[:n].std(0,unbiased=False).clamp_min(.001)) for x in raw]
    normalized=normalize_features(raw,stats)
    if not all(torch.isfinite(x).all() for x in (*raw,normalized)):
        raise FloatingPointError('Finite source feature statistics required')
    print('CUMULANT_RANK_SOURCE_REFERENCE',physical,'normalization_ids',n,flush=True)
    return physical,stats,normalized,raw[1]


def run(root,out,commit,cache,smoke=False):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    training_environments('cuda:0',os.environ)
    if out.exists() or not torch.cuda.is_available():raise ValueError('Unique source output and selected GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(MODEL_SEED)
    values,fit,counts=_data(root,'cuda:0',smoke,cache)
    physical,stats,references,clean_cumulants=source_reference(values,smoke)
    model=CumulantContextRecovery(width=WIDTH,layers=LAYERS).to('cuda:0').train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    rng=np.random.default_rng(ORDER_SEED)
    out.mkdir(parents=True);started=time.monotonic();losses=[]
    for epoch in range(1,2 if smoke else EPOCHS+1):
        order=rng.permutation(len(values[0]))
        if smoke:order=np.tile(order,3)
        total=torch.zeros((),device='cuda:0');rank_total=torch.zeros((),device='cuda:0')
        for start in range(0,len(order),BATCH):
            idx=torch.tensor(order[start:start+BATCH].tolist(),dtype=torch.long,device='cuda:0')
            received,anchor,direction,variance,clean=(v[idx] for v in values)
            record_ids=idx.remainder(physical)
            output=model(received,anchor,direction,variance)
            blocks=raw_features(output)
            characteristic=(blocks[1]-clean_cumulants[record_ids]).square().mean()
            query=normalize_features(blocks,stats)
            rank=rank_loss(query,references[record_ids],record_ids)
            loss=_loss(output,anchor,clean)+CUMULANT_WEIGHT*characteristic+RANK_WEIGHT*rank
            if not torch.isfinite(loss):raise FloatingPointError('Finite source loss required')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if start==0 and (model.out.weight.grad is None or model.out.weight.grad.abs().sum()<=0):
                raise ValueError('Finite nonzero source gradient required')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();total+=loss.detach()*len(idx);rank_total+=rank.detach()*len(idx)
        torch.cuda.synchronize();losses.append(float(total.cpu())/len(order))
        print('CUMULANT_RANK_SOURCE_EPOCH',epoch,'loss',losses[-1],'rank',float(rank_total.cpu())/len(order),
              'elapsed',round(time.monotonic()-started,1),flush=True)
    common=dict(schema=SCHEMA,commit=commit,epochs=1 if smoke else EPOCHS,fit_ids=len(fit),
        actual_source_view_count=len(values[0]),views_per_record=3,width=WIDTH,layers=LAYERS,
        model_seed=MODEL_SEED,order_seed=ORDER_SEED,lags=list(LAGS),cumulant_weight=CUMULANT_WEIGHT,
        rank_weight=RANK_WEIGHT,rank_temperature=TEMPERATURE,normalization_ids=min(NORMALIZATION_IDS,physical),
        rank_positive='same source physical record, including repeated views',
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        source_loss='normalized IQ + .02 DURA change + .1 cumulant + .002 source-record ranking',
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,DURA_frozen=True,
        source_counts=counts,source_counts_scope='complete_cached_three_views',
        elapsed_seconds=time.monotonic()-started,losses=losses)
    with (out/('smoke.pt' if smoke else 'final.pt')).open('xb') as f:
        torch.save(dict(**common,model=model.eval().cpu().state_dict()),f)
    core.write(out/('smoke.json' if smoke else 'source_summary.json'),dict(
        status='SOURCE_TRAINING_SMOKE_PASS' if smoke else 'SOURCE_TRAINED',**common))
    print('CUMULANT_RANK_SOURCE_DONE',commit,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--device',choices=('cuda:0',),default='cuda:0')
    p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.root,a.out,a.commit,a.cache,a.smoke)

if __name__=='__main__':main()
