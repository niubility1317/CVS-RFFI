"""Source-only paired-IQ comparison of fixed cumulant and ranking models.

Physical-record retrieval uses privileged clean source IQ and is not TX
identity accuracy or a deployable reference library. No F or target is read.
"""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.train_source import load_source_training,extract_context
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.source_retrieval_geometry import retrieval
from experiments.phase1_cumulant_context.model import CumulantContextRecovery,CUMULANT_WEIGHT,LAGS,cumulant_features
from experiments.phase1_cumulant_context.train import SCHEMA as BASE_SCHEMA
from experiments.phase1_cumulant_rank.train import SCHEMA as RANK_SCHEMA
from experiments.phase1_cumulant_rank.objective import raw_features,normalize_features,RANK_WEIGHT,TEMPERATURE,NORMALIZATION_IDS

BASE='20261109-phase1-dura-cumulant-context-source-s2026092701-r01'
BASE_COMMIT='5ed56dd1decb913bb0733339462e08cc22a57831'
RANK='20261109-phase1-dura-cumulant-rank-source-s2026092701-r01'
RANK_COMMIT='1e83c8f23c837cf6ff68c218d2934cad4403411b'


def load_fixed(root,name,schema,commit,rank=False):
    folder=root/'runs'/name
    summary=core.read(folder/'source_summary.json')
    state=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
    expected=dict(schema=schema,commit=commit,epochs=20,fit_ids=32768,actual_source_view_count=98304,
        views_per_record=3,width=384,layers=8,lags=list(LAGS),cumulant_weight=CUMULANT_WEIGHT,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,DURA_frozen=True)
    if rank:expected.update(rank_weight=RANK_WEIGHT,rank_temperature=TEMPERATURE,normalization_ids=NORMALIZATION_IDS)
    if summary.get('status')!='SOURCE_TRAINED' or any(
            obj.get(k)!=v for obj in (summary,state) for k,v in expected.items()):
        raise ValueError('Fixed source-only final checkpoint and contract required')
    model=CumulantContextRecovery(width=384,layers=8).to('cuda:0')
    model.load_state_dict(state['model'],strict=True)
    return model.eval().requires_grad_(False)


def features(iq,stats):
    return normalize_features(raw_features(iq),stats)


@torch.no_grad()
def run(root,cache,out,commit):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if out.exists() or not torch.cuda.is_available():raise ValueError('Unique output and selected GPU required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    ids,rx,scene,_,received,clean=source_inputs(root)
    _,_,_,fit=load_source_training(root,ids,fit_size=32768)
    if len(ids)!=672 or np.intersect1d(ids,fit).size:raise ValueError('Pilot must be disjoint from source model fit')
    data,_=load_cache(cache,fit,32,'cuda:0',deterministic=True)
    normalization=raw_features(data[4][:512].to('cuda:0'))
    stats=[(x.mean(0),x.std(0,unbiased=False).clamp_min(.001)) for x in normalization]
    del data
    dura=frozen_dura(root,'cuda:0')
    models={'cumulant':load_fixed(root,BASE,BASE_SCHEMA,BASE_COMMIT),
            'rank':load_fixed(root,RANK,RANK_SCHEMA,RANK_COMMIT,rank=True)}
    x=core.tensor(received[ids[:2]],'cuda:0');context=extract_context(dura,x)
    for model in models.values():
        if not torch.isfinite(model(x,*context)).all():raise FloatingPointError('Finite source smoke required')
    print('CUMULANT_PAIR_SOURCE_SMOKE_PASS 2',flush=True)
    out.mkdir(parents=True);started=time.monotonic();chunks={k:[] for k in ('clean','dura','cumulant','rank')}
    for start in range(0,len(ids),8):
        selected=ids[start:start+8];x=core.tensor(received[selected],'cuda:0');target=core.tensor(clean[selected],'cuda:0')
        anchor,direction,variance=extract_context(dura,x)
        chunks['clean'].append(target.cpu());chunks['dura'].append(anchor.cpu())
        for name,model in models.items():chunks[name].append(model(x,anchor,direction,variance).cpu())
        if (start+len(selected))%168==0:print('CUMULANT_PAIR_SOURCE_PROGRESS',start+len(selected),flush=True)
    data={name:torch.cat(v,0).to('cuda:0') for name,v in chunks.items()}
    if not all(torch.isfinite(value).all() for value in data.values()):
        raise FloatingPointError('Finite source IQ required')
    target=data['clean'];energy=target.square().mean((1,2)).clamp_min(1e-8)
    errors={name:(value-target).square().mean((1,2))/energy for name,value in data.items() if name!='clean'}
    cum_target=cumulant_features(target)
    cum_errors={name:(cumulant_features(value)-cum_target).square().mean(1) for name,value in data.items() if name!='clean'}
    embeddings={name:features(value,stats) for name,value in data.items()}
    if not all(torch.isfinite(value).all() for value in (*errors.values(),*cum_errors.values(),*embeddings.values())):
        raise FloatingPointError('Finite source statistics required')
    groups={}
    for group,receivers in (('RX4_6',(4,6)),('RX8',(8,))):
        for view in core.VIEWS:
            mask=np.isin(rx,receivers)&(scene==view);positions=np.flatnonzero(mask)
            if len(positions)!=48:raise ValueError('Complete source pilot cells required')
            index=torch.tensor(positions.tolist(),dtype=torch.long,device='cuda:0')
            reference=embeddings['clean'][index];values={}
            for name in ('dura','cumulant','rank'):
                query=embeddings[name][index]
                ranks=retrieval(query,reference,torch.zeros(query.shape[1],device='cuda:0'),
                                torch.ones(query.shape[1],device='cuda:0'))
                values[name]=dict(iq_nmse=float(errors[name][index].mean()),
                    cumulant_error=float(cum_errors[name][index].mean()),record_top1=ranks['top1'],record_top5=ranks['top5'])
            groups[group+'/'+view]=dict(count=48,models=values)
    core.write(out/'result.json',dict(status='SOURCE_CUMULANT_PAIR_COMPLETE',commit=commit,
        base_commit=BASE_COMMIT,rank_commit=RANK_COMMIT,records=len(ids),groups=groups,
        normalization_source_ids=512,reference_per_cell=48,elapsed_seconds=time.monotonic()-started,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_IQ_reads=0,target_truth_reads=0,
        model_updates=0,DURA_frozen=True,caveat='Source waveform NMSE and same-record retrieval are not TX identity accuracy; clean reference is unavailable at deployment.'))
    print('CUMULANT_PAIR_SOURCE_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True);a=p.parse_args();run(a.root,a.cache,a.out,a.commit)

if __name__=='__main__':main()
