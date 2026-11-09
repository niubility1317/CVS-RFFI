"""Source-only physical-record geometry of bounded lagged cumulants.

Matching clean records are a privileged diagnostic reference, not a
transmitter-identity label or a deployable reference library.
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
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.source_retrieval_geometry import descriptors,retrieval
from experiments.phase1_dura_orthogonal.train_source import load_source_training,extract_context
from experiments.phase1_cumulant_context.model import cumulant_features,LAGS


def standardize(value, mean, scale):
    if value.ndim!=2 or mean.shape!=value.shape[1:] or scale.shape!=mean.shape:
        raise ValueError('Matching source feature statistics required')
    return (value-mean)/scale/(value.shape[1]**.5)


def features(iq):
    return descriptors(iq)[0],cumulant_features(iq)


def run(root,cache_path,out,commit):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if out.exists() or not torch.cuda.is_available():raise ValueError('Unique output and selected GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    pilot,_,_,_,_,_=source_inputs(root)
    _,_,_,fit=load_source_training(root,pilot,fit_size=32768)
    data,counts=load_cache(cache_path,fit,32,'cuda:0',deterministic=True)
    dura=frozen_dura(root,'cuda:0')
    with torch.no_grad():
        actual=extract_context(dura,data[0][:8].to('cuda:0'))
        for value,cached in zip(actual,data[1:4]):
            torch.testing.assert_close(value,cached[:8].to('cuda:0'),rtol=1e-4,atol=1e-5)
        for offset in (32768,65536):
            if not torch.equal(data[4][:1024],data[4][offset:offset+1024]):
                raise ValueError('Same clean physical records required across source views')
        print('CUMULANT_GEOMETRY_SOURCE_SMOKE_PASS 8',flush=True)
        start=time.monotonic()
        train=features(data[4][:512].to('cuda:0'))
        reference=features(data[4][512:1024].to('cuda:0'))
        stats=[(x.mean(0),x.std(0,unbiased=False).clamp_min(.001)) for x in train]
        refs=[standardize(x,*s) for x,s in zip(reference,stats)]
        result={}
        for index,offset in enumerate((0,32768,65536),start=1):
            result[str(index)]={}
            for name,field in (('received',0),('dura',1)):
                q=features(data[field][offset+512:offset+1024].to('cuda:0'))
                q=[standardize(x,*s) for x,s in zip(q,stats)]
                queries=[*q,torch.cat(q,1)];refs_all=[*refs,torch.cat(refs,1)]
                result[str(index)][name]={key:retrieval(a,b,torch.zeros(a.shape[1],device=a.device),
                    torch.ones(a.shape[1],device=a.device)) for key,a,b in zip(('power','cumulant','power_plus_cumulant'),queries,refs_all)}
    out.mkdir(parents=True)
    core.write(out/'result.json',dict(status='SOURCE_CUMULANT_GEOMETRY_COMPLETE',commit=commit,
        normalization_source_ids=512,retrieval_source_ids=512,source_views=3,records_per_view=512,
        chance_top1=1/512,chance_top5=5/512,lags=list(LAGS),source_counts=counts,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,
        DURA_frozen=True,elapsed_seconds=time.monotonic()-start,results=result,
        caveat='Source physical-record retrieval can reflect waveform content, not transmitter identity; clean reference is privileged and unavailable at deployment.'))
    print('CUMULANT_GEOMETRY_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True);a=p.parse_args();run(a.root,a.cache,a.out,a.commit)

if __name__=='__main__':main()
