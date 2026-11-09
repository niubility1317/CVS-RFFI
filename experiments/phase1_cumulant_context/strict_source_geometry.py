"""Compare strict lag-slice centering to fixed source moment-difference features.

This is source-only record matching with privileged clean reference, not TX
identity accuracy, not a new recovery model, and not an intervention on runs.
"""
import argparse
import os
from pathlib import Path
import time
import torch
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.source_retrieval_geometry import descriptors,retrieval
from experiments.phase1_dura_orthogonal.train_source import load_source_training,extract_context
from experiments.phase1_cumulant_context.model import cumulant_features,LAGS
from experiments.phase1_cumulant_context.source_geometry import standardize


def strict_features(iq):
    if iq.ndim!=3 or iq.shape[1:]!=(2,256):raise ValueError('IQ windows required')
    z=torch.complex(iq[:,0],iq[:,1]);z=z-z.mean(-1,keepdim=True)
    z=z/z.abs().square().mean(-1,keepdim=True).clamp_min(1e-10).sqrt()
    results=[]
    for lag in LAGS:
        a,b=(z,z) if lag==0 else (z[:,:-lag],z[:,lag:])
        a=a-a.mean(-1,keepdim=True);b=b-b.mean(-1,keepdim=True)
        pa,pb=a.abs().square(),b.abs().square()
        cross=(a*b.conj()).mean(-1);pseudo=(a*b).mean(-1)
        first=(pa*pb).mean(-1)-pa.mean(-1)*pb.mean(-1)-cross.abs().square()-pseudo.abs().square()
        second=(pa*a*b.conj()).mean(-1)-2*pa.mean(-1)*cross-a.square().mean(-1)*(a.conj()*b.conj()).mean(-1)
        results.extend((first,second.real,second.imag))
    value=torch.stack(results,1)
    return value/(1+value.abs())


def features(iq):
    return descriptors(iq)[0],cumulant_features(iq),strict_features(iq)


@torch.no_grad()
def run(root,cache,out,commit):
    if out.exists():raise FileExistsError('Unique source output required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    pilot,_,_,_,_,_=source_inputs(root)
    _,_,_,fit=load_source_training(root,pilot,fit_size=32768)
    data,counts=load_cache(cache,fit,32,'cuda:0',deterministic=True)
    dura=frozen_dura(root,'cpu')
    context=extract_context(dura,data[0][:8])
    if not all(torch.isfinite(v).all() for v in context) or any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Finite frozen DURA source smoke required')
    for offset in (32768,65536):
        if not torch.equal(data[4][:1024],data[4][offset:offset+1024]):raise ValueError('Aligned source records required')
    print('STRICT_GEOMETRY_SOURCE_SMOKE_PASS 8',flush=True)
    started=time.monotonic();train=features(data[4][:512]);reference=features(data[4][512:1024])
    stats=[(x.mean(0),x.std(0,unbiased=False).clamp_min(.001)) for x in train]
    refs=[standardize(x,*s) for x,s in zip(reference,stats)]
    names=('power','window_centered','slice_centered','power_plus_window','power_plus_slice')
    result={}
    for index,offset in enumerate((0,32768,65536),start=1):
        result[str(index)]={}
        for name,field in (('received',0),('dura',1)):
            raw=features(data[field][offset+512:offset+1024]);q=[standardize(x,*s) for x,s in zip(raw,stats)]
            queries=[*q,torch.cat((q[0],q[1]),1),torch.cat((q[0],q[2]),1)]
            ref=[*refs,torch.cat((refs[0],refs[1]),1),torch.cat((refs[0],refs[2]),1)]
            result[str(index)][name]={k:retrieval(a,b,torch.zeros(a.shape[1]),torch.ones(a.shape[1]))
                                      for k,a,b in zip(names,queries,ref)}
    out.mkdir(parents=True)
    core.write(out/'result.json',dict(status='SOURCE_STRICT_GEOMETRY_COMPLETE',commit=commit,
        normalization_source_ids=512,retrieval_source_ids=512,source_views=3,records_per_view=512,
        chance_top1=1/512,chance_top5=5/512,lags=list(LAGS),source_counts=counts,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,
        DURA_frozen=True,elapsed_seconds=time.monotonic()-started,results=result,
        caveat='Source physical-record retrieval can reflect waveform content, not TX identity; clean reference is unavailable at deployment.'))
    print('STRICT_GEOMETRY_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True);a=p.parse_args();run(a.root,a.cache,a.out,a.commit)

if __name__=='__main__':main()
