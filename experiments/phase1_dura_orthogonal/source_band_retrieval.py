"""No-label source record retrieval using ordinary and band-balanced IQ."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
from torch.nn import functional as F

from experiments.phase1_dura_orthogonal.source_frequency_regression import load_context, FIT_SIZE
from experiments.phase1_dura_orthogonal.balanced_spectral_loss import BANDS, ENERGY_FLOOR_FRACTION


def features(iq, balanced):
    spectrum=torch.fft.fft(torch.complex(iq[:,0],iq[:,1]),dim=-1,norm='ortho')
    if balanced:
        freq=torch.fft.fftfreq(256,device=iq.device).abs()
        total=spectrum.abs().square().sum(-1).clamp_min(1e-12)
        adjusted=torch.empty_like(spectrum)
        for lo,hi in BANDS:
            mask=(freq>=lo)&(freq<hi)
            energy=spectrum[:,mask].abs().square().sum(-1)
            adjusted[:,mask]=spectrum[:,mask]/torch.maximum(energy,ENERGY_FLOOR_FRACTION*total).sqrt()[:,None]
        spectrum=adjusted
    vectors=torch.cat((spectrum.real,spectrum.imag),dim=1)
    return F.normalize(vectors,dim=1,eps=1e-12)


def ranks(query,reference,batch=256):
    if query.shape!=reference.shape or query.ndim!=2:
        raise ValueError('One aligned reference per query required')
    rank,ties=[],[]
    for offset in range(0,len(query),batch):
        score=query[offset:offset+batch]@reference.T
        idx=torch.arange(offset,offset+len(score),device=query.device)
        positive=score[torch.arange(len(score),device=query.device),idx]
        rank.append(1+(score>positive[:,None]).sum(1))
        ties.append((score==positive[:,None]).sum(1)-1)
    return torch.cat(rank),torch.cat(ties)


def diagnostic(data):
    received,anchor,variance,clean=data
    if (not torch.equal(clean[:FIT_SIZE],clean[FIT_SIZE:2*FIT_SIZE]) or
        not torch.equal(clean[:FIT_SIZE],clean[2*FIT_SIZE:])):
        raise ValueError('Same-record clean IQ must match across the three source views')
    ids=torch.arange(FIT_SIZE,device=received.device)
    fit=ids.remainder(8)!=0
    held=~fit
    ratio=(variance/received.square().mean((1,2)).clamp_min(1e-12)).reshape(3,FIT_SIZE)
    cuts=torch.quantile(ratio[:,fit].flatten(),torch.tensor([.25,.5,.75],device=received.device))
    reference=clean[:FIT_SIZE][held]
    out={}
    for balanced in (False,True):
        key=features(reference,balanced)
        for view in range(3):
            indices=ids[held]+view*FIT_SIZE
            group=torch.bucketize(ratio[view,held],cuts)
            for label,inputs in (('received',received),('dura',anchor)):
                rank,ties=ranks(features(inputs[indices],balanced),key)
                for bin_id in range(5):
                    selected=torch.ones_like(rank,dtype=torch.bool) if bin_id==0 else group==bin_id-1
                    name=f"{'balanced' if balanced else 'ordinary'}/view{21+view}/{label}/"+('all' if bin_id==0 else f'noise_q{bin_id}')
                    out[name]=dict(count=int(selected.sum()),top1=int((rank[selected]==1).sum()),
                        top5=int((rank[selected]<=5).sum()),ties=int((ties[selected]>0).sum()))
    return dict(bank_records=int(held.sum()),train_only_noise_cutoffs=cuts.cpu().tolist(),groups=out)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    data=load_context(args.cache)
    print('BAND_RETRIEVAL_SOURCE_CONTEXT_PASS',flush=True)
    start=time.monotonic()
    result=diagnostic(data)
    torch.cuda.synchronize()
    result.update(status='COMPLETE',commit=args.commit,elapsed_seconds=time.monotonic()-start,
        heldout='source ID index modulo 8 equals 0; 4096 aligned clean IQ references',
        feature_modes=['ordinary complex IQ cosine','per-record four-band normalized complex IQ cosine'],
        tie_rule='optimistic rank: 1 + count(scores strictly greater than positive); ties counted separately',
        device=torch.cuda.get_device_name(),TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,
        caveat='Same-record source retrieval is not transmitter identity; clean reference bank is unavailable at deployment.')
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('BAND_RETRIEVAL_SOURCE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
