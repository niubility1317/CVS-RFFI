"""Frequency decomposition of same-record DURA source-view errors."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

from experiments.phase1_dura_orthogonal.source_frequency_regression import FIT_SIZE, load_context


def band_metrics(anchors, clean):
    if anchors.ndim != 4 or anchors.shape[1:] != (3,2,256) or clean.shape != anchors[:,0].shape:
        raise ValueError('Three aligned source IQ views required')
    z=torch.complex(anchors[:,:,0], anchors[:,:,1])
    y=torch.complex(clean[:,0], clean[:,1])
    spectrum=torch.fft.fft(z-y[:,None], dim=-1, norm='ortho')
    clean_spectrum=torch.fft.fft(y, dim=-1, norm='ortho')
    energy=clean_spectrum.abs().square().sum(-1).clamp_min(1e-12)
    band=(torch.fft.fftfreq(256, device=clean.device).abs()*8).to(torch.int64).clamp_max(3)
    cross=(spectrum[:,0].conj()*spectrum[:,1] + spectrum[:,0].conj()*spectrum[:,2]
           + spectrum[:,1].conj()*spectrum[:,2]).real/3
    result=[]
    for i in range(4):
        mask=band == i
        share=clean_spectrum[:,mask].abs().square().sum(-1)/energy
        single=spectrum[:,:,mask].abs().square().sum(-1).mean(1)/energy
        average=spectrum[:,:,mask].mean(1).abs().square().sum(-1)/energy
        pair=cross[:,mask].sum(-1)/energy
        result.append(torch.stack((share,single,average,pair),1))
    values=torch.stack(result,1)
    torch.testing.assert_close(values[:,:,2],(values[:,:,1]+2*values[:,:,3])/3,rtol=1e-4,atol=1e-6)
    return values


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique source output and GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    x,a,v,y=load_context(args.cache)
    if not torch.equal(y[:FIT_SIZE],y[FIT_SIZE:2*FIT_SIZE]) or not torch.equal(y[:FIT_SIZE],y[2*FIT_SIZE:]):
        raise ValueError('Three clean source views must match by physical ID')
    anchors=torch.stack(tuple(a[j*FIT_SIZE:(j+1)*FIT_SIZE] for j in range(3)),1)
    smoke=band_metrics(anchors[:8],y[:8])
    if not torch.isfinite(smoke).all():
        raise FloatingPointError('Real source context smoke required')
    print('MULTIVIEW_BANDS_SOURCE_SMOKE_PASS',flush=True)
    started=time.monotonic()
    with torch.no_grad():
        values=torch.cat([band_metrics(anchors[i:i+512],y[i:i+512]) for i in range(0,FIT_SIZE,512)])
        ratios=v/x.square().mean((1,2)).clamp_min(1e-12)
        grouped=torch.stack(tuple(ratios[j*FIT_SIZE:(j+1)*FIT_SIZE] for j in range(3)),1).mean(1)
        cutoffs=torch.quantile(grouped,torch.tensor([.25,.5,.75],device=grouped.device))
        bins=torch.bucketize(grouped,cutoffs)
        groups={}
        for i in range(5):
            mask=torch.ones(FIT_SIZE,dtype=torch.bool,device=x.device) if i==0 else bins==i-1
            item=values[mask]
            if not len(item) or not torch.isfinite(item).all():
                raise FloatingPointError('Finite nonempty source group required')
            groups['all' if i==0 else f'observed_noise_q{i}']=dict(records=len(item),mean_metrics=item.mean(0).cpu().tolist())
    torch.cuda.synchronize()
    args.out.mkdir(parents=True)
    result=dict(status='MULTIVIEW_BANDS_SOURCE_COMPLETE',commit=args.commit,records=FIT_SIZE,views=3,
        band_definition='Absolute FFT frequency, four bins: [0,.125), [.125,.25), [.25,.375), [.375,.5] cycles/sample',
        metric_order=['clean_energy_share','single_error_contribution','three_mean_error_contribution','pair_error_cross_contribution'],
        group_metric='record mean estimated variance / received power',cutoffs=cutoffs.cpu().tolist(),groups=groups,
        elapsed_seconds=time.monotonic()-started,device=torch.cuda.get_device_name(),
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,
        caveat='Errors normalized by whole-record clean energy, bands are contributions not separately normalized NMSE. Source uses three simulated views; deployment receives one. Not identity accuracy or a universal bound.')
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('MULTIVIEW_BANDS_SOURCE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
