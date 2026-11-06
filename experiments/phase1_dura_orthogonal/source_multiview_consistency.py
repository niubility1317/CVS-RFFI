"""Inspect same-record three-view source errors and a nondeployable ensemble."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

from experiments.phase1_dura_orthogonal.source_frequency_regression import FIT_SIZE, load_context


def metrics(anchors, clean):
    if anchors.ndim != 4 or anchors.shape[1:] != (3, 2, 256) or clean.shape != anchors[:, 0].shape:
        raise ValueError('Three matched IQ views of each source record required')
    error = anchors-clean[:, None]
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    singles = error.square().mean((2, 3))/energy[:, None]
    ensemble = (anchors.mean(1)-clean).square().mean((1, 2))/energy
    signed_cross = torch.stack([(error[:, a]*error[:, b]).mean((1, 2))/energy
                                for a,b in ((0, 1), (0, 2), (1, 2))], 1).mean(1)
    # E[||mean(e)||²] equals (mean single error + 2 mean cross-product)/3.
    torch.testing.assert_close(ensemble, (singles.mean(1)+2*signed_cross)/3, rtol=1e-4, atol=1e-6)
    return torch.cat((singles, ensemble[:, None], signed_cross[:, None]), 1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args=p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique source output and GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    x,a,v,y=load_context(args.cache)
    if not torch.equal(y[:FIT_SIZE],y[FIT_SIZE:2*FIT_SIZE]) or not torch.equal(y[:FIT_SIZE],y[2*FIT_SIZE:]):
        raise ValueError('The three source views must have exactly matching clean IQ by physical ID')
    started=time.monotonic()
    with torch.no_grad():
        anchors=torch.stack(tuple(a[j*FIT_SIZE:(j+1)*FIT_SIZE] for j in range(3)),1)
        result=metrics(anchors,y[:FIT_SIZE])
        ratios=v/x.square().mean((1,2)).clamp_min(1e-12)
        grouped=torch.stack(tuple(ratios[j*FIT_SIZE:(j+1)*FIT_SIZE] for j in range(3)),1).mean(1)
        cutoffs=torch.quantile(grouped,torch.tensor([.25,.5,.75],device=grouped.device))
        bins=torch.bucketize(grouped,cutoffs)
        groups={}
        for group in range(5):
            mask=torch.ones(FIT_SIZE,dtype=torch.bool,device=x.device) if group==0 else bins==group-1
            item=result[mask]
            if not len(item) or not torch.isfinite(item).all():
                raise FloatingPointError('Finite nonempty source group required')
            groups['all' if group==0 else f'observed_noise_q{group}']=dict(
                records=len(item), mean_metrics=item.mean(0).cpu().tolist(),
                ensemble_better_than_each=(item[:,3,None] < item[:,:3]).sum(0).cpu().tolist(),
                ensemble_better_than_all=int((item[:,3,None] < item[:,:3]).all(1).sum()))
    torch.cuda.synchronize()
    args.out.mkdir(parents=True)
    value=dict(status='SOURCE_MULTIVIEW_CONSISTENCY_COMPLETE',commit=args.commit,records=FIT_SIZE,views=3,
        metric_order=['view21_nmse','view22_nmse','view23_nmse','mean_three_nmse','mean_pair_cross_error'],
        noise_group_metric='source-record mean of estimated variance / received power',
        noise_cutoffs=cutoffs.cpu().tolist(), groups=groups, elapsed_seconds=time.monotonic()-started,
        device=torch.cuda.get_device_name(), TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,
        model_updates=0,caveat='Each ensemble uses three simulated source views of the same clean record; Phase1 deployment has one view. IQ NMSE is not identity accuracy or a universal bound.')
    (args.out/'result.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print('SOURCE_MULTIVIEW_CONSISTENCY_COMPLETE',flush=True)


if __name__=='__main__':
    main()
