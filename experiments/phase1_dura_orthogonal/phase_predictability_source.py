"""Fit source-only phase proxies with record-disjoint held-out evaluation."""
import argparse
import os
from pathlib import Path
import sys

from experiments.phase1_dura_orthogonal.benchmark_flow_n607_sync import wait_for_idle

SOURCE = '20261107-phase1-dura-conditional-phase-source-s2026092701-r01'
SOURCE_COMMIT = 'afac42e77542d0f035996d1a2e8dde7decbd4ac5'
RIDGE = .01


def oracle_angle(anchor, clean):
    import torch
    ar, ai, cr, ci = anchor[:, 0], anchor[:, 1], clean[:, 0], clean[:, 1]
    return torch.atan2((ci*ar-cr*ai).sum(-1), (cr*ar+ci*ai).sum(-1)).clamp(-torch.pi/2, torch.pi/2)


def phase_features(received, anchor, innovation, variance):
    import torch
    from experiments.phase1_dura_orthogonal.conditional_phase_moments import complex_moments
    power = anchor.square().mean((1, 2)).clamp_min(1e-12)
    scale = power.sqrt()[:, None]
    signals = (received, anchor, innovation)
    means = torch.cat([x.mean(-1)/scale for x in signals] + [torch.log1p(variance/power)[:, None]], 1)
    moments = torch.cat((means, *(complex_moments(x) for x in signals)), 1)
    relative = []
    for left, right in ((received, anchor), (received, innovation), (anchor, innovation)):
        lr, li, rr, ri = left[:, 0], left[:, 1], right[:, 0], right[:, 1]
        denom = ((lr.square()+li.square()).mean(-1)*(rr.square()+ri.square()).mean(-1)).clamp_min(1e-24).sqrt()
        relative.extend((((lr*rr+li*ri).mean(-1)/denom)[:, None],
                         ((li*rr-lr*ri).mean(-1)/denom)[:, None]))
    return means, moments, torch.cat((moments, *relative), 1)


def fit_ridge(train_x, train_y, test_x, ridge=RIDGE):
    import torch
    if train_x.ndim != 2 or train_y.shape != (len(train_x),) or test_x.shape[1] != train_x.shape[1]:
        raise ValueError('Matching fit and held-out features required')
    mean = train_x.mean(0)
    std = train_x.std(0, unbiased=False).clamp_min(1e-5)
    train = torch.cat(((train_x-mean)/std, torch.ones((len(train_x), 1), device=train_x.device)), 1)
    test = torch.cat(((test_x-mean)/std, torch.ones((len(test_x), 1), device=test_x.device)), 1)
    penalty = torch.eye(train.shape[1], device=train.device, dtype=train.dtype)*ridge
    penalty[-1, -1] = 0
    beta = torch.linalg.solve(train.T@train/len(train)+penalty, train.T@train_y/len(train))
    return (test@beta).clamp(-torch.pi/2, torch.pi/2)


def angle_nmse(anchor, clean, angle):
    import torch
    cos, sin = angle.cos()[:, None], angle.sin()[:, None]
    ar, ai = anchor[:, 0], anchor[:, 1]
    rotated = torch.stack((cos*ar-sin*ai, sin*ar+cos*ai), 1)
    return (rotated-clean).square().mean((1, 2))/clean.square().mean((1, 2)).clamp_min(1e-8)


def summarize(anchor, clean, target, predicted):
    import torch
    baseline = angle_nmse(anchor, clean, torch.zeros_like(target))
    observed = angle_nmse(anchor, clean, predicted)
    ideal = angle_nmse(anchor, clean, target)
    return dict(count=len(target), target_mse=float(target.square().mean()),
        prediction_mse=float((target-predicted).square().mean()),
        anchor_nmse=float(baseline.mean()), predicted_nmse=float(observed.mean()), oracle_nmse=float(ideal.mean()),
        predicted_beats_anchor=int((observed<baseline).sum()),
        predicted_mean_abs_angle=float(predicted.abs().mean()), oracle_mean_abs_angle=float(target.abs().mean()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    uuid=wait_for_idle(args.out,args.gpu,allow_shared=True)
    if 'torch' in sys.modules:
        raise RuntimeError('Select GPU before importing torch')
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import numpy as np
    import torch
    from experiments.pamf_phase1_flow import experiment as core
    from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
    from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
    from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
    from experiments.phase1_dura_orthogonal.train_source import extract_context, SCHEMA

    if not torch.cuda.is_available():
        raise ValueError('Registered GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    folder=args.root/'runs'/SOURCE
    summary=core.read(folder/'source_summary.json')
    state=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    expected=dict(commit=SOURCE_COMMIT,architecture='conditional_phase_dura',objective='wide_conditional_phase',
                  epochs=5,DURA_frozen=True,DURA_training_reads=1,TX_labels_read=0,F_weights_read=0,
                  F_outputs_read=0,V_reads=0,target_reads=0)
    fit=np.asarray(state.get('fit_ids',[]))
    if (summary.get('status')!='SOURCE_TRAINED' or summary.get('fit_ids')!=32768 or state.get('schema')!=SCHEMA
            or fit.shape!=(32768,) or len(np.unique(fit))!=len(fit)
            or any(o.get(k)!=v for o in (summary,state) for k,v in expected.items())):
        raise ValueError('Matching source-only training contract required')
    pilot, _, _, _, _, _ = source_inputs(args.root)
    if np.intersect1d(fit,pilot).size:
        raise ValueError('Source fit must exclude source diagnostic pilot')
    cache=args.root/'runs/20261106-phase1-dura-wide-source-source-s2026092701-r01/source_context.pt'
    data,counts=load_cache(cache,fit,32,'cuda:0',deterministic=True)
    n=len(fit)
    if not torch.equal(data[4][:n],data[4][n:2*n]) or not torch.equal(data[4][:n],data[4][2*n:]):
        raise ValueError('All three source views must share the same ordered clean records')
    dura=frozen_dura(args.root,'cuda:0')
    with torch.no_grad():
        real=extract_context(dura,data[0][:32].cuda())
        if any(not torch.allclose(a,b[:32].cuda(),rtol=1e-5,atol=1e-6) for a,b in zip(real,data[1:4])):
            raise ValueError('Real frozen DURA context replay must agree')
        if any(p.requires_grad for p in dura.parameters()):
            raise ValueError('DURA must remain frozen')
    print('PHASE_PREDICTABILITY_SOURCE_SMOKE_PASS',32,flush=True)
    del dura
    received,anchor,innovation,variance,clean=(v.cuda() for v in data)
    target=oracle_angle(anchor,clean)
    features=phase_features(received,anchor,innovation,variance)
    if not all(torch.isfinite(x).all() for x in (*features,target)):
        raise ValueError('Finite source features and phase targets required')
    held_np=(fit%4==0)
    if not 0<held_np.sum()<n:
        raise ValueError('A record-disjoint modulo-four split is required')
    held=torch.as_tensor(np.tile(held_np,3),device=anchor.device)
    target_fit=target[~held]; target_held=target[held]
    results={}
    models={'zero':torch.zeros_like(target_held),
            'train_mean':target_fit.mean().expand_as(target_held).clamp(-torch.pi/2,torch.pi/2)}
    for name,x in zip(('means_noise','plus_complex_moments','plus_relative_correlations'),features):
        models[name]=fit_ridge(x[~held],target_fit,x[held])
    for name,predicted in models.items():
        results[name]=summarize(anchor[held],clean[held],target_held,predicted)
    if any(not np.isfinite(v) for row in results.values() for v in row.values()):
        raise ValueError('Finite held-out source metrics required')
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json',dict(status='PHASE_PREDICTABILITY_COMPLETE',commit=args.commit,
        records=n,views_per_record=3,fit_ids=int((~held_np).sum()),heldout_ids=int(held_np.sum()),
        split='physical_train_index_mod4_zero_heldout',ridge=RIDGE,counts=counts,results=results,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,
        source_only_regression_fit=True,physical_gpu=args.gpu,physical_gpu_uuid=uuid,
        caveat='Ridge probes fit source-only clean-derived angles; not a deployed recovery network or TX accuracy.'))
    print('PHASE_PREDICTABILITY_COMPLETE',n,'heldout_ids',int(held_np.sum()),flush=True)


if __name__=='__main__':
    main()
