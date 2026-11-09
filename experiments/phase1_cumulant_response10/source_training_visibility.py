"""Measure synthetic response-to-residual energy on actual source fit views."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.train_source import load_source_training
from experiments.phase1_dura_orthogonal.counterfactual_response import choose_perturbation, DIRECTIONS
from experiments.phase1_dura_orthogonal.oracle_action_source_probe import true_action
from experiments.pamf_action_restore.operators import Action

SOURCE_EPOCHS = (21, 22, 23)
BATCH = 128


def relative_db(effect, residual):
    if effect.ndim != 2 or effect.shape[1:] != (256,) or effect.shape != residual.shape:
        raise ValueError('Matching complex source windows required')
    if not torch.is_complex(effect) or not torch.is_complex(residual):
        raise ValueError('Complex source windows required')
    if not torch.isfinite(effect).all() or not torch.isfinite(residual).all():
        raise FloatingPointError('Finite source effect and residual required')
    a, b = effect.abs().square().sum(-1), residual.abs().square().sum(-1)
    if (a <= 1e-20).any() or (b <= 1e-20).any():
        raise ValueError('Nonzero practical source effect and residual required')
    return 10*torch.log10(a/b)


def cell_summary(values):
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty practical cell required')
    q=np.quantile(values, [.1,.5,.9])
    return dict(count=len(values), p10_db=float(q[0]), median_db=float(q[1]), p90_db=float(q[2]),
                below_minus20_at_1pct=int((values < -20).sum()),
                below_minus20_at_10pct=int((values < -40).sum()))


@torch.no_grad()
def run(root, out, commit):
    if out.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Unique output and CPU-only execution required')
    torch.set_num_threads(2)
    pilot, _, _, _, _, _ = source_inputs(root)
    _, fresh, clean, fit = load_source_training(root, pilot, fit_size=32768)
    if len(fit) != 32768 or np.intersect1d(fit,pilot).size:
        raise ValueError('Fixed legal source training fit required')
    dura = frozen_dura(root,'cpu')
    if any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Frozen DURA source smoke required')
    chunks, scenes, dirs = [], [], []
    clean_count = 0
    started = time.monotonic()
    for view, epoch in enumerate(SOURCE_EPOCHS):
        folder = fresh/'epochs'/f'{epoch:02d}'
        meta = core.read(folder/'complete.json')
        if (meta.get('count') != core.COUNT or meta.get('namespace') != f'learned-action/train/epoch{epoch:02d}'
                or any(meta.get(k) != 0 for k in ('TX_labels_read','F_outputs_read','target_reads'))):
            raise ValueError('Matched legal source simulator cache required')
        received=np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)
        kernel=np.load(folder/'kernel.npy',mmap_mode='r',allow_pickle=False)
        coefficients=np.load(folder/'coefficients.npy',mmap_mode='r',allow_pickle=False)
        with np.load(folder/'groups.npz',allow_pickle=False) as z: scene=z['scene'].copy()
        if (received.shape != clean.shape or received.dtype != np.float32 or kernel.shape != (core.COUNT,2,64)
                or coefficients.shape != (core.COUNT,4,256) or kernel.dtype != np.float32
                or coefficients.dtype != np.float32 or scene.shape != (core.COUNT,)):
            raise ValueError('Matching source simulator arrays required')
        for start in range(0,len(fit),BATCH):
            ids=fit[start:start+BATCH]
            x,y=core.tensor(received[ids],'cpu'),core.tensor(clean[ids],'cpu')
            if view == 0 and start == 0:
                smoke=dura(x[:2])
                if smoke.shape != (2,2,256) or not torch.isfinite(smoke).all():
                    raise FloatingPointError('Finite real-checkpoint source smoke required')
                out.mkdir(parents=True)
                print('SOURCE_TRAIN_VISIBILITY_SMOKE_PASS 2',flush=True)
            names=scene[ids]
            if not np.isin(names,core.VIEWS).all():
                raise ValueError('Known source scenes required')
            active=names != 'clean'
            clean_count += int((~active).sum())
            if not active.any(): continue
            mask=torch.tensor(active.tolist(),dtype=torch.bool)
            delta=choose_perturbation(y,view*len(fit)+start)[mask]
            action=true_action(Action,core.tensor(kernel[ids],'cpu')[mask],core.tensor(coefficients[ids],'cpu')[mask])
            original=torch.complex(y[mask,0],y[mask,1])
            observed=torch.complex(x[mask,0],x[mask,1])
            value=relative_db(action(delta),observed-action(original))
            chunks.append(np.from_dlpack(value.cpu().contiguous()).copy());scenes.append(names[active].copy())
            dirs.append(((np.arange(len(ids))+view*len(fit)+start)%len(DIRECTIONS))[active])
        print('SOURCE_TRAIN_VISIBILITY_PROGRESS',epoch,len(fit),flush=True)
    values,scene,direction=np.concatenate(chunks),np.concatenate(scenes),np.concatenate(dirs)
    if len(values)+clean_count != len(fit)*len(SOURCE_EPOCHS):
        raise ValueError('Complete source training view coverage required')
    groups={}
    for view in core.VIEWS[1:]:
        mask=scene==view
        groups[view]=dict(all_directions=cell_summary(values[mask]), directions={
            name:cell_summary(values[mask & (direction==i)]) for i,name in enumerate(DIRECTIONS)})
    with (out/'source_visibility.npz').open('xb') as f:
        np.savez(f,relative_db_1pct=values,scene=scene,direction=direction)
    core.write(out/'result.json',dict(status='SOURCE_TRAIN_RESPONSE_VISIBILITY_COMPLETE',commit=commit,
        fit_records=len(fit),source_epochs=list(SOURCE_EPOCHS),total_views=len(fit)*len(SOURCE_EPOCHS),
        practical_views=len(values),clean_excluded=clean_count,directions=list(DIRECTIONS),groups=groups,
        epsilon_relative_rms=.01,scale10_db_offset=20.,elapsed_seconds=time.monotonic()-started,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,model_updates=0,
        caveat='Residual includes noise and simulator mismatch; synthetic directions are not validated TX identity; 10pct follows fixed-action scaling, not a new simulation.'))
    print('SOURCE_TRAIN_VISIBILITY_COMPLETE',len(values),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();run(a.root,a.out,a.commit)

if __name__ == '__main__': main()
