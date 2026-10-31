"""Source-only counterfactual: is a nearby teacher boundary reachable?"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import experiment as core
from .anchor_source_diagnostic import MODEL_DURA_SHA, PARENT

RUN = '20261101-diagnostic-phase1-boundary-oracle-s2026092701-r01'
SOURCE_PARENT = '20261101-diagnostic-phase1-anchor-source-paired-s2026092701-r02'
EPSILONS = (0.02, 0.05)
STEPS = 12


def projected_targeted_steps(classify, reference, targets, clean_norm, *, epsilon, steps):
    """Input-only target oracle, with each L2 radius relative to source clean norm."""
    if reference.ndim < 2 or targets.shape != (len(reference),) or clean_norm.shape != targets.shape:
        raise ValueError('Aligned reference, targets and source clean norms required')
    if not 0 < epsilon <= 1 or type(steps) is not int or steps < 1 or (clean_norm <= 0).any():
        raise ValueError('Positive finite per-sample radius and step count required')
    base = reference.detach()
    radius = epsilon * clean_norm.detach()
    shape = (len(base),) + (1,) * (base.ndim - 1)
    y = base.clone()
    for _ in range(steps):
        with torch.enable_grad():
            y = y.detach().requires_grad_(True)
            logits = classify(y)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Finite classifier outputs required')
            loss = F.cross_entropy(logits, targets, reduction='sum')
            gradient, = torch.autograd.grad(loss, y)
        norms = gradient.flatten(1).norm(dim=1).clamp_min(1e-12)
        y = y.detach() - (radius / 4).view(shape) * gradient.detach() / norms.view(shape)
        delta = y - base
        size = delta.flatten(1).norm(dim=1).clamp_min(1e-12)
        factor = (radius / size).clamp(max=1)
        y = base + delta * factor.view(shape)
    return y.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip() != args.commit:
        raise ValueError('Pinned diagnostic release required')
    torch.set_num_threads(2)
    teacher, contract = core.teacher(args.root, 'cpu')
    cache, clean, roles = core.data(args.root, contract)
    previous = core.read(args.root / 'runs' / SOURCE_PARENT / 'complete.json')
    if (previous.get('status'), previous.get('DURA_sha256'), previous.get('source_unique_records'),
        previous.get('source_views')) != ('ANALYZED_SOURCE_ONLY', MODEL_DURA_SHA, 512, [1,20]):
        raise ValueError('Verified source-native original DURA provenance required')
    source = args.root / 'runs' / PARENT
    if core.read(source / 'launch.json').get('commit') != '7e2d22ed1fe0feeac1df8381be89e4bb527108c5':
        raise ValueError('Pinned source producer required')
    ids = np.load(cache / 'train/physical_ids.npy', allow_pickle=False)
    fixed = np.concatenate([np.flatnonzero(roles == role)[np.linspace(0, count-1, n, dtype=int)]
                            for role, count, n in (('L_s',6300,128), ('U_s',56700,384))])
    def classify(x):
        return teacher.classify_features(teacher.features(x))
    outputs = []
    for view in (1,20):
        with np.load(source / f'DURA_default-{view}.npz', allow_pickle=False) as z:
            saved = {k: z[k] for k in ('source_positions','physical_ids','roles','clean','native','scene','state')}
        ix = saved['source_positions']
        if not (np.array_equal(ix,fixed) and np.array_equal(saved['physical_ids'],ids[ix])
                and np.array_equal(saved['roles'],roles[ix]) and np.array_equal(saved['clean'],clean[ix])):
            raise ValueError('Exact aligned Phase1 source samples required')
        if saved['native'].shape != (512,2,256) or not np.isfinite(saved['native']).all():
            raise ValueError('Original DURA native source IQ required')
        references, predictions = [], []
        with torch.no_grad():
            for offset in range(0,512,32):
                sl=slice(offset,offset+32)
                references.append(classify(core.tensor(saved['clean'][sl],'cpu')).argmax(1))
                predictions.append(classify(core.tensor(saved['native'][sl],'cpu')).argmax(1))
        target, baseline = torch.cat(references), torch.cat(predictions)
        wrong = (target != baseline).nonzero(as_tuple=True)[0]
        if not len(wrong):
            raise ValueError('No source teacher disagreement to study')
        chosen = wrong.tolist()
        x = core.tensor(saved['native'][chosen],'cpu')
        truth_iq = core.tensor(saved['clean'][chosen],'cpu')
        clean_norm = truth_iq.flatten(1).norm(dim=1).clamp_min(1e-8)
        energy = truth_iq.square().mean((1,2)).clamp_min(1e-8)
        before_nmse = (x-truth_iq).square().mean((1,2)) / energy
        experiments = []
        for epsilon in EPSILONS:
            y = projected_targeted_steps(classify,x,target[wrong],clean_norm,epsilon=epsilon,steps=STEPS)
            with torch.no_grad():
                success = (classify(y).argmax(1) == target[wrong])
                delta = (y-x).square().mean((1,2)) / energy
                after_nmse = (y-truth_iq).square().mean((1,2)) / energy
            if not torch.isfinite(y).all() or float(delta.max()) > epsilon**2 + 1e-6:
                raise ValueError('Projected source waveform exceeds fixed budget')
            groups=[]
            for key in ('scene','state','roles'):
                groups_array=saved[key][chosen]
                for value in np.unique(groups_array):
                    mask=torch.tensor((groups_array==value).tolist(),dtype=torch.bool)
                    groups.append(dict(group=f'{key}:{value}',count=int(mask.sum()),success=int(success[mask].sum())))
            experiments.append(dict(epsilon=epsilon,per_sample_delta_nmse_limit=epsilon**2,
                attempted=int(len(wrong)),target_reached=int(success.sum()),delta_nmse_mean=float(delta.mean()),
                delta_nmse_max=float(delta.max()),before_nmse_mean=float(before_nmse.mean()),
                after_nmse_mean=float(after_nmse.mean()),groups=groups))
        outputs.append(dict(source_view=view,source_count=512,teacher_disagreements=int(len(wrong)),experiments=experiments))
    if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise ValueError('Frozen Tri-SIG-TIME must not be updated')
    out=args.root/'runs'/RUN
    if out.exists(): raise FileExistsError('Unique diagnostic output required')
    out.mkdir()
    core.write(out/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,commit=args.commit,F_row=core.F_ROW,
        source_unique_records=512,source_observations=1024,source_views=[1,20],TX_labels_read=0,V_reads=0,
        test_IQ_reads=0,model_updates=0,gradient_scope='DURA output IQ only',steps=STEPS,epsilons=list(EPSILONS),results=outputs,
        caveat='Oracle uses paired source clean teacher class and clean norm; not deployable, not TX accuracy, not independent test evidence.'))
    print(json.dumps(dict(status='ANALYZED_SOURCE_ONLY',run=RUN)),flush=True)


if __name__ == '__main__':
    main()
