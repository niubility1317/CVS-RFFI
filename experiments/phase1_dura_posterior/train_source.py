"""Train DURA's posterior-conditioned stage corrections using source IQ only."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_posterior.integrated import PosteriorInDura
from experiments.phase1_dura_posterior.numerics import checked_logits
from experiments.phase1_dura_posterior.pipeline import GatedPosteriorDura
from experiments.phase1_dura_posterior.source_objective import source_objective
from experiments.phase1_spectral_synthesis.fit_sampling import select_fit


VIEWS = tuple(core.VIEWS)
EPOCHS = 5
BATCH = 8
FIT_SIZE = 8192


def snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def unchanged(module, before):
    return (not any(p.requires_grad or p.grad is not None for p in module.parameters())
            and all(torch.equal(value.detach().cpu(), before[key]) for key, value in module.state_dict().items()))


@torch.no_grad()
def prepare_source(root, source_cache, fit, clean, rx, gate, physical, dura, reader, device, source):
    """Precompute only gate-active source inputs; use no device identity labels."""
    fresh = root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
    if (fresh/'train/physical_ids.npy').resolve() != (source_cache/'train/physical_ids.npy').resolve():
        raise ValueError('The source views must share the same physical record IDs')
    chunks = []
    counts = {}
    for epoch in (21, 22, 23):
        metadata = core.read(fresh/'epochs'/f'{epoch:02d}'/'complete.json')
        if (metadata.get('count') != 63000 or metadata.get('namespace') != f'learned-action/train/epoch{epoch:02d}'
                or any(metadata.get(key) != 0 for key in ('TX_labels_read', 'F_outputs_read', 'target_reads'))):
            raise ValueError('Approved source-only simulated view required')
        received, scene = source.epoch_data(fresh, epoch)
        for start in range(0, len(fit), BATCH):
            ids = fit[start:start+BATCH]
            x = core.tensor(received[ids], device)
            logits = gate(x).reshape(-1)
            if logits.shape != (len(x),) or not torch.isfinite(x).all() or not torch.isfinite(logits).all():
                raise FloatingPointError('Finite source IQ and gate scores are required')
            active = logits.sigmoid() > 0.8
            if not active.any():
                continue
            chosen = ids[np.asarray(active.cpu().tolist(), dtype=bool)]
            x = x[active]
            y = core.tensor(clean[chosen], device)
            posterior, logvar = physical(x)
            base = dura(x)
            features = reader.features(y)
            clean_logits = reader.classify_features(features)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.isfinite(features).all() or not torch.isfinite(clean_logits).all():
                raise FloatingPointError('Finite source IQ and clean-reader scores are required')
            base_logits = checked_logits(reader, base)
            groups = torch.tensor(rx[chosen].tolist(), dtype=torch.long, device=device)
            chunks.append(tuple(t.detach().cpu() for t in (x, y, posterior, logvar, base,
                                                            features, clean_logits, base_logits, groups)))
            for view in np.unique(scene[chosen]):
                counts[str(view)] = counts.get(str(view), 0) + int(np.count_nonzero(scene[chosen] == view))
        print('DURA_POSTERIOR_SOURCE_PREP', epoch, json.dumps(counts), flush=True)
    if not chunks:
        raise ValueError('At least one gate-active source training example is required')
    return tuple(torch.cat(items, 0) for items in zip(*chunks)), counts


@torch.no_grad()
def fixed_source_predictions(root, cache, clean, rx, validation, heldout, fit, pipe, reader, device, smoke, source):
    received, scenes = source.epoch_data(cache, 19)
    rng = np.random.default_rng(2026135057)
    ids, groups = [], []
    count = 2 if smoke else 48
    for name, subset in (('RX4_6', validation), ('RX8', heldout)):
        for view in VIEWS:
            eligible = subset[scenes[subset] == view]
            if len(eligible) < count:
                raise ValueError('Every source receiver/scene stratum is required')
            selected = np.sort(rng.choice(eligible, size=count, replace=False))
            ids.extend(selected.tolist())
            groups.extend([name]*count)
    ids = np.asarray(ids, dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or np.intersect1d(ids, fit).size:
        raise ValueError('Unique source evaluation records must be disjoint from this fit set')
    values = []
    for start in range(0, len(ids), BATCH):
        idx = ids[start:start+BATCH]
        x = core.tensor(received[idx], device)
        y = core.tensor(clean[idx], device)
        base = pipe.integrated.dura(x)
        output, active = pipe(x)
        if not torch.equal(output[~active], base[~active]):
            raise ValueError('Closed gates must return the original DURA waveform')
        clean_logits = checked_logits(reader, y)
        confidence, pseudo = F.softmax(clean_logits, dim=1).max(1)
        energy = y.square().mean((1, 2)).clamp_min(1e-8)
        columns = [confidence, pseudo.float(), active.float()]
        for signal in (x, base, output):
            pred = checked_logits(reader, signal).argmax(1)
            error = (signal-y).square().mean((1, 2))/energy
            columns.extend((pred.float(), error))
        row = torch.stack(columns, dim=1).cpu().contiguous()
        if not torch.isfinite(row).all():
            raise ValueError('Source predictions must be finite')
        values.append(row)
    return ids, np.asarray(groups), rx[ids], scenes[ids], np.from_dlpack(torch.cat(values).contiguous())


def main():
    from experiments.phase1_dura_posterior.physical_source_smoke import load_source_models, s
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--posterior-kind', choices=('mean', 'nonlocal'), default='mean')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('A unique source training output is required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026110227)
    cache, clean, rx, training, valid_prior, hold_prior = s.source_contract(args.root)
    validation, heldout = s.new_splits(cache, rx, training, valid_prior, hold_prior)
    prior_ids = np.sort(np.random.default_rng(2026120355).choice(training, size=8192, replace=False))
    fit = select_fit(training, prior_ids, FIT_SIZE)
    if np.intersect1d(fit, np.r_[prior_ids, validation, heldout]).size:
        raise ValueError('Disjoint source fit, source prior and heldout records are required')
    upstream_labeled_fit = 0
    if args.posterior_kind == 'nonlocal':
        parent_root = args.root/'runs/20261102-phase1-posterior-nonlocal-source-s2026092701-r02'
        parent = torch.load(parent_root/'checkpoint.pt', map_location='cpu', weights_only=False)
        if not np.array_equal(parent.get('fit_ids'), fit):
            raise ValueError('The independent nonlocal prior must use the registered source fit IDs')
        upstream_labeled_fit = core.read(parent_root/'prediction_complete.json')['L_s_fit_labels_used']
    selected = fit[:64] if args.smoke else fit
    reader, contract = core.teacher(args.root, args.device)
    _, _, roles = core.data(args.root, contract)
    physical, gate = load_source_models(args.root, args.posterior_kind)
    physical = physical.to(args.device).eval().requires_grad_(False)
    gate = gate.to(args.device).eval().requires_grad_(False)
    dura = frozen_dura(args.root, args.device)
    integrated = PosteriorInDura(dura).to(args.device)
    frozen = (reader, dura, gate, physical)
    before = [snapshot(module) for module in frozen]
    started = time.monotonic()
    data, counts = prepare_source(args.root, cache, selected, clean, rx, gate, physical, dura, reader, args.device, s)
    if args.smoke and len(data[0]) < BATCH:
        raise ValueError('Source smoke needs an active training batch')
    optimizer = torch.optim.AdamW(integrated.corrections.parameters(), lr=0.0003, weight_decay=0.0001)
    rng = torch.Generator().manual_seed(2026110228)
    losses = []
    for epoch in range(1, 2 if args.smoke else EPOCHS+1):
        integrated.train()
        order = torch.randperm(len(data[0]), generator=rng)
        total = 0.
        for start in range(0, len(order), BATCH):
            take = order[start:start+BATCH]
            x, y, posterior, logvar, base, feat, clean_logits, base_logits, groups = [t[take].to(args.device) for t in data]
            output = integrated(x, posterior, logvar)
            loss, terms = source_objective(reader, output, y, base, feat, clean_logits, base_logits, groups)
            if not torch.isfinite(loss):
                raise FloatingPointError('Finite source training loss required')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for correction in integrated.corrections:
                grad = correction.coefficients.weight.grad
                if grad is None or not torch.isfinite(grad).all():
                    raise ValueError('Finite correction gradients required at both DURA stages')
            torch.nn.utils.clip_grad_norm_(integrated.corrections.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(take)
        losses.append(total/len(order))
        print('DURA_POSTERIOR_EPOCH', epoch, 'loss', losses[-1], 'seconds', round(time.monotonic()-started, 1), flush=True)
    if not all(unchanged(module, state) for module, state in zip(frozen, before)):
        raise ValueError('Original DURA, reader, gate and posterior must remain frozen')
    pipe = GatedPosteriorDura(integrated, gate, physical).eval()
    ids, splits, eval_rx, scenes, values = fixed_source_predictions(args.root, cache, clean, rx, validation,
                                                                     heldout, fit, pipe, reader, args.device, args.smoke, s)
    if args.smoke:
        core.write(args.out/'smoke.json', dict(status='SOURCE_TRAINING_SMOKE_PASS', fit_ids=len(selected),
                   active_exposures=len(data[0]), eval_records=len(ids), counts=counts, losses=losses,
                   posterior_kind=args.posterior_kind,
                   upstream_L_s_fit_labels_used=upstream_labeled_fit,
                   F_frozen=True, DURA_frozen=True, U_s_truth_reads=0, TX_labels_read=0, V_reads=0, test_reads=0))
        return
    checkpoint = args.out/'checkpoint.pt'
    with checkpoint.open('xb') as stream:
        torch.save(dict(schema='phase1_dura_posterior_source_v1', commit=args.commit, fit_ids=fit,
                        epochs=EPOCHS, counts=counts, losses=losses,
                        posterior_kind=args.posterior_kind,
                        upstream_L_s_fit_labels_used=upstream_labeled_fit,
                        corrections=integrated.corrections.state_dict()), stream)
    prediction = args.out/'source_predictions.npz'
    with prediction.open('xb') as stream:
        np.savez(stream, indices=ids, split=splits, rx=eval_rx, scene=scenes, values=values)
    core.write(args.out/'prediction_complete.json', dict(status='SOURCE_PREDICTIONS_FIXED', commit=args.commit,
               count=len(ids), fit_ids=len(fit), active_exposures=len(data[0]), counts=counts, losses=losses,
               posterior_kind=args.posterior_kind,
               upstream_L_s_fit_labels_used=upstream_labeled_fit,
               F_frozen=True, DURA_frozen=True, gate_frozen=True, posterior_frozen=True,
               TX_labels_read=0, U_s_truth_reads=0, V_reads=0, test_reads=0, model_updates_after_prediction=0,
               values='clean_confidence,clean_pseudoclass,gate,raw_class,raw_iq,dura_class,dura_iq,new_class,new_iq',
               caveat='Source IDs have been explored; pseudo classes are not identity truth.'))
    print('DURA_POSTERIOR_SOURCE_PREDICTIONS_FIXED', len(ids), flush=True)


if __name__ == '__main__':
    main()
