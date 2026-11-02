"""Train dynamic Volterra restoration on paired Phase1 source IQ."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.phase1_dynamic_volterra.model import DynamicVolterra
from experiments.phase1_equivariant_dictionary.objective import source_loss
from experiments.phase1_spectral_synthesis.fit_sampling import select_fit

EPOCHS = 5
FIT_SIZE = 8192
BATCH = 8
SCHEMA = 'phase1_dynamic_volterra_v1'
LOG_PREFIX = 'DYNAMIC_VOLTERRA'
PAIRED_VIEWS = False
USES_DURA = False
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')


def snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def unchanged(module, before):
    return (not any(p.requires_grad or p.grad is not None for p in module.parameters())
            and all(torch.equal(value.detach().cpu(), before[key]) for key, value in module.state_dict().items()))


def training_indices(chosen, records_per_view, paired):
    if paired:
        return torch.cat([chosen+i*records_per_view for i in range(3)])
    return chosen


def extra_frozen_components(stats, eye):
    return ()


def extra_frozen_model_components(model):
    return ()


def prepare_training_input(received):
    """Optional source-only augmentation; default preserves cached source views."""
    return received


def source_objective(reader, output, anchor, clean, clean_features, groups):
    return source_loss(reader, output, clean, clean_features, groups)


def training_source_objective(reader, output, anchor, clean, clean_features, groups, received, log_variance):
    return source_objective(reader, output, anchor, clean, clean_features, groups)


def auxiliary_source_objective(received, output, groups):
    return output.new_zeros(())


def combine_source_objectives(main_loss, auxiliary_loss, model):
    return main_loss + auxiliary_loss


def load_components(root, device, deps):
    blind_state = torch.load(root/'runs'/deps.SOURCE/'checkpoint_epoch10.pt', map_location='cpu', weights_only=False)
    fixed = torch.load(root/'runs'/deps.s.GATE_RUN/'source/selected.pt', map_location='cpu', weights_only=False)
    if (blind_state.get('commit') != '94693a342bb2daa684c0f21415d7b949f9fb2107'
            or blind_state.get('epoch') != 10 or fixed.get('commit') != deps.s.GATE_COMMIT
            or fixed.get('selected', {}).get('kind') != 'gated_direct_iq'):
        raise ValueError('Fixed source-only blind action and gate are required')
    blind = deps.BlindActionNet().to(device); blind.load_state_dict(blind_state['model'], strict=True)
    gate = deps.s.BlindGate().to(device); gate.load_state_dict(fixed['gate'], strict=True)
    return blind.eval().requires_grad_(False), gate.eval().requires_grad_(False), None, None


@torch.no_grad()
def input_noise(x, blind, stats, eye, deps):
    _, logvar = blind(x)
    if logvar.shape != (len(x),) or not torch.isfinite(x).all() or not torch.isfinite(logvar).all():
        raise FloatingPointError('Finite input IQ and one estimated noise value per window are required')
    return x, logvar


@torch.no_grad()
def predict_batch(x, model, blind, gate, stats, eye, reader, deps):
    scores = gate(x).reshape(-1)
    if scores.shape != (len(x),) or not torch.isfinite(x).all() or not torch.isfinite(scores).all():
        raise FloatingPointError('Finite received waveform and gate scores are required')
    active = scores.sigmoid() > 0.8
    parent = x.clone()
    output = x.clone()
    if active.any():
        received_base, logvar = input_noise(x[active], blind, stats, eye, deps)
        parent[active] = received_base
        output[active] = model(x[active], received_base, logvar)
    choices = []
    for value in (x, parent, output):
        logits = reader.classify_features(reader.features(value))
        if logits.shape != (len(x), 6) or not torch.isfinite(value).all() or not torch.isfinite(logits).all():
            raise FloatingPointError('Finite waveform and reader scores are required')
        choices.append(logits.argmax(1))
    return parent, output, active, choices


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
    import source_pilot as deps
    core, s = deps.core, deps.s
    if tuple(core.VIEWS) != VIEWS:
        raise ValueError('The established seven Phase1 views are required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source training output required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026110317)
    cache, clean, rx, training, valid_prior, hold_prior = s.source_contract(args.root)
    validation, heldout = s.new_splits(cache, rx, training, valid_prior, hold_prior)
    prior_ids = np.sort(np.random.default_rng(2026120355).choice(training, size=8192, replace=False))
    fit = select_fit(training, prior_ids, FIT_SIZE)
    if np.intersect1d(fit, np.r_[prior_ids, validation, heldout]).size:
        raise ValueError('Disjoint source fit, physical-prior fit and source holdouts are required')
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    reader, _ = core.teacher(args.root, args.device)
    if args.smoke:
        selected = np.sort(np.r_[fit[roles[fit] == 'L_s'][:32], fit[roles[fit] == 'U_s'][:32]])
        if len(selected) != 64:
            raise ValueError('Both source roles are required in the smoke batch')
    else:
        selected = fit
    blind, gate, stats, eye = load_components(args.root, args.device, deps)
    reader.eval().requires_grad_(False)
    model = DynamicVolterra().to(args.device)
    model_frozen = extra_frozen_model_components(model)
    frozen = (reader, blind, gate, *extra_frozen_components(stats, eye), *model_frozen)
    before = [snapshot(module) for module in frozen]
    fresh = args.root/'runs/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/source_cache'
    if (fresh/'train/physical_ids.npy').resolve() != (cache/'train/physical_ids.npy').resolve():
        raise ValueError('Fresh source views must preserve physical IDs')
    chunks, scene_counts = [], {}
    started = time.monotonic()
    for epoch in (21, 22, 23):
        info = core.read(fresh/'epochs'/f'{epoch:02d}'/'complete.json')
        if (info.get('count') != 63000 or info.get('namespace') != f'learned-action/train/epoch{epoch:02d}'
                or any(info.get(k) != 0 for k in ('TX_labels_read', 'F_outputs_read', 'target_reads'))):
            raise ValueError('Approved source-only view required')
        received, scene = s.epoch_data(fresh, epoch)
        for start in range(0, len(selected), BATCH):
            idx = selected[start:start+BATCH]
            x = prepare_training_input(core.tensor(received[idx], args.device))
            target = core.tensor(clean[idx], args.device)
            base, logvar = input_noise(x, blind, stats, eye, deps)
            with torch.no_grad():
                features = reader.features(target)
            if not all(torch.isfinite(t).all() for t in (x, target, features)):
                raise FloatingPointError('Finite paired source data and frozen-reader scores are required')
            groups = torch.tensor(rx[idx].tolist(), dtype=torch.long, device=args.device)
            chunks.append(tuple(t.detach().cpu() for t in (x, base, logvar, target, features, groups)))
            for view in np.unique(scene[idx]):
                scene_counts[str(view)] = scene_counts.get(str(view), 0)+int(np.count_nonzero(scene[idx] == view))
        print(LOG_PREFIX+'_SOURCE_PREP', epoch, json.dumps(scene_counts), 'seconds', round(time.monotonic()-started, 1), flush=True)
    if not chunks:
        raise ValueError('Paired source training data required')
    data = tuple(torch.cat(items, 0) for items in zip(*chunks))
    del chunks
    if PAIRED_VIEWS:
        size = len(selected)
        if (len(data[0]) != 3*size or not torch.equal(data[3][:size], data[3][size:2*size])
                or not torch.equal(data[3][:size], data[3][2*size:])
                or not torch.equal(data[5][:size], data[5][size:2*size])
                or not torch.equal(data[5][:size], data[5][2*size:])):
            raise ValueError('Three source views must preserve the same physical-record order')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(2026110318)
    losses = []
    adaptive_gradient_batches = 0
    for epoch in range(1, 2 if args.smoke else EPOCHS+1):
        order = torch.randperm(len(selected) if PAIRED_VIEWS else len(data[0]), generator=generator)
        total = 0.
        model.train()
        batch_size = 8 if PAIRED_VIEWS else 16
        for start in range(0, len(order), batch_size):
            chosen = training_indices(order[start:start+batch_size], len(selected), PAIRED_VIEWS)
            x, base, lv, target, features, groups = [t[chosen].to(args.device) for t in data]
            output = model(x, base, lv)
            loss, _ = training_source_objective(reader, output, base, target, features, groups, x, lv)
            loss = combine_source_objectives(loss, auxiliary_source_objective(x, output, groups), model)
            if not torch.isfinite(loss):
                raise FloatingPointError('Finite source training loss is required')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = model.gate_output.weight.grad
            if gradient is None or not torch.isfinite(gradient).all() or (epoch == 1 and start == 0 and gradient.abs().sum() <= 0):
                raise ValueError('The residual gate needs finite nonzero initial gradients')
            if model.entry.weight.grad is not None and model.entry.weight.grad.abs().sum() > 0:
                adaptive_gradient_batches += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(chosen)
        losses.append(total/len(data[0]))
        print(LOG_PREFIX+'_EPOCH', epoch, 'loss', losses[-1], 'seconds', round(time.monotonic()-started, 1), flush=True)
    if adaptive_gradient_batches < 1 or not all(unchanged(module, state) for module, state in zip(frozen, before)):
        raise ValueError('Dynamic memory network must learn while reader, blind action and gate remain frozen')
    model.eval().requires_grad_(False)
    received, scene = s.epoch_data(cache, 19)
    rng = np.random.default_rng(2026135057)
    ids, splits = [], []
    per_cell = 2 if args.smoke else 48
    for group, subset in (('RX4_6', validation), ('RX8', heldout)):
        for view in VIEWS:
            eligible = subset[scene[subset] == view]
            if len(eligible) < per_cell:
                raise ValueError('All source receiver/scene cells are required')
            chosen = np.sort(rng.choice(eligible, size=per_cell, replace=False))
            ids.extend(chosen.tolist())
            splits.extend([group]*per_cell)
    ids = np.asarray(ids, dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or np.intersect1d(ids, fit).size:
        raise ValueError('Source evaluation IDs must be unique and disjoint from model fitting')
    values = []
    with torch.no_grad():
        for start in range(0, len(ids), BATCH):
            idx = ids[start:start+BATCH]
            x = core.tensor(received[idx], args.device)
            y = core.tensor(clean[idx], args.device)
            base, output, active, classes = predict_batch(x, model, blind, gate, stats, eye, reader, deps)
            clean_logits = reader.classify_features(reader.features(y))
            if not torch.isfinite(y).all() or not torch.isfinite(clean_logits).all():
                raise FloatingPointError('Finite source clean waveform and reader scores are required')
            confidence, pseudo = F.softmax(clean_logits, dim=1).max(1)
            energy = y[:, :, 64:-64].square().sum(1).mean(1).clamp_min(1e-12)
            row = [confidence, pseudo.float(), active.float()]
            for wave, predicted in zip((x, base, output), classes):
                _, error = s.align(wave, y)
                row.extend((predicted.float(), error/energy))
            values.append(torch.stack(row, dim=1).cpu())
    values = np.from_dlpack(torch.cat(values).contiguous())
    if values.shape != (len(ids), 9) or not np.isfinite(values).all():
        raise ValueError('Finite complete fixed source predictions are required')
    summary = dict(commit=args.commit, fit_ids=len(selected), active_exposures=len(data[0]),
                   source_view_counts=scene_counts, losses=losses, adaptive_gradient_batches=adaptive_gradient_batches,
                   F_frozen=True, blind_frozen=True, gate_frozen=True, DURA_loaded=USES_DURA, paired_views=PAIRED_VIEWS,
                   extra_frozen_model_count=len(model_frozen),
                   TX_labels_read=0,
                   U_s_truth_reads=0, V_reads=0, test_reads=0)
    if USES_DURA:
        summary['DURA_frozen'] = True
    if args.smoke:
        core.write(args.out/'smoke.json', dict(status='SOURCE_TRAINING_SMOKE_PASS', eval_records=len(ids), **summary))
        return
    with (args.out/'checkpoint.pt').open('xb') as stream:
        torch.save(dict(schema=SCHEMA, commit=args.commit, fit_ids=fit,
                        epochs=EPOCHS, model=model.state_dict(), losses=losses), stream)
    with (args.out/'source_predictions.npz').open('xb') as stream:
        np.savez(stream, indices=ids, split=np.asarray(splits), rx=rx[ids], scene=scene[ids], values=values)
    core.write(args.out/'prediction_complete.json', dict(status='SOURCE_PREDICTIONS_FIXED', count=len(ids),
               model_updates_after_prediction=0,
               columns='clean_confidence,clean_pseudoclass,gate,raw_class,raw_aligned_iq,parent_class,parent_aligned_iq,restored_class,restored_aligned_iq',
               caveat='Source records have previously been explored; pseudo labels are not identity truth.', **summary))
    print(LOG_PREFIX+'_SOURCE_PREDICTIONS_FIXED', len(ids), flush=True)


if __name__ == '__main__':
    main()
