"""Fixed E200-parent to E40 geometry fine-tuning, with source-only diagnostics."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'code')]
from experiments.cvs_stage2_geometry import design as d
from experiments.cvs_stage2_geometry.actions import load_actions
from experiments.cvs_stage2_geometry.features import identity
from experiments.cvs_stage2_geometry.objectives import keep_loss, lt_loss, contract as objective_contract
from experiments.cvs_stage2_geometry.receiver import load_receiver
from experiments.cvs_stage2_decouple.common import (build_model, verify_parent, IdentityView,
    identity_logits, margins, parameter_groups, learning_rate, clean)
from experiments.cvs_stage2_decouple.source import SourceStream, seed_all
from experiments.cvs_stage2_decouple.deterministic_forward import deterministic_identity, identity_mode_contract
from experiments.cvs_stage2_decouple.source_counterfactual_audit import ReadOnlyState
from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags


def dump(path, value):
    d.write(path, clean(value))


class SourceIdentityView(IdentityView):
    """Always logits; expose only the identity subtree to the time-path probe."""
    @property
    def id_backbone(self):
        return self.model.id_backbone


def save_final_checkpoint(c, model, optimizer, out):
    """One fixed-final serialization path, also exercised by bounded checks."""
    checkpoint = Path(out)/'final_ssdg.pth'
    torch.save(dict(model=model.state_dict(), baseline_args=model._stage2_baseline_args,
        config=c, run_id=c['run_id'], candidate_id=c['row_id'], epoch=40, optimizer_steps=8880,
        checkpoint_selection='final_only', parent_checkpoint=c['parent_checkpoint'],
        optimizer=optimizer.state_dict(), target_access=False, rng=torch.get_rng_state()), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=next(model.parameters()).device,
                                    weights_only=False)['model'], strict=True)
    return checkpoint


def save_completion(c, out, result):
    dump(Path(out)/'completion.json', result)
    dump(Path(out).parent/'source_frozen.json', dict(status='SOURCE_FROZEN', run_id=c['run_id'],
        row_id=c['row_id'], config=c, checkpoint=str(Path(out)/'final_ssdg.pth'), epoch=40,
        optimizer_steps=8880, target_access=False, target_scores_consumed=False,
        selection='fixed_final_E40'))


def objective(c, logits, y, parent_margin, step, *, clean_logits=None, augmented_logits=None,
              features32=None, relation=None, bank=None):
    """CE on all128; finite clean/action risks on the same final32 only."""
    ce = F.cross_entropy(logits, y, label_smoothing=0.)
    zero = logits.sum() * 0.
    keep = finite_lt = receiver = zero
    details = {}
    scheduled = c['enabled']['actions'] and step % c['schedule']['action_every'] == 0
    eligible = 0
    if scheduled:
        if clean_logits is None or augmented_logits is None or clean_logits.shape != (32, logits.shape[1]):
            raise ValueError('Scheduled mechanism requires deterministic paired final32 logits')
        if len(y) != 128 or parent_margin.shape != y.shape:
            raise ValueError('Registered128 identity batch and parent margins required')
        selected_y, selected_parent = y[-32:], parent_margin[-32:]
        keep = keep_loss(clean_logits, selected_parent, selected_y, c['losses']['margin_required'])
        finite_lt = lt_loss(clean_logits, augmented_logits, selected_y, c['losses']['margin_required'])
        eligible = int((selected_parent > 0).sum())
    elif clean_logits is not None or augmented_logits is not None or features32 is not None:
        raise ValueError('Auxiliary forward outside the registered action schedule')
    if c['enabled']['receiver'] and step % c['schedule']['receiver_every'] == 0:
        if not scheduled or features32 is None or relation is None or bank is None:
            raise ValueError('Scheduled R requires the reused deterministic clean joint160')
        receiver, details = bank.identity_loss(features32, relation['indices'], relation)
    ramp = min(step / c['losses']['ramp_steps'], 1.)
    weighted = dict(ce=ce, keep=ramp*c['losses']['keep']*keep,
                    finite_lt=ramp*c['losses']['finite_lt']*finite_lt,
                    receiver=ramp*c['losses']['receiver']*receiver)
    return sum(weighted.values()), weighted, dict(ce=ce, keep=keep, finite_lt=finite_lt,
        receiver=receiver, ramp=ramp, keep_eligible=eligible,
        clean_margin=margins(logits, y).mean(), receiver_details=details)


def forward_objective(c, model, x, y, parent_margin, step, *, actions=None, relation=None, bank=None):
    """Ordinary train CE plus dropout-free differentiable paired mechanisms."""
    times = {}
    cuda = x.is_cuda
    def mark(name):
        if cuda:
            times[name] = torch.cuda.Event(enable_timing=True)
            times[name].record()
        else:
            times[name] = time.perf_counter()
    mark('start')
    logits = identity_logits(model, x)
    mark('ce_end')
    clean_logits = aug_logits = features32 = None
    kind, stats = None, {}
    scheduled = c['enabled']['actions'] and step % c['schedule']['action_every'] == 0
    if scheduled:
        if actions is None:
            raise ValueError('Registered action bank missing')
        kind = c['schedule']['action_order'][(step // c['schedule']['action_every'] - 1) % 4]
        policy = 'coverage' if c['action_policy'] == 'coverage' else 'conditional'
        augmented, stats = actions.transform(x[-32:], kind, c['model_seed'] + step*13, policy=policy)
        with deterministic_identity(model):
            clean_logits, features32 = identity(model, x[-32:], verify=True)
            aug_logits, _ = identity(model, augmented, verify=True)
        cm, am = margins(clean_logits, y[-32:]), margins(aug_logits, y[-32:])
        drop = (cm-am).detach()
        stats.update(realized_margin_drop_mean=float(drop.mean()),
                     realized_margin_drop_q90=float(torch.quantile(drop, .9)),
                     clean_deterministic_margin_mean=float(cm.detach().mean()),
                     augmented_margin_mean=float(am.detach().mean()),
                     augmented_accuracy=float((aug_logits.argmax(1) == y[-32:]).float().mean()),
                     source_relation_reused=relation is not None, selected_positions='96:128',
                     dropout_pair_mode='identity Dropout family disabled, autograd retained',
                     no_augmented_ce=True)
    mark('aux_end')
    total, weighted, parts = objective(c, logits, y, parent_margin, step,
        clean_logits=clean_logits, augmented_logits=aug_logits, features32=features32,
        relation=relation, bank=bank)
    mark('objective_end')
    if cuda:
        times['objective_end'].synchronize()
        elapsed = lambda a, b: times[a].elapsed_time(times[b])/1000.
    else:
        elapsed = lambda a, b: times[b]-times[a]
    timing = dict(main_forward_seconds=elapsed('start', 'ce_end'),
                  action_transform_and_paired_forward_seconds=elapsed('ce_end', 'aux_end'),
                  objective_seconds=elapsed('aux_end', 'objective_end'))
    return total, weighted, parts, dict(logits=logits, clean_logits=clean_logits,
        features32=features32, action_kind=kind, action_stats=stats, timing=timing)


def gradient_diagnostics(weighted, raw, groups, ce_epsilon=1e-8):
    """Absolute norms first. Ratios are diagnostic only and never tune weights."""
    result = dict(ce_ratio_denominator_epsilon=ce_epsilon, ratios_used_for_weighting=False)
    for prefix, components in (('weighted', weighted), ('raw', raw)):
        for name, value in components.items():
            for group in groups:
                grads = torch.autograd.grad(value, group['params'], retain_graph=True,
                    allow_unused=True) if value.requires_grad else []
                squared = sum(g.detach().float().square().sum() for g in grads if g is not None)
                result[f'{prefix}_grad_{name}_{group["name"]}'] = (
                    float(squared.sqrt()) if torch.is_tensor(squared) else 0.)
        for group in groups:
            base = result[f'{prefix}_grad_ce_{group["name"]}']
            result[f'{prefix}_ce_tiny_{group["name"]}'] = base <= ce_epsilon
            for name in components:
                if name != 'ce':
                    result[f'{prefix}_ratio_{name}_{group["name"]}'] = (
                        result[f'{prefix}_grad_{name}_{group["name"]}']/base if base > ce_epsilon else None)
    return result


@torch.no_grad()
def relation_observations(logits, deterministic_logits, y, parent_margin, sample):
    if sample is None:
        return []
    records = []
    for relation in sample['relations']:
        if not relation['valid']:
            continue
        record = dict(slot=relation['slot'], c=relation['c'], active=relation['active'],
                      group_ids=relation['group_ids'], alpha=relation['alpha'])
        for name, positions in zip(('g', 'h'), relation['positions']):
            p = torch.tensor(positions, device=y.device)
            parent = parent_margin[-32:][p]
            train = margins(logits[-32:], y[-32:])[p]
            record[name] = dict(train_margin=float(train.mean()), parent_margin=float(parent.mean()),
                train_accuracy=float((logits[-32:][p].argmax(1) == y[-32:][p]).float().mean()),
                parent_correct=int((parent > 0).sum()), count=len(p))
            if deterministic_logits is not None:
                det = margins(deterministic_logits, y[-32:])[p]
                record[name].update(deterministic_margin=float(det.mean()),
                    deterministic_parent_margin_change=float((det-parent).mean()),
                    deterministic_accuracy=float((deterministic_logits[p].argmax(1) == y[-32:][p]).float().mean()))
        records.append(record)
    return records


def train(c, device='cuda:0', *, design_module=d):
    from experiments.cvs_stage2_decouple.data import cached_loader
    from experiments.cvs_feature_disentangle.validation import evaluate_source_stress
    from experiments.cvs_feature_disentangle.source import source_validation
    from experiments.cvs_phase1_stack.recover import verify_contract
    d = design_module
    d.validate(c)
    parent_c = d.parent_config(c['model_seed'])
    prepared = Path(c['preparation'])
    pdone = d.read(prepared/'preparation_complete.json')
    if (pdone['status'] != 'SOURCE_GEOMETRY_AUDIT_COMPLETE' or pdone['seed'] != c['model_seed'] or
            pdone['target_access'] or not pdone['model_state_restored'] or pdone['count'] != 6300 or
            pdone['identity_updates'] != 0 or pdone['parent_contract'] != parent_c):
        raise ValueError('Invalid frozen geometry preparation')
    provenance = verify_parent(parent_c)
    packed = torch.load(prepared/'source_cache.pt', map_location='cpu', weights_only=False)
    verify_contract(packed['contract'], d.read(d.SOURCE))
    verify_contract(pdone['source_contract'], packed['contract'])
    cache = packed['labeled']
    if (len(cache['ids']) != 6300 or len(set(cache['ids'])) != 6300 or
            sorted(cache['ids']) != sorted(packed['contract']['role_ids']['L_s'])):
        raise ValueError('Cached L_s physical IDs differ')
    if (len(packed['validation']['ids']) != 27000 or
            sorted(packed['validation']['ids']) != sorted(packed['contract']['role_ids']['V'])):
        raise ValueError('Cached V physical IDs differ')
    bank = load_receiver(torch.load(c.get('receiver_payload', prepared/'receiver.pt'), map_location='cpu', weights_only=False))
    if c.get('receiver_method') is not None and bank.payload.get('method') != c['receiver_method']:
        raise ValueError('Frozen receiver estimator differs from registered method')
    if (bank.payload['seed'] != c['model_seed'] or bank.payload['source_ids'] != cache['ids'] or
            not torch.equal(bank.parent_feature, cache['parent_feature'])):
        raise ValueError('Frozen receiver/cache physical identity or parent vectors differ')
    out = Path(c['output_root'])
    out.mkdir(parents=True, exist_ok=False)
    vloader = cached_loader(packed['validation'], 256)
    stream = SourceStream(len(cache['x']), bank, c['model_seed'])
    seed_all(c['model_seed'])
    torch.set_num_threads(2)
    started = time.perf_counter()
    with numerical_context(d.FULL_FP32_POLICY):
        model = build_model(parent_c, device)
        groups, names = parameter_groups(model)
        optimizer = torch.optim.AdamW(groups, weight_decay=c['optimizer']['weight_decay'])
        actions = None
        if c['enabled']['actions']:
            payload = torch.load(prepared/'actions.pt', map_location='cpu', weights_only=False)
            if payload['seed'] != c['model_seed'] or payload['source_ids'] != cache['ids']:
                raise ValueError('Frozen action source seed or physical order differs')
            actions = load_actions(payload, device)
        model.train()
        mode_contract = identity_mode_contract(model)
        resolved = dict(c, pid=os.getpid(), cwd=os.getcwd(), python=sys.executable,
            commit=(ROOT/'release_commit.txt').read_text().strip() if (ROOT/'release_commit.txt').exists() else 'local_check',
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), backend_flags=actual_flags(),
            hardware=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',
            parameter_groups=names, trainable_parameters=sum(p.numel() for g in groups for p in g['params']),
            total_parameters=sum(p.numel() for p in model.parameters()), optimizer_state='fresh AdamW after strict parent load',
            deterministic_identity=mode_contract, finite_objective=objective_contract(),
            execution=dict(parent_resident=False, teacher=False, unlabeled_iterator=False, satellite_generator=False,
                action_forward=c['enabled']['actions'], augmented_ce=False, ce_ratio_calibration=False,
                receiver_network_forward=False, relation_sampling_all_arms=True,
                auxiliary_fit_during_identity_training=False, paired_deterministic_forward_every=4 if actions else None))
        dump(out/'resolved_config.json', resolved)
        dump(out/'initialization.json', provenance)
        dump(out/'source_contract.json', packed['contract'])
        print('RESOLVED '+json.dumps(clean(resolved)), flush=True)
        counts = dict(identity_updates=0, linear=0, temporal=0, relation_batches=0,
            receiver_attempts=0, receiver_active_relations=0, ce_packets=0, mechanism_clean_packets=0,
            mechanism_augmented_packets=0, teacher_calls=0, unlabeled_packets=0, satellite_calls=0,
            augmented_ce_calls=0, auxiliary_optimizer_steps=0)
        epoch_rows = []
        cuda = str(device).startswith('cuda')
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        with gzip.open(out/'step_metrics.jsonl.gz', 'xt', encoding='utf-8', compresslevel=3) as steps:
            for epoch in range(1, c['epochs']+1):
                epoch_started = time.perf_counter()
                sums = {}
                last_gradient_audit = {}
                model.train()
                for local in range(1, c['steps_per_epoch']+1):
                    step = (epoch-1)*c['steps_per_epoch']+local
                    indices, relation = stream.step(step)
                    x, y = cache['x'][indices].to(device), cache['y'][indices].to(device)
                    pm = cache['parent_margin'][indices].to(device)
                    for group in optimizer.param_groups:
                        group['lr'] = learning_rate(step, group['peak'])
                    optimizer.zero_grad(set_to_none=True)
                    loss, weighted, parts, observed = forward_objective(c, model, x, y, pm, step,
                        actions=actions, relation=relation, bank=bank)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite loss; no skipped update or automatic retry')
                    audit_started = time.perf_counter()
                    grad = gradient_diagnostics(weighted, {k:parts[k] for k in weighted}, groups) if step % 888 == 0 else {}
                    if grad:
                        last_gradient_audit = dict(step=step, **grad)
                    if cuda:
                        torch.cuda.synchronize(device)
                    audit_seconds = time.perf_counter()-audit_started
                    backward_started = time.perf_counter()
                    loss.backward()
                    gradnorm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], float('inf'), error_if_nonfinite=True)
                    optimizer.step()
                    if cuda:
                        torch.cuda.synchronize(device)
                    backward_seconds = time.perf_counter()-backward_started
                    counts['identity_updates'] += 1
                    counts['ce_packets'] += 128
                    if observed['action_kind'] is not None:
                        counts[observed['action_kind']] += 1
                        counts['mechanism_clean_packets'] += 32
                        counts['mechanism_augmented_packets'] += 32
                    if relation is not None:
                        counts['relation_batches'] += 1
                    if relation is not None and c['enabled']['receiver']:
                        counts['receiver_attempts'] += 1
                        counts['receiver_active_relations'] += parts['receiver_details']['active_relations']
                    scalars = {k:float(v.detach()) if torch.is_tensor(v) else v for k,v in parts.items() if k != 'receiver_details'}
                    scalars.update({'weighted_'+k:float(v.detach()) for k,v in weighted.items()})
                    correct = pm > 0
                    scalars.update(total_loss=float(loss.detach()), total_gradient_norm=float(gradnorm),
                        lr_encoder=optimizer.param_groups[0]['lr'], lr_fusion_head=optimizer.param_groups[1]['lr'],
                        gradient_audit_seconds=audit_seconds, backward_optimizer_seconds=backward_seconds,
                        parent_correct_count=int(correct.sum()),
                        parent_correct_retained_count=int(((observed['logits'].argmax(1) == y)&correct).sum()),
                        **observed['timing'])
                    for key,value in scalars.items():
                        sums[key] = sums.get(key, 0.)+value
                    record = dict(step=step, epoch=epoch, metrics=scalars, gradient_audit=grad,
                        action_kind=observed['action_kind'], action_stats=observed['action_stats'],
                        receiver=parts['receiver_details'], relation_observations=relation_observations(
                            observed['logits'], observed['clean_logits'], y, pm, relation),
                        relation_sampling=relation['diagnostics'] if relation else None)
                    steps.write(json.dumps(clean(record), allow_nan=False)+'\n')
                steps.flush()
                with ReadOnlyState(model) as validation_state:
                    model.eval()
                    with torch.no_grad():
                        validation = source_validation(IdentityView(model), vloader, device)
                if not all(validation_state.report.values()):
                    raise ValueError('Source V changed identity state')
                row = dict(epoch=epoch, optimizer_steps=counts['identity_updates'],
                    epoch_time_s=time.perf_counter()-epoch_started,
                    **{k:v/c['steps_per_epoch'] for k,v in sums.items()}, **validation, **counts,
                    gradient_audit=last_gradient_audit or None)
                epoch_rows.append(clean(row))
                (out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in epoch_rows), encoding='utf-8')
                flat = [{k:v for k,v in r.items() if not isinstance(v, (dict,list))} for r in epoch_rows]
                with (out/'epoch_metrics.csv').open('w', encoding='utf-8', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
                    writer.writeheader()
                    writer.writerows(flat)
                print('EPOCH '+json.dumps(clean(row), allow_nan=False), flush=True)
        if counts['identity_updates'] != 8880 or counts['relation_batches'] != 1110:
            raise ValueError('Incomplete identity/relationship budget')
        if actions and (counts['linear'] != 1110 or counts['temporal'] != 1110):
            raise ValueError('Incomplete balanced action budget')
        training_seconds = time.perf_counter()-started
        checkpoint = save_final_checkpoint(c, model, optimizer, out)
        evaluate_source_stress(SourceIdentityView(model), vloader, device, c, out)
        preparation_audit = d.read(prepared/'audit.json')
        peak = torch.cuda.max_memory_allocated(device) if cuda else None
        result = dict(status='SOURCE_TRAINED', epoch=40, optimizer_steps=8880, logged_steps=8880,
            config=c, checkpoint=str(checkpoint), checkpoint_selection='final_only', target_access=False,
            target_evaluated=False, final_source_metrics=validation, mechanism_execution=counts,
            elapsed_seconds=time.perf_counter()-started, preparation_seconds=preparation_audit['elapsed_seconds'],
            shared_preparation_seed=c['model_seed'], peak_cuda_memory_bytes=peak,
            resources=dict(identity_training_including_epoch_validation_seconds=training_seconds,
                final_source_diagnostics_seconds=time.perf_counter()-started-training_seconds,
                peak_cuda_memory_bytes=peak, trainable_parameters=resolved['trainable_parameters'],
                total_parameters=resolved['total_parameters'], parent_training_epochs=200, incremental_epochs=40,
                timing_scope='shared GPU wall; parent E200 and per-seed shared preparation reported separately'),
            parent_provenance=provenance, source_stress=str(out/'source_stress.json'))
        save_completion(c, out, result)
        print('SOURCE_TRAINED '+json.dumps(clean(result)), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    train(d.read(args.config), args.device)
