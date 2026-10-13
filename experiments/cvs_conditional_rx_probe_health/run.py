"""Source-only fixed FINAL30 diagnostics; no target prediction or scoring entry point."""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import design as d
from experiments.cvs_source_pseudotarget_health.frozen import preserve_rng, snapshot, assert_unchanged
from experiments.cvs_source_pseudotarget_health.stats import calibration, validate_role
from experiments.cvs_source_pseudotarget_health.probes import capture, descriptors, partition, run_probes


def _write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify_registration():
    spec = d.read(d.ROOT / 'registration' / 'experiment.json')
    if spec.get('run_id') != d.RUN:
        raise ValueError('Registration run ID differs')
    matrix = spec.get('rows')
    expected = {row['row_id']: d.config(row) for row in d.rows()}
    if not isinstance(matrix, list) or len(matrix) != len(expected):
        raise ValueError('Registration must contain all nine fixed rows')
    if len({row.get('row_id') for row in matrix}) != len(expected):
        raise ValueError('Registration row IDs repeat')
    for row in matrix:
        if row.get('row_id') not in expected or row.get('config') != expected[row['row_id']]:
            raise ValueError('Registration source probe row/config differs')
    return spec


def load_models(device):
    # These two provenance/model functions read only source artifacts. Do not call
    # the parent module's prediction, target preflight, evaluate or score helpers.
    from experiments.cvs_conditional_rx_adversary.predict import source_proof, own_model
    models = []; contracts = []
    for row in d.rows():
        c = d.source_config(row); source = source_proof(c)
        model = own_model(c, 'FINAL', device)
        if hasattr(model, 'auxiliary'):
            raise ValueError('Training RX auxiliary state must not enter frozen identity')
        models.append(model)
        contracts.append(dict(config=d.config(row), source_config=c,
                              source_contract=d.read(source / 'source_contract.json'),
                              source_frozen=d.read(source / 'source_frozen.json'),
                              actual_training=d.read(source / 'completion.json'),
                              actual_resolved=d.read(source / 'resolved_config.json'),
                              checkpoint=(source / 'FINAL.pth').as_posix(),
                              checkpoint_contract='VERIFIED exact source roles; own scratch FINAL30',
                              auxiliary_state_loaded=False, identity_updates=0,
                              source_performance_gate=False, target_access=False,
                              U_cache_opened=False, U_truth_read=False,
                              reused_for='source diagnostic only; identity never updated'))
    return models, contracts


def validate_source_role(cache, role, expected_ids):
    if role not in d.ROLE_COUNTS:
        raise ValueError('Only L_s and V source roles are consumed')
    if not isinstance(cache, dict) or set(cache) != {'x', 'y', 'rx', 'day', 'ids'}:
        raise ValueError('Legal raw labeled source schema required')
    count = len(cache['ids'])
    for name in ('rx', 'day'):
        if not torch.is_tensor(cache[name]) or cache[name].shape != (count,) or cache[name].dtype != torch.long:
            raise ValueError('Legal source RX/day label vector required')
    count = validate_role(cache, role, expected_ids)
    if any(value.requires_grad for value in cache.values() if torch.is_tensor(value)):
        raise ValueError('Detached raw source role required')
    return count


def load_roles():
    from experiments.cvs_nocfo_foundation.source import load_source
    c = d.source_config(d.rows()[0])
    if c['source_loader_config']['unlabeled']:
        raise ValueError('Diagnostic loader must leave the U cache unopened')
    L, unused_U, V, contract = load_source(c['source_loader_config'])
    if unused_U is not None:
        raise ValueError('Expected hidden U to remain unopened')
    roles = {'L_s': L, 'V': V}
    for role, cache in roles.items():
        if validate_source_role(cache, role, contract['role_ids'][role]) != d.ROLE_COUNTS[role]:
            raise ValueError('Fixed source role count differs')
    if set(L['ids']) & set(V['ids']):
        raise ValueError('Source L_s and V roles overlap')
    return roles, contract


@torch.no_grad()
def scan(models, cache, role, device):
    if role not in d.ROLE_COUNTS:
        raise ValueError('Only frozen L_s/V scanning is allowed')
    probabilities = [[] for _ in models]
    features = [{name: [] for name in d.FEATURES} for _ in models]
    for begin in range(0, len(cache['ids']), d.BATCH_SIZE):
        x = cache['x'][begin:begin + d.BATCH_SIZE].to(device)
        for i, model in enumerate(models):
            with capture(model) as maps:
                values = model.forward_identity_only(x)
            current = descriptors(maps, values['id_feat_joint'])
            if tuple(current) != d.FEATURES or any(
                    current[name].shape != (len(x), dim)
                    for name, dim in zip(d.FEATURES, d.DIMENSIONS)):
                raise ValueError('Actual fixed descriptor dimensions differ')
            for name, tensor in current.items():
                features[i][name].append(tensor)
            logits = values['tx_logits']
            if logits.shape != (len(x), d.CLASSES) or not torch.isfinite(logits).all():
                raise FloatingPointError('Source frozen ordinary cosine logits differ')
            probabilities[i].append(logits.softmax(1).cpu())
        if begin % 16384 == 0:
            print(f'SOURCE_SCAN role={role} inputs_per_model={min(begin + len(x), len(cache["ids"]))}', flush=True)
    return torch.stack([torch.cat(parts) for parts in probabilities]), [
        {name: torch.cat(parts) for name, parts in value.items()} for value in features]


def source_summary(q, cache, role):
    if role not in d.ROLE_COUNTS:
        raise ValueError('Only labeled source L_s/V health summaries are allowed')
    if cache['y'].shape != (len(q),) or cache['y'].dtype != torch.long:
        raise ValueError('Source TX labels differ')

    def group(mask):
        if not mask.any():
            raise ValueError('Empty source health group')
        selected = q[mask]; y = cache['y'][mask]
        value = calibration(selected, y)
        predicted = selected.argmax(1)
        confusion = torch.bincount(y * d.CLASSES + predicted,
                                   minlength=d.CLASSES ** 2).reshape(d.CLASSES, d.CLASSES)
        value.update(confusion=confusion.tolist(), TX_order=list(range(d.CLASSES)))
        return value

    rx = cache['rx']; day = cache['day']; y = cache['y']
    return dict(role=role, summary=group(torch.ones(len(q), dtype=torch.bool)),
                RX={str(r): group(rx == r) for r in d.RX},
                day={str(dayvalue): group(day == dayvalue) for dayvalue in d.DAYS},
                RX_day={f'RX{r}_day{dayvalue}': group((rx == r) & (day == dayvalue))
                        for r in d.RX for dayvalue in d.DAYS},
                labeled_TX={str(tx): group(y == tx) for tx in range(d.CLASSES)},
                source_ordinary_cosine_health=True, performance_gate=False,
                target_access=False, U_cache_opened=False, U_truth_read=False)


def _probe_summary(probes):
    if tuple(probes['layers']) != d.FEATURES:
        raise ValueError('All three fixed source probe layers required')
    layers = {}
    for level in d.FEATURES:
        records = probes['layers'][level]
        if set(records) != {'unconditional', *(f'TX{tx}' for tx in range(d.CLASSES))}:
            raise ValueError('All independent source RX probe heads required')
        layers[level] = {}
        for role in ('training', 'L_fold2', 'source_V'):
            conditional = [records[f'TX{tx}'][role] for tx in range(d.CLASSES)]
            count = sum(item['count'] for item in conditional)
            correct = sum(item['correct_count'] for item in conditional)
            layers[level][role] = dict(
                unconditional=dict(records['unconditional'][role]),
                TX_conditional=dict(count=count, correct_count=correct,
                                    accuracy=correct / count,
                                    loss=sum(item['count'] * item['loss'] for item in conditional) / count,
                                    confusion=np.sum([item['confusion'] for item in conditional], axis=0).tolist(),
                                    RX_order=list(d.RX), aggregation='micro across all six true source TX groups'))
    return layers


def analysis(records):
    expected = d.rows()
    if len(records) != len(expected) or {item['row_id'] for item in records} != {row['row_id'] for row in expected}:
        raise ValueError('Full fixed nine-row source probe matrix required')
    compact = []
    for row in expected:
        item = next(v for v in records if v['row_id'] == row['row_id'])
        compact.append(dict(**row, source_TX={role: item['source_summary'][role]['summary']
                                            for role in d.ROLE_COUNTS},
                            RX_probes=_probe_summary(item['rx_probes'])))
    groups = []
    for arm in d.ARMS:
        members = [item for item in compact if item['arm'] == arm]
        group = dict(arm=arm, seeds=list(d.SEEDS), rows=len(members),
                     source_TX_accuracy_mean={role: float(np.mean([v['source_TX'][role]['accuracy'] for v in members]))
                                              for role in d.ROLE_COUNTS}, layers={})
        for level in d.FEATURES:
            group['layers'][level] = {role: {
                kind: dict(accuracy_mean=float(np.mean([v['RX_probes'][level][role][kind]['accuracy'] for v in members])),
                           accuracy_std=float(np.std([v['RX_probes'][level][role][kind]['accuracy'] for v in members], ddof=1)))
                for kind in ('unconditional', 'TX_conditional')}
                for role in ('training', 'L_fold2', 'source_V')}
        groups.append(group)
    paired = []
    for item in compact:
        if item['arm'] == 'DETACHED_AUX':
            continue
        base = next(v for v in compact if v['arm'] == 'DETACHED_AUX' and v['model_seed'] == item['model_seed'])
        paired.append(dict(row_id=item['row_id'], model_seed=item['model_seed'], control_row=base['row_id'],
                           layers={level: {role: {
                               kind: 100 * (item['RX_probes'][level][role][kind]['accuracy'] -
                                            base['RX_probes'][level][role][kind]['accuracy'])
                               for kind in ('unconditional', 'TX_conditional')}
                               for role in ('training', 'L_fold2', 'source_V')}
                               for level in d.FEATURES}, unit='percentage_points'))
    return dict(status='SOURCE_PROBE_ANALYZED', run_id=d.RUN, rows=compact,
                three_seed=groups, paired=paired, probe_count=189,
                probe_optimizer_updates=18900, probe_diagnostic_confusions=567,
                identity_updates=0, V_updates=0, U_cache_opened=False,
                target_access=False, target_metrics_consumed=False,
                source_performance_gate=False,
                limitations=['L fold2 held out only from the new linear probe; identity previously trained on all L.',
                             'V contains known source RX; this is not an unseen RX or identity blind test.',
                             'Same TX grouping is observational RX information, not a pure RX causal decomposition.',
                             'Independent probe state is discarded and never enters identity or query inference.'])


def _rng_state(device):
    return dict(torch=torch.get_rng_state().clone(), python=random.getstate(), numpy=np.random.get_state(),
                cuda=torch.cuda.get_rng_state(torch.device(device)).clone() if str(device).startswith('cuda') else None)


def _assert_rng(before, device):
    after = _rng_state(device)
    if not torch.equal(before['torch'], after['torch']) or before['python'] != after['python']:
        raise ValueError('Source probe diagnostic changed CPU/Python RNG')
    if (before['numpy'][0] != after['numpy'][0] or not np.array_equal(before['numpy'][1], after['numpy'][1])
            or before['numpy'][2:] != after['numpy'][2:]):
        raise ValueError('Source probe diagnostic changed NumPy RNG')
    if before['cuda'] is not None and not torch.equal(before['cuda'], after['cuda']):
        raise ValueError('Source probe diagnostic changed selected CUDA RNG')


def preflight(output, device):
    from experiments.cvs_equivariant_identity.precision import numerical_context
    output = Path(output)
    if output.exists():
        raise FileExistsError('Existing no-query smoke output; reconcile before retry')
    before_rng = _rng_state(device)
    with preserve_rng(device), numerical_context(d.FULL_FP32_POLICY):
        models, contracts = load_models(device); records = []
        for model, contract in zip(models, contracts):
            before = snapshot(model)
            with torch.no_grad(), capture(model) as maps:
                values = model.forward_identity_only(torch.zeros(2, 2, 256, device=device))
            logits = values['tx_logits']; features = descriptors(maps, values['id_feat_joint'])
            if logits.shape != (2, d.CLASSES) or not torch.isfinite(logits).all() or any(
                    features[k].shape != (2, dim) for k, dim in zip(d.FEATURES, d.DIMENSIONS)):
                raise ValueError('Actual frozen checkpoint no-query smoke failed')
            records.append(dict(row_id=contract['config']['row_id'], **assert_unchanged(model, before),
                                checkpoint=contract['checkpoint'], logits_shape=list(logits.shape),
                                features={k: list(v.shape) for k, v in features.items()}))
    _assert_rng(before_rng, device)
    result = dict(status='PASS', run_id=d.RUN, real_checkpoints=9, rows=records,
                  no_query=True, source_IQ_opened=False, U_cache_opened=False,
                  target_query_access=False, U_truth_read=False, identity_updates=0, RNG_preserved=True)
    _write_new(output, result)
    print('SOURCE_CONDITIONAL_RX_PROBE_NO_QUERY_SMOKE_PASS checkpoints=9', flush=True)
    return result


def run(output, device='cpu', registration=True):
    from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
    if registration:
        verify_registration()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); torch.set_num_threads(2); before_rng = _rng_state(device)
    with preserve_rng(device), numerical_context(d.FULL_FP32_POLICY):
        models, contracts = load_models(device); frozen_before = [snapshot(model) for model in models]
        roles, source_contract = load_roles()
        if any(contract['source_contract'] != source_contract for contract in contracts):
            raise ValueError('Frozen models do not share the loaded source role contract')
        resolved = dict(run_id=d.RUN, pid=os.getpid(), cwd=os.getcwd(), python=sys.executable,
                        commit=(d.ROOT / 'release_commit.txt').read_text().strip(), device=device,
                        backend_flags=actual_flags(), rows=9, source_roles=dict(d.ROLE_COUNTS),
                        source_packets=9 * sum(d.ROLE_COUNTS.values()), identity_updates=0,
                        V_updates=0, target_query_access=False, U_cache_opened=False,
                        U_truth_read=False, source_performance_gate=False,
                        probe=dict(d.PROBE), probe_count=189, probe_optimizer_updates=18900,
                        probe_diagnostic_confusions=567, auxiliary_state_loaded=False)
        _write_new(output / 'resolved_config.json', resolved)
        _write_new(output / 'source_contract.json', source_contract)
        all_probabilities = {}; all_features = {}
        for role, cache in roles.items():
            all_probabilities[role], all_features[role] = scan(models, cache, role, device)
            print(f'SOURCE_ROLE_COMPLETE role={role} models=9 count={len(cache["ids"])}', flush=True)
        folds, partition_info = partition(roles['L_s']); completed = []; records = []
        for i, (model, contract) in enumerate(zip(models, contracts)):
            row_id = contract['config']['row_id']; row_output = output / row_id
            row_output.mkdir(exist_ok=False); raw_output = row_output / 'source'; raw_output.mkdir(exist_ok=False)
            raw = {role: dict(probabilities=all_probabilities[role][i], features=all_features[role][i],
                              y=cache['y'], rx=cache['rx'], day=cache['day'], ids=list(cache['ids']))
                   for role, cache in roles.items()}
            raw.update(folds=folds, partition=partition_info, raw_IQ_persisted=False,
                       identity_state_persisted=False, target_access=False, U_cache_opened=False)
            with (raw_output / 'raw_source.pt').open('xb') as stream:
                torch.save(raw, stream)
            summaries = {role: source_summary(all_probabilities[role][i], cache, role)
                         for role, cache in roles.items()}
            probes = run_probes(all_features['L_s'][i], all_features['V'][i], roles['L_s'], roles['V'],
                               contract['config']['model_seed'], device)
            frozen = assert_unchanged(model, frozen_before[i])
            _write_new(row_output / 'model_contract.json', contract)
            _write_new(row_output / 'source_summary.json', dict(row_id=row_id, roles=summaries, **frozen,
                        target_query_access=False, U_cache_opened=False, U_truth_read=False, V_updates=0,
                        source_performance_gate=False, ordinary_cosine_head=True))
            _write_new(row_output / 'rx_probes.json', probes)
            row_done = dict(status='SOURCE_PROBE_HEALTH_COMPLETE', row_id=row_id, roles=dict(d.ROLE_COUNTS),
                            source_packets=sum(d.ROLE_COUNTS.values()), **frozen, probe_count=21,
                            probe_optimizer_updates=2100, probe_diagnostic_confusions=63,
                            raw_source_artifact=(raw_output / 'raw_source.pt').as_posix(),
                            target_query_access=False, U_cache_opened=False, U_truth_read=False, V_updates=0)
            _write_new(row_output / 'completion.json', row_done); completed.append(row_done)
            records.append(dict(row_id=row_id, source_summary=summaries, rx_probes=probes))
            print(f'SOURCE_CONDITIONAL_RX_PROBE_ROW_COMPLETE {row_id}', flush=True)
        for model, before in zip(models, frozen_before):
            assert_unchanged(model, before)
        report = analysis(records); _write_new(output / 'analysis.json', report)
    _assert_rng(before_rng, device)
    done = dict(status='SOURCE_PROBE_HEALTH_COMPLETE', run_id=d.RUN, rows=9, completed=completed,
                source_packets=9 * sum(d.ROLE_COUNTS.values()), source_role_packets=dict(d.ROLE_COUNTS),
                identity_updates=0, target_query_access=False, U_cache_opened=False, U_truth_read=False,
                V_updates=0, source_performance_gate=False, RNG_preserved=True,
                parameters_buffers_exactly_unchanged=True, modes_unchanged=True,
                identity_gradient_count=0, probe_count=189, probe_optimizer_updates=18900,
                probe_diagnostic_confusions=567, elapsed_seconds=time.perf_counter() - started)
    _write_new(output / 'completion.json', done)
    print('SOURCE_CONDITIONAL_RX_PROBE_COMPLETE rows=9 source_packets=299700 identity_updates=0', flush=True)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True)
    parser.add_argument('--preflight', '--remote-preflight', dest='preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight(args.output, args.device)
    else:
        run(args.output, args.device)


if __name__ == '__main__':
    main()
