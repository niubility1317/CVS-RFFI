"""Frozen source-only health CLI. No target reader, predictor or scorer is called."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from . import design as d
from .frozen import preserve_rng, snapshot, assert_unchanged
from .stats import summarize, validate_role
from .probes import capture, descriptors, run_probes


def _write_new(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify_registration():
    spec = d.read(d.ROOT / 'registration' / 'experiment.json')
    if spec.get('run_id') != d.RUN:
        raise ValueError('Registration run ID differs')
    matrix = spec.get('rows')
    if not isinstance(matrix, list) or len(matrix) != len(d.rows()):
        raise ValueError('Registration must contain exactly three fixed rows')
    expected = {row['row_id']: d.config(row) for row in d.rows()}
    for row in matrix:
        # Existing registry specs expose configs directly under each row.
        if row.get('row_id') not in expected or row.get('config') != expected[row['row_id']]:
            raise ValueError('Registration source health row/config differs')
    if len({row['row_id'] for row in matrix}) != len(expected):
        raise ValueError('Registration row IDs repeat')
    return spec


def load_models(device):
    # source_proof and own_model do not call their module's target prediction helpers.
    from experiments.cvs_repeated_rx_margin.predict import source_proof, own_model
    models = []; contracts = []
    for row in d.rows():
        c = d.source_config(row); source = source_proof(c)
        models.append(own_model(c, 'FINAL', device))
        contracts.append(dict(config=d.config(row), source_config=c,
                              source_contract=d.read(source / 'source_contract.json'),
                              source_frozen=d.read(source / 'source_frozen.json'),
                              actual_training=d.read(source / 'completion.json'),
                              actual_resolved=d.read(source / 'resolved_config.json'),
                              checkpoint=(source / 'FINAL.pth').as_posix(),
                              checkpoint_contract='VERIFIED matching physical source roles; scratch FINAL30',
                              source_performance_gate=False, target_access=False,
                              U_truth_read=False, reused_for='diagnostic only; identity never updated'))
    return models, contracts


def load_roles():
    from experiments.cvs_nocfo_foundation.source import load_source
    from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled
    c = d.source_config(d.rows()[0])
    L, unused_U, V, contract = load_source(c['source_loader_config'])
    if unused_U is not None:
        raise ValueError('Expected source loader to leave hidden U unopened')
    U = torch.load(c['U_cache'], map_location='cpu', weights_only=False)
    validate_unlabeled(U, contract['role_ids']['U_s'])
    roles = {'L_s': L, 'V': V, 'U_s': U}
    for role, cache in roles.items():
        count = validate_role(cache, role, contract['role_ids'][role])
        if count != d.ROLE_COUNTS[role]:
            raise ValueError('Fixed source role count differs')
    sets = [set(cache['ids']) for cache in roles.values()]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise ValueError('Source roles overlap')
    return roles, contract


@torch.no_grad()
def scan(models, cache, role, device):
    probabilities = [[] for _ in models]
    features = [{name: [] for name in d.FEATURES} for _ in models] if role != 'U_s' else None
    for begin in range(0, len(cache['ids']), d.BATCH_SIZE):
        x = cache['x'][begin:begin + d.BATCH_SIZE].to(device)
        for i, model in enumerate(models):
            if features is not None:
                with capture(model) as maps:
                    values = model.forward_identity_only(x)
                current = descriptors(maps, values['id_feat_joint'])
                for name, tensor in current.items():
                    features[i][name].append(tensor)
                logits = values['tx_logits']
            else:
                logits = model(x)
            if logits.shape != (len(x), d.CLASSES) or not torch.isfinite(logits).all():
                raise FloatingPointError('Source frozen model logits differ')
            probabilities[i].append(logits.softmax(1).cpu())
        if begin % 16384 == 0:
            print(f'SOURCE_SCAN role={role} inputs_per_model={min(begin + len(x), len(cache["ids"]))}', flush=True)
    return torch.stack([torch.cat(parts) for parts in probabilities]), (
        [{name: torch.cat(parts) for name, parts in value.items()} for value in features]
        if features is not None else None)


def preflight(output, device):
    from experiments.cvs_equivariant_identity.precision import numerical_context
    output = Path(output)
    if output.exists():
        raise FileExistsError('Existing no-query smoke output; reconcile before retry')
    with preserve_rng(device), numerical_context(d.FULL_FP32_POLICY):
        models, contracts = load_models(device)
        records = []
        for model, contract in zip(models, contracts):
            before = snapshot(model)
            with torch.no_grad():
                with capture(model) as maps:
                    values = model.forward_identity_only(torch.zeros(2, 2, 256, device=device))
                logits = values['tx_logits']; features = descriptors(maps, values['id_feat_joint'])
            if logits.shape != (2, d.CLASSES) or not torch.isfinite(logits).all():
                raise ValueError('Actual frozen checkpoint no-query smoke failed')
            record = assert_unchanged(model, before)
            records.append(dict(row_id=contract['config']['row_id'], **record,
                                source_checkpoint=contract['checkpoint'], shape=list(logits.shape),
                                features={k: list(v.shape) for k, v in features.items()}))
    result = dict(status='PASS', run_id=d.RUN, real_checkpoints=3, no_query=True,
                  source_IQ_opened=False, target_query_access=False, U_truth_read=False,
                  rows=records, identity_updates=0)
    _write_new(output, result)
    print('SOURCE_HEALTH_NO_QUERY_SMOKE_PASS checkpoints=3', flush=True)
    return result


def run(output, device, registration=True):
    from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
    if registration:
        verify_registration()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); torch.set_num_threads(2)
    with preserve_rng(device), numerical_context(d.FULL_FP32_POLICY):
        models, contracts = load_models(device)
        before = [snapshot(model) for model in models]
        roles, source_contract = load_roles()
        if any(contract['source_contract'] != source_contract for contract in contracts):
            raise ValueError('Frozen models do not share the actual loaded source roles')
        resolved = dict(run_id=d.RUN, pid=os.getpid(), cwd=os.getcwd(), python=sys.executable,
                        commit=(d.ROOT / 'release_commit.txt').read_text().strip(),
                        device=device, backend_flags=actual_flags(), rows=3,
                        source_packets=3 * sum(d.ROLE_COUNTS.values()), identity_updates=0,
                        target_query_access=False, U_truth_read=False, V_updates=0,
                        source_performance_gate=False, teacher='FINAL30 weak eval current frozen student',
                        peer='fixed other-two mean T1 probability', threshold=d.THRESHOLD,
                        probe=dict(d.PROBE), source_roles=dict(d.ROLE_COUNTS))
        _write_new(output / 'resolved_config.json', resolved)
        _write_new(output / 'source_contract.json', source_contract)
        all_summaries = [dict() for _ in models]; all_features = {}
        for role, cache in roles.items():
            q, features = scan(models, cache, role, device)
            if features is not None:
                all_features[role] = features
            for i in range(len(models)):
                other = q[[j for j in range(len(models)) if j != i]]
                y = None if role == 'U_s' else cache['y']
                all_summaries[i][role] = summarize(q[i], other, cache['rx'], cache['day'], role, y)
            print(f'SOURCE_ROLE_COMPLETE role={role} models=3 count={len(cache["ids"])}', flush=True)
        completed = []
        for i, (model, contract, summaries) in enumerate(zip(models, contracts, all_summaries)):
            row_id = contract['config']['row_id']; row_output = output / row_id
            row_output.mkdir(exist_ok=False)
            probe = run_probes(all_features['L_s'][i], all_features['V'][i], roles['L_s'], roles['V'],
                               contract['config']['model_seed'], device)
            frozen = assert_unchanged(model, before[i])
            _write_new(row_output / 'model_contract.json', contract)
            _write_new(row_output / 'source_summary.json', dict(row_id=row_id, roles=summaries, **frozen,
                        teacher='frozen FINAL30 weak eval; no historical EMA reconstructed',
                        target_query_access=False, U_truth_read=False, V_updates=0))
            _write_new(row_output / 'rx_probes.json', probe)
            row_done = dict(status='SOURCE_HEALTH_COMPLETE', row_id=row_id, roles=dict(d.ROLE_COUNTS),
                            source_packets=sum(d.ROLE_COUNTS.values()), **frozen,
                            target_query_access=False, U_truth_read=False, V_updates=0)
            _write_new(row_output / 'completion.json', row_done)
            completed.append(row_done)
            print(f'SOURCE_HEALTH_ROW_COMPLETE {row_id}', flush=True)
    done = dict(status='SOURCE_HEALTH_COMPLETE', run_id=d.RUN, rows=3, completed=completed,
                source_packets=3 * sum(d.ROLE_COUNTS.values()), source_role_packets=dict(d.ROLE_COUNTS),
                identity_updates=0, target_query_access=False, U_truth_read=False, V_updates=0,
                source_performance_gate=False, RNG_preserved=True,
                elapsed_seconds=time.perf_counter() - started)
    _write_new(output / 'completion.json', done)
    print('SOURCE_HEALTH_COMPLETE rows=3 source_packets=270000 identity_updates=0', flush=True)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True)
    parser.add_argument('--remote-preflight', action='store_true')
    args = parser.parse_args()
    if args.remote_preflight:
        preflight(args.output, args.device)
    else:
        run(args.output, args.device)


if __name__ == '__main__':
    main()
