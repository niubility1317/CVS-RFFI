"""Frozen source attribution; only L_s defines the diagnostic coefficient mean."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

RUN = '20261003-diagnostic-cvs-frontfilter-attribution-source-manysig-m8-r01'
RELEASE = 'cvs_frontfilter_attribution_source_20261003_r01'
SOURCE_RUN = '20261003-phase1-cvs-frontfilter-identity-manysig-m8-r01'
SOURCE_RELEASE = 'cvs_frontfilter_identity_20261003_r01'
SOURCE_COMMIT = 'dd5518d1493f2f79fb4213e4731e9f88cb28a349'
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
OWNER = 'codex/root/frontfilter-source-attribution-20261003'
VARIANTS = ('frontfilter_static', 'frontfilter_dynamic')
CONDITIONS = ('all_on', 'mean_L', 'g_identity')
STATIC_ATOL = 1e-5
STATIC_RTOL = 1e-5


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def conditions(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered variant')
    return CONDITIONS


def validate_config(c):
    keys = {'source_release', 'source_commit', 'source_output', 'variant',
            'model_seed', 'output_root', 'conditions', 'role', 'coefficient_role',
            'launch_owner'}
    if set(c) != keys or c['role'] != 'V' or c['coefficient_role'] != 'L_s':
        raise ValueError('Only fixed source L_s coefficient mean and V evaluation allowed')
    if c['conditions'] != list(conditions(c['variant'])) or c['launch_owner'] != OWNER:
        raise ValueError('Unregistered conditions or launch owner')
    if (c['source_commit'] != SOURCE_COMMIT or type(c['model_seed']) is not int
            or c['model_seed'] not in range(2026092701, 2026092705)):
        raise ValueError('Unexpected frozen source checkpoint')
    rid = c['variant'] + '-s' + str(c['model_seed'])
    if (c['source_release'] != PROJECT + '/releases/' + SOURCE_RELEASE
            or c['source_output'] != PROJECT + '/runs/' + SOURCE_RUN + '/' + rid + '/source'
            or c['output_root'] != PROJECT + '/runs/' + RUN + '/' + rid):
        raise ValueError('Unexpected source or exclusive diagnostic path')
    return c


def validate_payload(c, resolved, contract, initial, payload):
    """Extra checkpoint checks after the source family's metadata-only audit."""
    from experiments.cvs_frontfilter_identity.model import filter_contract
    from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
    expected_contract = filter_contract(c['variant'])
    expected = dict(epoch=200, selection='fixed_last_epoch', method='cvs_frontfilter_identity',
                    variant=c['variant'], config=resolved, source_contract=contract,
                    initialization=initial, classes=contract['classes'], num_classes=6)
    if any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError('Checkpoint payload provenance mismatch')
    if (resolved.get('commit') != SOURCE_COMMIT
            or resolved.get('frontfilter') != expected_contract
            or resolved.get('frontfilter_actual') != expected_contract
            or resolved.get('frontfilter_active') is not True
            or resolved.get('precision') != 'float32'
            or resolved.get('numerical_policy') != FULL_FP32_POLICY
            or resolved.get('backend_flags') != FULL_FP32_POLICY
            or resolved.get('total_parameters') != expected_contract['total_parameters']):
        raise ValueError('Actual frozen source commit, frontend or FP32 contract mismatch')
    scratch = dict(status='SCRATCH', scratch_only=True, checkpoint=None, ancestors=[],
                   checkpoint_sources=[], target_access=False, target_contact=False,
                   model_seed=c['model_seed'], physical_roles='EXACT_MATCH',
                   selection='fixed_last_epoch')
    if initial != scratch:
        raise ValueError('Source initialization is not own uncontaminated scratch')
    state = payload.get('model')
    if not isinstance(state, dict) or not state:
        raise ValueError('Missing checkpoint model state')
    for value in state.values():
        if not isinstance(value, torch.Tensor) or value.is_complex() or (value.is_floating_point()
                and (value.dtype != torch.float32 or not torch.isfinite(value).all())):
            raise ValueError('Checkpoint model state is not finite FP32')


def checked_role_index(dataset, expected_ids, role, count):
    """Check all physical IDs from dataset indices before touching any packet."""
    from baselines.common.practical_source import physical_id
    if (dataset.role != role or len(dataset) != count or len(expected_ids) != count
            or len(set(expected_ids)) != count):
        raise ValueError('Source role count/identity mismatch')
    index = {}
    for position in dataset.indices:
        item = dataset.base.index[position]
        sid = physical_id(item)
        if sid in index:
            raise ValueError('Repeated physical source ID')
        index[sid] = (int(item.tx_i), int(item.rx_i), int(item.day_i))
    if set(index) != set(expected_ids):
        raise ValueError('Physical source role mismatch')
    return index


def batch_metadata(batch, index, *, with_truth):
    ids = [m['sample_id'] for m in batch['meta']]
    receiver = batch['receiver'].tolist()
    day = batch['day'].tolist()
    truth = batch['label'].tolist() if with_truth else None
    if len(ids) != len(receiver) or len(ids) != len(day) or len(ids) != len(set(ids)):
        raise ValueError('Inconsistent source batch metadata')
    for j, (sid, meta) in enumerate(zip(ids, batch['meta'])):
        if (sid not in index or (receiver[j], day[j]) != index[sid][1:]
                or meta.get('rx_i') != receiver[j] or meta.get('day_i') != day[j]
                or (with_truth and truth[j] != index[sid][0])):
            raise ValueError('Source physical TX/RX/day metadata mismatch')
    return ids, receiver, day, truth


@torch.no_grad()
def coefficient_mean(model, loader, index, device):
    """Read IQ and physical metadata only; no label or V access."""
    ids, values = [], []
    for batch in loader:
        batch_ids, _, _, _ = batch_metadata(batch, index, with_truth=False)
        coefficients = model.frontfilter.coefficients(batch['iq'].to(device))
        if (coefficients.shape != (len(batch_ids), 2, 4) or coefficients.dtype != torch.float32
                or not torch.isfinite(coefficients).all()
                or (torch.linalg.vector_norm(coefficients, dim=1).sum(-1) > 1 + 1e-6).any()):
            raise ValueError('Invalid bounded source coefficients')
        ids.extend(batch_ids)
        values.extend(coefficients.cpu().tolist())
    if len(ids) != len(index) or len(set(ids)) != len(ids) or set(ids) != set(index):
        raise ValueError('Incomplete or repeated L_s coefficient traversal')
    array = np.asarray(values, dtype=np.float32)
    mean64 = array.mean(axis=0, dtype=np.float64)
    mean32 = mean64.astype(np.float32)
    return ids, array, mean64, mean32


@torch.no_grad()
def components(model, x, fixed_mean):
    """Apply each frontend once, then exactly the same trained backbone/head."""
    from experiments.cvs_neural_residual_identity.model import NeuralResidualCVS
    from experiments.cvs_channel_order_identity.model import packet_complex_fir
    if (x.dtype != torch.float32 or x.ndim != 3 or x.shape[1:] != (2, 256)
            or fixed_mean.shape != (2, 4) or fixed_mean.dtype != x.dtype
            or not torch.isfinite(x).all() or not torch.isfinite(fixed_mean).all()):
        raise ValueError('Expected finite FP32 IQ and frozen coefficient mean')
    gx, coefficients = model.frontfilter(x)
    fixed = fixed_mean[None].expand(len(x), -1, -1)
    mean_kernel = model.frontfilter.kernel(fixed)
    mean_x = x + model.frontfilter.rho * packet_complex_fir(x[:, :, None, :], mean_kernel)[:, :, 0]
    inputs = dict(all_on=gx, mean_L=mean_x, g_identity=x)
    features = {name: NeuralResidualCVS.features(model, value) for name, value in inputs.items()}
    logits = {name: model.classify_features(value) for name, value in features.items()}
    x_norm = x.flatten(1).norm(dim=1).clamp_min(1e-12)
    feature_norm = features['all_on'].norm(dim=1).clamp_min(1e-12)
    scalar = dict(
        g_input_relative=(gx-x).flatten(1).norm(dim=1)/x_norm,
        mean_L_input_relative=(mean_x-x).flatten(1).norm(dim=1)/x_norm,
        mean_L_all_on_input_relative=(mean_x-gx).flatten(1).norm(dim=1)/x_norm,
        coefficient_L_mean_distance=(coefficients-fixed).flatten(1).norm(dim=1),
        coefficient_l1=torch.linalg.vector_norm(coefficients, dim=1).sum(-1),
        kernel_l1=torch.linalg.vector_norm(model.frontfilter.kernel(coefficients), dim=1).sum(-1),
    )
    for name in ('mean_L', 'g_identity'):
        scalar[name+'_logit_distance'] = (logits[name]-logits['all_on']).norm(dim=1)
        scalar[name+'_logit_max_abs_difference'] = (logits[name]-logits['all_on']).abs().amax(1)
        scalar[name+'_feature_distance'] = (features[name]-features['all_on']).norm(dim=1)
        scalar[name+'_feature_relative'] = scalar[name+'_feature_distance']/feature_norm
    if any(not torch.isfinite(value).all() for value in (*logits.values(), *scalar.values())):
        raise ValueError('Nonfinite frozen diagnostic')
    return logits, scalar


def assert_state_unchanged(model, frozen):
    state = model.state_dict()
    if (state.keys() != frozen.keys() or any(not torch.equal(value, state[key])
                                           for key, value in frozen.items())):
        raise ValueError('Model parameters or buffers changed')


@torch.no_grad()
def no_source_smoke(model, device):
    probe = torch.zeros(2, 2, 256, device=device)
    probe[1, 0] = 1.
    scores, _ = components(model, probe, torch.zeros(2, 4, device=device))
    if not torch.allclose(scores['all_on'], model(probe), atol=1e-6, rtol=1e-6):
        raise ValueError('No-query full model smoke mismatch')


def classification_rows(logits, truth, receiver):
    """Float64 log-sum-exp CE and paired all_on comparisons on fixed logits."""
    truth, receiver = np.asarray(truth), np.asarray(receiver)
    if set(logits) != set(CONDITIONS) or not len(truth):
        raise ValueError('Incomplete source conditions')
    rows, predictions = [], {}
    for name in CONDITIONS:
        value = np.asarray(logits[name], dtype=np.float64)
        if (value.shape != (len(truth), 6) or not np.isfinite(value).all()
                or truth.shape != receiver.shape or truth.dtype.kind not in 'iu'
                or np.any((truth < 0) | (truth >= 6))):
            raise ValueError('Invalid source classification arrays')
        predictions[name] = value.argmax(1)
    all_correct = predictions['all_on'] == truth
    for name in CONDITIONS:
        value = np.asarray(logits[name], dtype=np.float64)
        shifted = value-value.max(axis=1, keepdims=True)
        ce = np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(truth)), truth]
        correct = predictions[name] == truth
        rates = {str(rx): float(correct[receiver == rx].mean()) for rx in sorted(set(receiver.tolist()))}
        rows.append(dict(condition=name, count=len(truth), accuracy=float(correct.mean()),
                         ce=float(ce.mean()), rx_accuracy=rates, worst_rx=min(rates.values()),
                         prediction_changes=int((predictions[name] != predictions['all_on']).sum()),
                         helped_by_all_on=int((all_correct & ~correct).sum()),
                         hurt_by_all_on=int((~all_correct & correct).sum())))
    return rows, predictions


def validate_v_coverage(ids, truth, receiver, day, index, cell_count=300):
    expected = {(tx, rx, d) for tx in range(6) for rx in (1, 3, 4, 6, 8) for d in (1, 2, 3)}
    if (len(ids) != len(index) or len(set(ids)) != len(ids) or set(ids) != set(index)
            or not (len(ids) == len(truth) == len(receiver) == len(day))):
        raise ValueError('Physical V coverage mismatch')
    cells = Counter(zip(truth, receiver, day))
    if set(cells) != expected or set(cells.values()) != {cell_count}:
        raise ValueError('Incomplete or unbalanced V TX/RX/day cells')


def execute(c):
    validate_config(c)
    release = Path(c['source_release'])
    if (release/'release_commit.txt').read_text(encoding='utf-8').strip() != SOURCE_COMMIT:
        raise ValueError('Frozen source release differs')
    sys.path[:0] = [str(release), str(release/'code')]
    from experiments.cvs_frontfilter_identity.dispatch import read_source_record
    from experiments.cvs_frontfilter_identity.model import build, filter_contract
    from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
    from experiments.cvs_identity_ce.source import source_args
    from baselines.common.practical_source import build_contract_split
    from baselines.common.cvs_data import make_cvs_loader
    source = Path(c['source_output'])
    resolved, contract, initial = [read(source/name) for name in
                                  ('resolved_config.json', 'source_contract.json', 'initialization.json')]
    record = read_source_record(c, read(resolved['source_contract']), 'cvs_frontfilter_identity')
    payload = torch.load(source/'last.pt', map_location='cpu', weights_only=False)
    validate_payload(c, resolved, contract, initial, payload)
    out = Path(c['output_root'])
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda:0')
    torch.set_num_threads(2)
    with numerical_context(resolved['numerical_policy']):
        model = build(c['variant']).to(device)
        model.load_state_dict(payload['model'], strict=True)
        if model.contract() != filter_contract(c['variant']):
            raise ValueError('Loaded actual frontend contract differs')
        model.eval()
        del payload
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        frozen = {key: value.detach().clone() for key, value in model.state_dict().items()}
        no_source_smoke(model, device)
        # The frozen checkpoint is checked before any source packet is loaded.
        split = build_contract_split(source_args(dict(resolved, output_root=str(out/'source_loader'))))
        if (split.test or split.named_tests or split.split_info['counts'] !=
                {'L_s': 6300, 'U_s': 56700, 'V': 27000}
                or read(out/'source_loader'/'source_contract.json') != contract):
            raise ValueError('Only exact original source split allowed')
        l_index = checked_role_index(split.train, contract['role_ids']['L_s'], 'L_s', 6300)
        v_index = checked_role_index(split.val, contract['role_ids']['V'], 'V', 27000)
        if set(l_index) & set(v_index):
            raise ValueError('Coefficient and evaluation roles overlap')
        write(out/'resolved_config.json', dict(c, pid=os.getpid(), cwd=os.getcwd(),
              python=sys.executable, hardware=torch.cuda.get_device_name(device),
              backend_flags=actual_flags(), source_record=record, target_access=False,
              optimizer_updates=0, source_selection_changed=False, coefficient_labels_used=False,
              ce_definition='float64 stable log-sum-exp of saved FP32 logits'))
        print('RESOLVED_CONFIG '+json.dumps(read(out/'resolved_config.json')), flush=True)
        tic = time.perf_counter()
        l_loader = make_cvs_loader(split.train, batch_size=256, shuffle=False,
                                  num_workers=0, device=device, drop_last=False)
        l_ids, coefficients, mean64, mean32 = coefficient_mean(model, l_loader, l_index, device)
        np.savez(out/'source_L_coefficients.npz', ids=np.asarray(l_ids), coefficients=coefficients)
        write(out/'mean_L_coefficients.json', dict(count=len(l_ids), role='L_s', ids=l_ids,
              mean=mean32.tolist(), mean_float64=mean64.tolist(), dtype='float32',
              accumulation_dtype='float64', frozen_before_V=True, used_labels=False,
              definition='Unweighted effective coefficient mean over all original L_s physical packets'))
        fixed_mean = torch.tensor(mean32.tolist(), dtype=torch.float32, device=device)
        del l_loader
        # No V loader or V forward exists until the L_s mean is fixed and saved.
        loader = make_cvs_loader(split.val, batch_size=256, shuffle=False,
                                num_workers=0, device=device, drop_last=False)
        ids, truth, receiver, day = [], [], [], []
        logits, scalar = {name: [] for name in CONDITIONS}, {}
        static_match = True
        with torch.no_grad():
            for batch in loader:
                batch_ids, rx, days, y = batch_metadata(batch, v_index, with_truth=True)
                scores, stats = components(model, batch['iq'].to(device), fixed_mean)
                if c['variant'] == 'frontfilter_static':
                    static_match &= bool(torch.allclose(scores['all_on'], scores['mean_L'],
                                                        atol=STATIC_ATOL, rtol=STATIC_RTOL))
                for name, value in scores.items():
                    logits[name].extend(value.cpu().tolist())
                for name, value in stats.items():
                    scalar.setdefault(name, []).extend(value.cpu().double().tolist())
                ids.extend(batch_ids)
                truth.extend(y)
                receiver.extend(rx)
                day.extend(days)
                if len(ids) % 4096 == 0:
                    print('PROGRESS '+json.dumps(dict(source_v_packets=len(ids))), flush=True)
        validate_v_coverage(ids, truth, receiver, day, v_index)
        logits = {name: np.asarray(value, dtype=np.float32) for name, value in logits.items()}
        rows, predictions = classification_rows(logits, truth, receiver)
        for row in rows:
            name = row['condition']
            np.savez(out/(name+'_source_predictions.npz'), ids=np.asarray(ids),
                     truth=np.asarray(truth), receiver=np.asarray(receiver), day=np.asarray(day),
                     predictions=predictions[name], logits=logits[name])
            print('CONDITION '+json.dumps(row), flush=True)
        np.savez(out/'source_scalar_diagnostics.npz', ids=np.asarray(ids),
                 receiver=np.asarray(receiver), day=np.asarray(day),
                 **{name: np.asarray(value, dtype=np.float64) for name, value in scalar.items()})
        if (abs(rows[0]['accuracy']-record['accuracy']) > 1e-12
                or abs(rows[0]['worst_rx']-record['worst_rx']) > 1e-12):
            raise ValueError('All-on does not reproduce E200 metrics')
        if not static_match:
            raise ValueError('Static mean_L failed measured FP32 output equivalence')
        assert_state_unchanged(model, frozen)
        summaries = {name: dict(mean=float(np.mean(value)), p10=float(np.quantile(value, .1)),
                     median=float(np.median(value)), p90=float(np.quantile(value, .9)),
                     maximum=float(np.max(value))) for name, value in scalar.items()}
        done = dict(status='SOURCE_ATTRIBUTION_COMPLETE', variant=c['variant'], model_seed=c['model_seed'],
                    rows=rows, scalar_summary=summaries, source_v_count=len(ids),
                    conditions=list(CONDITIONS), coefficient_role='L_s', coefficient_count=len(l_ids),
                    seconds=time.perf_counter()-tic, target_access=False, optimizer_updates=0,
                    source_selection_changed=False, model_unchanged=True,
                    static_mean_L_check=dict(applicable=c['variant'] == 'frontfilter_static',
                                            passed=static_match if c['variant'] == 'frontfilter_static' else None,
                                            atol=STATIC_ATOL, rtol=STATIC_RTOL,
                                            prediction_changes=rows[1]['prediction_changes']),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    claim='Frozen source-only diagnostic substitution. L_s mean uses no labels; V fits no state. '
                          'Off-distribution interventions are not retraining effects, channel recovery, '
                          'TX/RX disentanglement or target generalization; not for model selection.')
        write(out/'completion.json', done)
        print(json.dumps(done), flush=True)
    return done


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    execute(read(parser.parse_args().config))
