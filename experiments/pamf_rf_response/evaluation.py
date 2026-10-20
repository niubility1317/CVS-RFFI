import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .models import directions, oracle_response
from .runtime import source_inputs, load_model
from experiments.pamf_restoration.data import SCENES, write_json
from experiments.pamf_restoration.runner import array_tensor, tensor_array, read_json
from experiments.pamf_action_restore.evaluation import teacher_action
from experiments.pamf_restoration.models import complex_iq, pair

VIEWS = ('clean', *SCENES)
METRICS = ('gain', 'response_nmse', 'response_cosine', 'response_energy_ratio',
           'curvature_nmse', 'oracle_nmse', 'oracle_gain', 'fisher_energy', 'oracle_CG_relative_residual')


def row_name(name, amplitude):
    return f'{name}_r{int(amplitude*100):02d}'


def complete_contract(value, count):
    required = dict(status='SOURCE_COUNTERFACTUALS_COMPLETE', count=count, views=list(VIEWS),
        rows=list(d.ROWS), cells=70, parent=d.PARENT, parent_commit=d.PARENT_COMMIT,
        candidate=d.CANDIDATE, TX_truth_reads=0, F_access=0, target_reads=0,
        optimizer_updates=0, source_V_state_updates=0, oracle_not_deployable=True)
    if any(value.get(key) != item for key, item in required.items()):
        raise ValueError('All fixed source counterfactuals must precede truth grouping')


@torch.no_grad()
def predict(args):
    folder, ids, _, _ = source_inputs(args)
    model, _ = load_model(args, 'cuda:0')
    clean = np.load(folder/'clean.npy', mmap_mode='r', allow_pickle=False)
    if clean.shape != (1800, 2, 256):
        raise ValueError('Original complete source target panel required')
    output = args.output/'prediction'
    output.mkdir()
    write_json(args.output/'gpu_ready.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
                                               GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    print('SOURCE_RF_RESPONSE_GPU_READY', flush=True)
    started_all = time.perf_counter()
    timings = {}
    for view in VIEWS:
        observed = np.load(folder/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        known = args.parent_action/'cache/validation_actions'/view
        metadata = read_json(known/'complete.json')
        if (metadata['count'] != len(ids) or metadata['namespace'] != 'restoration/validation/draw0'
                or metadata['TX_labels_read'] or metadata['F_outputs_read'] or metadata['target_reads']
                or metadata['native_engine_modified']):
            raise ValueError('Original unchanged source action handles required')
        arrays = {key: np.load(known/(key+'.npy'), mmap_mode='r', allow_pickle=False)
                  for key in ('kernel', 'coefficients', 'variance')}
        if (observed.shape != clean.shape or arrays['kernel'].shape != (1800, 2, 64)
                or arrays['coefficients'].shape != (1800, 4, 256) or arrays['variance'].shape != (1800,)):
            raise ValueError('Complete matched source action and observation required')
        view_root = output/view
        view_root.mkdir()
        base_array = np.lib.format.open_memmap(view_root/'base.npy', mode='w+', dtype=np.float32, shape=clean.shape)
        outputs = {}
        for name in d.ROWS:
            root = view_root/name
            root.mkdir()
            outputs[name] = {key: np.lib.format.open_memmap(root/(key+'.npy'), mode='w+', dtype=np.float32, shape=shape)
                for key, shape in [('plus', clean.shape), ('minus', clean.shape), ('oracle', clean.shape),
                                   ('observation_delta', clean.shape), ('CG_residual', (1800,))]}
        started = time.perf_counter()
        for begin in range(0, len(ids), 128):
            index = np.arange(begin, min(begin+128, len(ids)))
            x = array_tensor(clean[index], 'cuda:0')
            r = array_tensor(observed[index], 'cuda:0')
            action = teacher_action(arrays, index, 'cuda:0')
            variance = array_tensor(arrays['variance'][index], 'cuda:0')
            base = model(r)
            if not torch.isfinite(base).all():
                raise FloatingPointError('Nonfinite original restorer output')
            base_array[index] = tensor_array(base)
            for amplitude in d.AMPLITUDES:
                for name, (delta, _) in directions(x, index, amplitude).items():
                    row = row_name(name, amplitude)
                    change = action(delta)
                    plus, minus = model(r+pair(change)), model(r-pair(change))
                    oracle, residual = oracle_response(action, change, variance)
                    for key, value in [('plus', plus), ('minus', minus), ('oracle', pair(oracle)),
                                       ('observation_delta', pair(change)), ('CG_residual', residual)]:
                        if not torch.isfinite(value).all():
                            raise FloatingPointError('Nonfinite source response;preserve partial outputs')
                        outputs[row][key][index] = tensor_array(value)
        base_array.flush()
        for item in outputs.values():
            for array in item.values():
                array.flush()
        torch.cuda.synchronize()
        timings[view] = time.perf_counter()-started
        print(json.dumps(dict(event='SOURCE_RF_RESPONSE_VIEW_COMPLETE', view=view, count=len(ids),
                              rows=list(d.ROWS), seconds=timings[view])), flush=True)
    np.save(output/'physical_ids.npy', ids, allow_pickle=False)
    complete = dict(status='SOURCE_COUNTERFACTUALS_COMPLETE', count=len(ids), views=list(VIEWS),
        rows=list(d.ROWS), cells=70, parent=d.PARENT, parent_commit=d.PARENT_COMMIT, candidate=d.CANDIDATE,
        TX_truth_reads=0, F_access=0, target_reads=0, optimizer_updates=0, source_V_state_updates=0,
        oracle_not_deployable=True, per_view_seconds=timings, seconds=time.perf_counter()-started_all)
    complete_contract(complete, len(ids))
    write_json(output/'complete.json', complete)


def load_predictions(path, ids):
    path = Path(path)
    complete_contract(read_json(path/'complete.json'), len(ids))
    if not np.array_equal(np.load(path/'physical_ids.npy', allow_pickle=False), ids):
        raise ValueError('Source response IDs differ;truth stays closed')
    result = {}
    for view in VIEWS:
        base = np.load(path/view/'base.npy', mmap_mode='r', allow_pickle=False)
        if base.shape != (len(ids), 2, 256) or not np.isfinite(base).all():
            raise ValueError('Incomplete base output;truth stays closed')
        result[view] = dict(base=base, rows={})
        for name in d.ROWS:
            row = {}
            for key in ('plus', 'minus', 'oracle', 'observation_delta', 'CG_residual'):
                value = np.load(path/view/name/(key+'.npy'), mmap_mode='r', allow_pickle=False)
                shape = (len(ids),) if key == 'CG_residual' else (len(ids), 2, 256)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('Incomplete/nonfinite source response;truth stays closed')
                row[key] = value
            result[view]['rows'][name] = row
    return result


def numpy_metrics(arrays, base, delta, valid, variance):
    # Independent CPU float64 arithmetic from saved raw counterfactual IQ.
    delta = np.asarray(delta, dtype=np.float64)
    plus, minus = np.asarray(arrays['plus'], dtype=np.float64), np.asarray(arrays['minus'], dtype=np.float64)
    response = (plus-minus)/2
    curvature = (plus+minus)/2-np.asarray(base, dtype=np.float64)
    oracle = np.asarray(arrays['oracle'], dtype=np.float64)
    obs_delta = np.asarray(arrays['observation_delta'], dtype=np.float64)
    dot = lambda a, b: (a*b).sum((1, 2))
    energy = np.maximum(dot(delta, delta), 1e-20)
    response_energy = dot(response, response)
    oracle_nmse = dot(oracle-delta, oracle-delta)/energy
    fisher = dot(obs_delta, obs_delta)/variance
    cg = np.asarray(arrays['CG_residual'], dtype=np.float64)
    observable = valid & (fisher >= 9) & (oracle_nmse <= .1) & (cg <= .001)
    result = dict(valid=valid, observable=observable, gain=dot(response, delta)/energy,
        response_nmse=dot(response-delta, response-delta)/energy,
        response_cosine=dot(response, delta)/np.sqrt(np.maximum(response_energy*energy, 1e-30)),
        response_energy_ratio=response_energy/energy, curvature_nmse=dot(curvature, curvature)/energy,
        oracle_nmse=oracle_nmse, oracle_gain=dot(oracle, delta)/energy,
        fisher_energy=fisher, oracle_CG_relative_residual=cg)
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError('Invalid response metrics')
    return result


def aggregate(values, mask):
    valid = mask & values['valid']
    observable = mask & values['observable']
    output = dict(count=int(mask.sum()), valid_count=int(valid.sum()), observable_count=int(observable.sum()))
    for prefix, selected in (('valid', valid), ('observable', observable)):
        for key in METRICS:
            data = values[key][selected]
            output[f'{prefix}_{key}_mean'] = float(data.mean()) if len(data) else None
            output[f'{prefix}_{key}_median'] = float(np.median(data)) if len(data) else None
    return output


def interpret(records):
    criteria = d.configuration()['source_rule']
    supported = []
    comparisons = {}
    for direction in d.DIRECTIONS:
        name = row_name(direction, .01)
        rows = [r for r in records if r['view'] == criteria['primary_view'] and r['row'] == name
                and r['dimension'] == 'overall' and r['value'] == 'all']
        if len(rows) != 1:
            raise ValueError('Exactly one complete primary source summary required')
        value = rows[0]
        passed = (value['observable_count'] >= criteria['minimum_observable_records']
            and value['observable_response_nmse_median'] >= criteria['median_response_NMSE_min']
            and value['observable_gain_median'] <= criteria['median_gain_max']
            and value['observable_curvature_nmse_median'] <= criteria['median_curvature_NMSE_max'])
        comparisons[direction] = dict(source_attenuation=bool(passed), **value)
        if passed and direction != 'random_control':
            supported.append(direction)
    return dict(status='SOURCE_RESPONSE_DIAGNOSTIC_COMPLETE',
        supports_RF_response_training=len(supported) >= criteria['minimum_supported_directions'],
        supported_directions=supported, comparisons=comparisons, thresholds=criteria,
        target_promotion=False, training_effect_unproven=True, real_TX_fidelity_unproven=True, goal_complete=False)


def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU source grouping required')
    folder, ids, source, index = source_inputs(args)
    predictions = load_predictions(args.output/'prediction', ids)
    # Source truth is opened only after all 70 counterfactual cells are complete.
    labels = np.load(args.source_cache/'V/labels.npy', allow_pickle=False)[index]
    if set(np.unique(labels).tolist()) != set(range(6)):
        raise ValueError('Original six-TX source grouping required')
    clean = np.load(folder/'clean.npy', mmap_mode='r', allow_pickle=False)
    groups = [('overall', 'all', np.ones(len(ids), bool))]
    for dimension, values in [('TX', labels), ('RX', source.rx[index]), ('day', source.day[index])]:
        groups += [(dimension, str(value), values == value) for value in np.unique(values)]
    records = []
    clean_tensor = array_tensor(clean, 'cpu')
    for view in VIEWS:
        variance = np.load(args.parent_action/'cache/validation_actions'/view/'variance.npy', allow_pickle=False)
        if variance.shape != (len(ids),) or not np.isfinite(variance).all() or (variance <= 0).any():
            raise ValueError('Positive matched source simulator variances required')
        for amplitude in d.AMPLITUDES:
            for direction, (delta, valid) in directions(clean_tensor, np.arange(len(ids)), amplitude).items():
                name = row_name(direction, amplitude)
                values = numpy_metrics(predictions[view]['rows'][name], predictions[view]['base'],
                    tensor_array(pair(delta)), tensor_array(valid).astype(bool), variance)
                np.savez(args.output/f'{view}.{name}.metrics.npz', **values)
                for dimension, value, mask in groups:
                    records.append(dict(view=view, row=name, direction=direction, amplitude=amplitude,
                        dimension=dimension, value=value, **aggregate(values, mask)))
    summary = dict(status='ANALYZED', role='source', count=len(ids), cells=70, records=records,
        configuration=d.configuration(), all_counterfactuals_before_truth=True, source_V_state_updates=0,
        optimizer_updates=0, target_reads=0, F_counterfactual_reads=0, epochs=0,
        oracle_not_deployable=True, goal_complete=False)
    write_json(args.output/'summary.json', summary)
    write_json(args.output/'interpretation.json', interpret(records))
