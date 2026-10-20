import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from . import design as d
from .runtime import load_checkpoint, frozen_context, validate_calibration
from .models import weighted_map
from experiments.pamf_residual_restore.runtime import load_checkpoint as load_residual
from experiments.pamf_fcrossrx_readout.frozen import load_reader, REFERENCE_RUN
from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference
from experiments.pamf_restoration.models import complex_iq, pair
from experiments.pamf_restoration.data import SCENES, write_json
from experiments.pamf_restoration.runner import array_tensor, tensor_array, read_json
from experiments.pamf_restoration.frozen import classification_statistics
from experiments.pamf_restoration.feature_diagnostic import capped_pair_margin

VIEWS = ('clean', *SCENES)
REFERENCES = ('identity', 'coarse', 'r03_reference', 'r07_native', 'global_floor')
FORMAL_F50 = '20261019-phase1-restoration-formal-FcrossRX50-s2026092701-r09'


def reference(args, role):
    if role == 'source':
        folder = args.parent_action/'cache/validation'
        ids, source, index = source_reference(folder, args.source_cache)
        return folder, ids, source, index
    if role != 'target':
        raise ValueError('Registered source/target role required')
    decision = read_json(args.output/'source_decision.json')
    if decision['status'] != 'SOURCE_DECISION_FROZEN' or not decision['qualified'] or args.kind not in decision['target_candidates']:
        raise ValueError('Frozen source qualification required before any target access')
    manifest = read_json(args.formal_reference/'manifest.json')
    if (manifest['status'] != 'FORMAL_REFERENCE_READY' or manifest['count'] != 168000 or manifest['views'] != list(VIEWS)
            or not manifest['original_IQ_bytes_reused'] or manifest['TX_truth_reads'] != 0):
        raise ValueError('Existing formal reference handles required')
    ids = np.load(args.formal_reference/'physical_ids.npy', allow_pickle=False)
    if ids.shape != (168000,):
        raise ValueError('Formal count differs')
    return args.formal_reference, ids, None, None


def row_folder(args, role):
    return args.output/args.kind if role == 'source' else args.output/'target_confirmation'/args.kind


@torch.no_grad()
def predict(args, role, *, checkpoint_loader=None, reference_loader=None, checkpoint_budget=(20, 4940)):
    source, ids, _, _ = (reference if reference_loader is None else reference_loader)(args, role)
    row = row_folder(args, role)
    if role == 'target':
        row.mkdir(parents=True)
    if args.kind == 'reference_controls':
        if role != 'source':
            raise ValueError('Reference controls are source only')
        physics, _ = frozen_context(args, 'global_floor', 'cuda:0')
        r03, payload = load_residual(args.parent_residual/'residual_deterministic/final.pt', 'residual_deterministic', 'cuda:0')
        if payload['epoch'] != 20 or payload['updates'] != 4940:
            raise ValueError('Original fixed R03 reference required')
        calibration = read_json(args.output/'calibration.json'); validate_calibration(calibration)
        floor = calibration['excess_variance']; names = REFERENCES
    else:
        loader = load_checkpoint if checkpoint_loader is None else checkpoint_loader
        model, payload = loader(args.output/args.kind/'final.pt', 'cuda:0')
        if (payload['epoch'], payload['updates']) != checkpoint_budget or payload['kind'] != args.kind:
            raise ValueError('Fixed final candidate epoch/update required')
        model.requires_grad_(False); names = ('restored',)
    reader = load_reader(args.reader_folder, args.source_cache, 'cuda:0').requires_grad_(False).eval()
    folder = row/'prediction'; folder.mkdir()
    timings = {}; started_all = time.perf_counter()
    for view in VIEWS:
        received = np.load(source/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        if received.shape != (len(ids), 2, 256):
            raise ValueError('Fixed original received view shape differs')
        arrays = {name: {key: np.lib.format.open_memmap(folder/(view+'.'+name+'.'+key+'.npy'), mode='w+',
                    dtype=np.float32, shape=shape) for key, shape in
                    (('iq', received.shape), ('logits', (len(ids), 6)), ('features', (len(ids), 160)))} for name in names}
        variance_array = None
        if args.kind in ('native_map_prior', 'uncertain_map_prior'):
            variance_array = np.lib.format.open_memmap(folder/(view+'.variance_mean.npy'), mode='w+', dtype=np.float32, shape=(len(ids),))
        started = time.perf_counter()
        for begin in range(0, len(ids), 128):
            end = min(begin+128, len(ids)); r = array_tensor(received[begin:end], 'cuda:0')
            if args.kind == 'reference_controls':
                base, action, native = physics.estimate(r)
                outputs = dict(identity=r, coarse=base, r03_reference=r03(r),
                    r07_native=pair(weighted_map(action, complex_iq(r), complex_iq(base), native)),
                    global_floor=pair(weighted_map(action, complex_iq(r), complex_iq(base), native[:, None]+floor)))
            else:
                restored, variance, _, _ = model.training_outputs(r)
                outputs = dict(restored=restored)
                if variance_array is not None:
                    variance_array[begin:end] = tensor_array(variance.mean(-1))
            for name, output in outputs.items():
                if not torch.isfinite(output).all():
                    raise FloatingPointError('Nonfinite restored current IQ')
                features = reader.features(output); logits = reader.classify_features(features)
                if features.shape != (len(r), 160) or logits.shape != (len(r), 6) or not torch.isfinite(logits).all() or not torch.isfinite(features).all():
                    raise ValueError('Invalid original frozen F50 output')
                for key, value in (('iq', output), ('features', features), ('logits', logits)):
                    arrays[name][key][begin:end] = tensor_array(value)
        for fields in arrays.values():
            for array in fields.values():
                array.flush()
        if variance_array is not None:
            variance_array.flush()
        torch.cuda.synchronize(); timings[view] = time.perf_counter()-started
        print(json.dumps(dict(event='UNCERTAINTY_VIEW_COMPLETE', kind=args.kind, role=role, view=view, count=len(ids))), flush=True)
    np.save(folder/'physical_ids.npy', ids, allow_pickle=False)
    write_json(folder/'complete.json', dict(status='PREDICTIONS_COMPLETE', kind=args.kind, role=role, count=len(ids),
        views=list(VIEWS), readouts=list(names), TX_truth_reads=0, state_updates=0, head_fit_updates=0,
        F_run=REFERENCE_RUN, F_seed=2026092701, per_view_seconds=timings, seconds=time.perf_counter()-started_all))
    if role == 'source' and args.kind != 'reference_controls':
        values = array_tensor(np.load(source/'practical_low_urban.npy', allow_pickle=False)[:16], 'cuda:0')
        for _ in range(2):
            model(values[:1])
        times = []; torch.cuda.reset_peak_memory_stats()
        for i in range(16):
            torch.cuda.synchronize(); start = time.perf_counter(); model(values[i:i+1]); torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        write_json(row/'batch1_profile.json', dict(samples=16, warmup_calls=2, includes_frozen_context=True,
            median_ms=float(np.median(times)), p90_ms=float(np.quantile(times, .9)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), source_only=True))


def prediction_arrays(folder, ids, kind, role):
    complete = read_json(folder/'complete.json')
    names = REFERENCES if kind == 'reference_controls' else ('restored',)
    expected = dict(status='PREDICTIONS_COMPLETE', kind=kind, role=role, count=len(ids), views=list(VIEWS),
        readouts=list(names), TX_truth_reads=0, state_updates=0, head_fit_updates=0, F_run=REFERENCE_RUN, F_seed=2026092701)
    if any(complete.get(key) != value for key, value in expected.items()):
        raise ValueError('All registered views required before truth')
    if not np.array_equal(np.load(folder/'physical_ids.npy', allow_pickle=False), ids):
        raise ValueError('Prediction physical IDs differ from the fixed inputs')
    result = {}
    for view in VIEWS:
        result[view] = {}
        for name in names:
            result[view][name] = {}
            for key, shape in (('iq', (len(ids), 2, 256)), ('logits', (len(ids), 6)), ('features', (len(ids), 160))):
                array = np.load(folder/(view+'.'+name+'.'+key+'.npy'), mmap_mode='r', allow_pickle=False)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError('Incomplete or nonfinite predictions; truth stays closed')
                result[view][name][key] = array
        if kind in ('native_map_prior', 'uncertain_map_prior'):
            variance = np.load(folder/(view+'.variance_mean.npy'), mmap_mode='r', allow_pickle=False)
            if variance.shape != (len(ids),) or not np.isfinite(variance).all() or (variance <= 0).any():
                raise ValueError('Invalid estimated variance diagnostics')
    return result


def waveform(estimate, clean):
    values = {key: [] for key in ('iq_nmse', 'prefix232_nmse', 'tail24_nmse', 'fft_log_power_L1')}
    for begin in range(0, len(clean), 2048):
        x, y = clean[begin:begin+2048], estimate[begin:begin+2048]
        error = (y-x)**2
        for key, selected in (('iq_nmse', slice(None)), ('prefix232_nmse', slice(0, 232)), ('tail24_nmse', slice(-24, None))):
            values[key].append(error[:, :, selected].mean((1, 2))/np.maximum((x[:, :, selected]**2).mean((1, 2)), 1e-4))
        a = np.fft.fft(y[:, 0]+1j*y[:, 1], norm='ortho')
        b = np.fft.fft(x[:, 0]+1j*x[:, 1], norm='ortho')
        values['fft_log_power_L1'].append(np.mean(abs(np.log(abs(a)**2+1e-3)-np.log(abs(b)**2+1e-3)), -1))
    return {key: np.concatenate(parts) for key, parts in values.items()}


def target_baseline(args, ids):
    if args.formal_f50.name != FORMAL_F50:
        raise ValueError('Original complete F50 baseline required')
    done = read_json(args.formal_f50/'prediction/complete.json')
    if (done['status'] != 'PREDICTIONS_COMPLETE' or done['count'] != 168000 or done['views'] != list(VIEWS)
            or done['TX_truth_reads'] != 0 or done['state_updates'] != 0 or not done['original_default_head']
            or done['F_run'] != REFERENCE_RUN or done['F_seed'] != 2026092701):
        raise ValueError('Completed same-F50 baseline predictions required')
    if not np.array_equal(np.load(args.formal_f50/'prediction/physical_ids.npy', allow_pickle=False), ids):
        raise ValueError('Formal baseline physical IDs differ')
    result = {}
    for view in VIEWS:
        logits = np.load(args.formal_f50/'prediction'/(view+'.identity.logits.npy'), mmap_mode='r', allow_pickle=False)
        if logits.shape != (len(ids), 6) or not np.isfinite(logits).all():
            raise ValueError('Incomplete original F50 baseline')
        result[view] = logits
    clean_features = np.load(args.formal_f50/'prediction/clean.identity.features.npy', mmap_mode='r', allow_pickle=False)
    if clean_features.shape != (len(ids), 160) or not np.isfinite(clean_features).all():
        raise ValueError('Incomplete paired clean F50 reference')
    return result, clean_features


def score(args, role, *, reference_loader=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU truth-side scorer required')
    folder, ids, source, index = (reference if reference_loader is None else reference_loader)(args, role)
    row = row_folder(args, role)
    predictions = prediction_arrays(row/'prediction', ids, args.kind, role)
    if role == 'source':
        refs = predictions if args.kind == 'reference_controls' else prediction_arrays(args.output/'reference_controls/prediction', ids, 'reference_controls', 'source')
        baseline = {view: refs[view]['identity']['logits'] for view in VIEWS}
        clean_features = refs['clean']['identity']['features']
        # First TX-label open after every prediction validates.
        labels = np.load(args.source_cache/'V/labels.npy', allow_pickle=False)[index]
        rx, day = source.rx[index], source.day[index]
    else:
        baseline, clean_features = target_baseline(args, ids)
        # First target-truth open, after new and existing predictions validate.
        truth = read_json(args.target_truth)
        labels = np.asarray([truth[str(pid)]['label'] for pid in ids], np.int64)
        rx = np.asarray([str(truth[str(pid)]['receiver']) for pid in ids])
        day = np.asarray([str(truth[str(pid)].get('day', 'unavailable')) for pid in ids])
    clean = np.load(folder/'clean.npy', mmap_mode='r', allow_pickle=False)
    clean_unit = clean_features/np.maximum(np.linalg.norm(clean_features, axis=1, keepdims=True), 1e-8)
    records = []; views = {}
    for view in VIEWS:
        views[view] = {}
        before = baseline[view].argmax(1)
        for name, arrays in predictions[view].items():
            metrics = waveform(arrays['iq'], clean)
            decision = arrays['logits'].argmax(1)
            groups = [('overall', 'all', np.ones(len(ids), bool))]
            for dimension, values in (('TX', labels), ('RX', rx), ('day', day)):
                groups += [(dimension, str(value), values == value) for value in np.unique(values)]
            for dimension, value, mask in groups:
                records.append(dict(view=view, readout=name, dimension=dimension, value=value,
                    **classification_statistics(labels[mask], decision[mask], before[mask]),
                    iq_nmse=float(metrics['iq_nmse'][mask].mean())))
            unit = arrays['features']/np.maximum(np.linalg.norm(arrays['features'], axis=1, keepdims=True), 1e-8)
            views[view][name] = dict(waveform={key: float(value.mean()) for key, value in metrics.items()},
                paired_clean_feature_cosine_distance=float((1-(unit*clean_unit).sum(1)).mean()),
                paired_clean_vs_other_TX=capped_pair_margin(unit, clean_unit, labels, 512))
    write_json(row/'summary.json', dict(status='ANALYZED', kind=args.kind, role=role, count=len(ids), views=views,
        records=records, F_run=REFERENCE_RUN, F_seed=2026092701, original_default_head=True,
        all_predictions_before_truth=True, training_TX_labels_read=0, training_F_access=False,
        head_fit_updates=0, target_access=role == 'target', goal_complete=False))
