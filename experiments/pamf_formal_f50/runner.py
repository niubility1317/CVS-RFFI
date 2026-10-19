"""Original frozen F50 reads already completed formal restoration IQ."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from experiments.pamf_fcrossrx_readout.frozen import load_reader, REFERENCE_RUN
from experiments.pamf_formal_restore.runner import source_parent, source_residual_parent
from experiments.pamf_formal_restore.reference import COUNT
from experiments.pamf_restoration.data import IQSource, SCENES, write_json
from experiments.pamf_restoration.runner import array_tensor, tensor_array, read_json, code_commit
from experiments.pamf_restoration.dispatcher import capacity
from experiments.pamf_restoration.frozen import classification_statistics
from experiments.pamf_restoration.feature_diagnostic import capped_pair_margin

ROOT = Path(__file__).resolve().parents[2]
RUN = '20261019-phase1-restoration-formal-FcrossRX50-s2026092701-r08'
REPLACEMENT_RUN = '20261019-phase1-restoration-formal-FcrossRX50-s2026092701-r09'
COARSE_RUN = '20261019-phase1-restoration-formal-frozen-coarse-s2026092701-r04'
RESIDUAL_RUN = '20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05'
COARSE_COMMIT = 'a173cc5002262f243abeb5a904433fcac42ee1cf'
RESIDUAL_COMMIT = '9522a4edb63f30293542de7326ce1e53aa816ed5'
VIEWS = ('clean', *SCENES)
MODES = ('identity', 'coarse', 'residual')


def plan_settings(plan):
    if plan == 'R08_gpu4':
        return RUN, 4
    if plan == 'R09_gpu5':
        return REPLACEMENT_RUN, 5
    raise ValueError('Only the two preregistered resource plans are allowed')


def configuration(plan='R08_gpu4'):
    run, gpu = plan_settings(plan)
    return dict(schema='formal_F50_frozen_readout_v1', run=run, count=COUNT, views=list(VIEWS), readouts=list(MODES),
        F_run=REFERENCE_RUN, F_seed=2026092701, F_epochs=50, F_updates=11100, original_default_head=True,
        new_training_updates=0, adaptation_updates=0, head_fit_updates=0, channel_generation_calls=0,
        restorer_choice='Existing R02/R03 source-frozen selections;reuse R04/R05 completed IQ,not target ranking',
        IQ_prediction_owner='R04/R05 complete immutable predictions,not this readout',
        feature_reference_cap=512, GPU=gpu, capacity_lock='receiver_residual_capacity4.lock',
        registered_plan=plan, replaces_unstarted_run=RUN if plan == 'R09_gpu5' else None,
        truth_last=True, no_R07_result_or_checkpoint_use=True,
        scope='Fixed three-way full formal readout on exposed engineering proxy;not new independent blind/in-orbit evidence')


def completed_inputs(coarse, residual, expected_count=COUNT):
    if coarse.name != COARSE_RUN or residual.name != RESIDUAL_RUN:
        raise ValueError('Only original completed formal IQ roots are registered')
    for root, commit, candidate in ((coarse, COARSE_COMMIT, 'tf_unet_mixed'),
                                    (residual, RESIDUAL_COMMIT, 'residual_deterministic')):
        complete = read_json(root/'complete.json')
        launch = read_json(root/'launch.json')
        prediction = read_json(root/candidate/'prediction/complete.json')
        if (launch['commit'] != commit or complete['status'] != 'ANALYZED' or complete['count'] != expected_count or
                complete['training_updates'] != 0 or complete['adaptation_updates'] != 0 or not complete['truth_last'] or
                prediction['status'] != 'PREDICTIONS_COMPLETE' or prediction['candidate'] != candidate or
                prediction['count'] != expected_count or prediction['views'] != list(VIEWS) or
                prediction['TX_truth_reads'] != 0 or prediction['state_updates'] != 0):
            raise ValueError('Completed zero-update formal prediction contract differs')
    reference = residual/'reference'
    manifest = read_json(reference/'manifest.json')
    if (manifest['status'] != 'FORMAL_REFERENCE_READY' or manifest['count'] != expected_count or
            manifest['views'] != list(VIEWS) or not manifest['original_IQ_bytes_reused'] or manifest['TX_truth_reads'] != 0):
        raise ValueError('Existing formal reference handles required')
    ids = np.load(reference/'physical_ids.npy', allow_pickle=False)
    if ids.shape != (expected_count,):
        raise ValueError('Formal input count differs')
    paths = dict(identity=reference, coarse=coarse/'tf_unet_mixed/prediction',
                 residual=residual/'residual_deterministic/prediction')
    for name, folder in paths.items():
        if name != 'identity' and not np.array_equal(np.load(folder/'physical_ids.npy', allow_pickle=False), ids):
            raise ValueError('Completed restoration physical IDs differ')
        for view in VIEWS:
            array = np.load(folder/(view+'.npy'), mmap_mode='r', allow_pickle=False)
            if array.shape != (expected_count, 2, 256) or array.dtype != np.float32:
                raise ValueError('Completed IQ view missing or wrong shape')
    return ids, paths


@torch.no_grad()
def predict(args):
    import fcntl
    _, gpu = plan_settings(args.plan)
    source_parent(args.parent_coarse); source_residual_parent(args.parent_residual)
    while True:
        with (args.output.parent/'receiver_residual_capacity4.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            card = capacity()[gpu]
            if card['free_mb'] >= 12000 and len(card['pids']) < 2:
                reader = load_reader(args.reader_folder, args.source_cache, 'cuda:0')
                x = array_tensor(IQSource(args.source_cache, 'L_s').iq[:2], 'cuda:0')
                feature = reader.features(x); logits = reader.classify_features(feature)
                if feature.shape != (2, 160) or logits.shape != (2, 6) or not torch.isfinite(logits).all():
                    raise ValueError('Actual source original-F50 checkpoint smoke failed')
                torch.cuda.synchronize()
                break
        time.sleep(2)
    write_json(args.output/'smoke.json', dict(status='PASS', real_source_samples=2, target_reads=0,
        TX_labels_read=0, original_default_head=True, head_fit_updates=0))
    write_json(args.output/'gpu_ready.json', dict(pid=os.getpid(), cwd=str(ROOT), argv=sys.argv, GPU=gpu))
    ids, inputs = completed_inputs(args.formal_coarse, args.formal_residual)
    output = args.output/'prediction'; output.mkdir()
    timings = {}
    for view in VIEWS:
        started = time.perf_counter()
        arrays = {key: np.load(folder/(view+'.npy'), mmap_mode='r', allow_pickle=False) for key, folder in inputs.items()}
        features = {key: np.lib.format.open_memmap(output/(view+'.'+key+'.features.npy'), mode='w+', dtype=np.float32, shape=(len(ids), 160)) for key in MODES}
        scores = {key: np.lib.format.open_memmap(output/(view+'.'+key+'.logits.npy'), mode='w+', dtype=np.float32, shape=(len(ids), 6)) for key in MODES}
        for begin in range(0, len(ids), 256):
            end = min(begin+256, len(ids))
            for key, array in arrays.items():
                x = array_tensor(array[begin:end], 'cuda:0')
                if not torch.isfinite(x).all():
                    raise ValueError('Nonfinite original completed IQ')
                z = reader.features(x); logits = reader.classify_features(z)
                if not torch.isfinite(z).all() or not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite frozen F50 output')
                features[key][begin:end] = tensor_array(z); scores[key][begin:end] = tensor_array(logits)
        for group in (features, scores):
            for array in group.values():
                array.flush()
        torch.cuda.synchronize(); timings[view] = time.perf_counter()-started
        print(json.dumps(dict(event='F50_VIEW_COMPLETE', view=view, count=len(ids), seconds=timings[view])), flush=True)
    np.save(output/'physical_ids.npy', ids, allow_pickle=False)
    write_json(output/'complete.json', dict(status='PREDICTIONS_COMPLETE', count=len(ids), views=list(VIEWS),
        readouts=list(MODES), TX_truth_reads=0, state_updates=0, original_default_head=True,
        per_view_seconds=timings, F_run=REFERENCE_RUN, F_seed=2026092701, restored_IQ_reused=True))


def prediction_arrays(folder, expected_count=COUNT, expected_ids=None):
    complete = read_json(folder/'complete.json')
    expected = dict(status='PREDICTIONS_COMPLETE', count=expected_count, views=list(VIEWS), readouts=list(MODES),
        TX_truth_reads=0, state_updates=0, original_default_head=True, F_run=REFERENCE_RUN, F_seed=2026092701,
        restored_IQ_reused=True)
    if any(complete.get(key) != value for key, value in expected.items()):
        raise ValueError('All formal predictions must finish before truth is opened')
    ids = np.load(folder/'physical_ids.npy', allow_pickle=False)
    if ids.shape != (expected_count,):
        raise ValueError('Formal prediction ID coverage differs')
    if expected_ids is not None and not np.array_equal(ids, expected_ids):
        raise ValueError('Formal prediction IDs differ from the original observation')
    result = {}
    for view in VIEWS:
        result[view] = {}
        for key in MODES:
            logits = np.load(folder/(view+'.'+key+'.logits.npy'), mmap_mode='r', allow_pickle=False)
            features = np.load(folder/(view+'.'+key+'.features.npy'), mmap_mode='r', allow_pickle=False)
            if (logits.shape != (expected_count, 6) or features.shape != (expected_count, 160) or
                    not np.isfinite(logits).all() or not np.isfinite(features).all()):
                raise ValueError('Incomplete/nonfinite formal prediction; truth remains closed')
            result[view][key] = (logits, features)
    return ids, result


def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU truth-side scorer required')
    expected_ids = np.load(args.formal_residual/'reference/physical_ids.npy', allow_pickle=False)
    ids, predictions = prediction_arrays(args.output/'prediction', expected_ids=expected_ids)
    # First and only truth open, after all 21 prediction sets validate.
    truth = read_json(args.target_truth)
    labels = np.asarray([truth[str(pid)]['label'] for pid in ids], np.int64)
    rx = np.asarray([str(truth[str(pid)]['receiver']) for pid in ids])
    day = np.asarray([str(truth[str(pid)].get('day', 'unavailable')) for pid in ids])
    clean = predictions['clean']['identity'][1]
    clean_unit = clean/np.maximum(np.linalg.norm(clean, axis=1, keepdims=True), 1e-8)
    records = []; features = {}
    for view in VIEWS:
        features[view] = {}
        baseline = predictions[view]['identity'][0].argmax(1)
        for key in MODES:
            logits, z = predictions[view][key]
            decision = logits.argmax(1)
            groups = [('overall', 'all', np.ones(len(ids), bool))]
            for dimension, values in (('TX', labels), ('RX', rx), ('day', day)):
                groups += [(dimension, str(value), values == value) for value in np.unique(values)]
            for dimension, value, mask in groups:
                records.append(dict(view=view, readout=key, dimension=dimension, value=value,
                    **classification_statistics(labels[mask], decision[mask], baseline[mask])))
            unit = z/np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
            competitors = logits.copy(); competitors[np.arange(len(ids)), labels] = -np.inf
            features[view][key] = dict(mean_cosine_distance_to_original_clean=float((1-(unit*clean_unit).sum(1)).mean()),
                true_TX_margin_mean=float((logits[np.arange(len(ids)), labels]-competitors.max(1)).mean()),
                paired_clean_vs_other_TX=capped_pair_margin(unit, clean_unit, labels, 512),
                inference_or_selection_use=False)
    write_json(args.output/'summary.json', dict(status='ANALYZED', count=len(ids), original_F_run=REFERENCE_RUN,
        F_seed=2026092701, original_default_head=True, new_training_updates=0, head_fit_updates=0,
        all_predictions_before_truth=True, records=records, features=features, goal_complete=False,
        limitation='Single frozen F seed,existing engineered seven-view benchmark;feature competitor cap512,notglobalnearestTX;notinorbit'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'score'))
    parser.add_argument('--plan', choices=('R08_gpu4', 'R09_gpu5'), default='R08_gpu4')
    for key in ('formal-coarse', 'formal-residual', 'parent-coarse', 'parent-residual', 'reader-folder', 'source-cache', 'output', 'target-truth'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    for key in ('formal_coarse', 'formal_residual', 'parent_coarse', 'parent_residual', 'reader_folder', 'source_cache', 'output', 'target_truth'):
        setattr(args, key, getattr(args, key).resolve())
    run, gpu = plan_settings(args.plan)
    if args.output.name != run or code_commit() != args.expected_commit:
        raise ValueError('Registered readout run/code differs')
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if args.mode == 'score':
        score(args); return
    if args.output.exists():
        raise FileExistsError('Immutable readout output exists; reconcile')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('Only the registered physical GPU is allowed')
    args.output.mkdir()
    write_json(args.output/'launch.json', dict(pid=os.getpid(), cwd=str(ROOT), argv=sys.argv, commit=args.expected_commit,
        owner='codex/root/formal-F50-readout', GPU=gpu, started=time.time()))
    write_json(args.output/'resolved_config.json', configuration(args.plan))
    try:
        predict(args)
        command = [sys.executable, '-u', '-m', 'experiments.pamf_formal_f50.runner', 'score']
        for key in ('plan', 'formal_coarse', 'formal_residual', 'parent_coarse', 'parent_residual', 'reader_folder', 'source_cache', 'output', 'target_truth', 'expected_commit'):
            command.extend(['--'+key.replace('_', '-'), str(getattr(args, key))])
        subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
        write_json(args.output/'complete.json', dict(status='ANALYZED', count=COUNT, readouts=list(MODES),
            original_F50=True, training_updates=0, truth_last=True, restored_IQ_reused=True, goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json', dict(status='FAILED_TECHNICAL', error=repr(error),
            traceback=traceback.format_exc(), outputs_preserved=True, automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
