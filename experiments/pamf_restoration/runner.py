"""Serial local source-only screen, followed by one frozen confirmation.

Run as python -m experiments.pamf_restoration.runner. Explicit output roots
must not exist. Each completed row writes all seven prediction views before
a separate CPU scoring process is allowed to read waveform references.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT/'code') not in sys.path:
    sys.path.append(str(ROOT/'code'))

from . import design as d
from .data import SCENES, IQSource, prepare_cache, fixed_noise, write_json
from .models import build, NAMES, ConditionalDiffusion, ConditionalFlow
from .views import PAMFViews, objective, metrics


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def code_commit():
    marker = ROOT/'release_commit.txt'
    if marker.is_file():
        return marker.read_text(encoding='utf-8').strip()
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def array_tensor(array, device):
    value = np.ascontiguousarray(array, dtype=np.float32)
    return torch.frombuffer(bytearray(value.tobytes()), dtype=torch.float32).reshape(value.shape).to(device)


def tensor_array(tensor):
    # N607's Torch2.1/NumPy2 C bridge is incompatible. This verified route
    # also works on the local Torch2.10 environment without a package change.
    return np.asarray(tensor.detach().cpu().tolist(), dtype=np.float32)


def predict_batch(model, received, ids):
    if isinstance(model, (ConditionalDiffusion, ConditionalFlow)):
        return model(received, fixed_noise(ids, d.EVALUATION_SEED, received.shape, received.device))
    return model(received)


def save_checkpoint(path, model, optimizer, epoch, updates, source_cache):
    # No dataset arrays, label files, identity classifier or EMA in checkpoint.
    torch.save(dict(schema='pamf_restorer_checkpoint_v1', model=model.state_dict(),
                    optimizer=optimizer.state_dict(), epoch=epoch, updates=updates,
                    config=d.configuration(), source_cache=str(source_cache),
                    TX_labels_read=0, target_training=False, initialization='scratch'), path)


def load_checkpoint(path, name, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload['schema'] != 'pamf_restorer_checkpoint_v1' or payload['config'] != d.configuration():
        raise ValueError('Restoration checkpoint configuration differs')
    if payload['TX_labels_read'] != 0 or payload['target_training'] or payload['initialization'] != 'scratch':
        raise ValueError('Restoration checkpoint provenance differs')
    model = build(name).to(device)
    model.load_state_dict(payload['model'], strict=True)
    return model.eval(), payload


def train_row(name, cache, output, device, source_cache, frozen_bundle):
    folder = output/name
    folder.mkdir()
    write_json(folder/'resolved_config.json', dict(d.configuration(), candidate=name))
    torch.manual_seed(d.MODEL_SEED)
    torch.cuda.manual_seed_all(d.MODEL_SEED)
    model = build(name).to(device)
    extractor = PAMFViews().to(device).eval()
    write_json(folder/'gpu_ready.json', dict(pid=os.getpid(), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
               cwd=str(ROOT), argv=sys.argv, time=time.time()))
    clean = np.load(cache/'train/clean.npy', mmap_mode='r', allow_pickle=False)
    received = np.load(cache/'train/mixed.npy', mmap_mode='r', allow_pickle=False)
    ids = np.load(cache/'train/physical_ids.npy', allow_pickle=False)
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=d.LEARNING_RATE, weight_decay=d.WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(clean)/d.BATCH_SIZE)
    total_steps = steps_per_epoch*d.EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    shuffle = np.random.default_rng(d.DATA_SEED)
    diffusion_rng = torch.Generator(device=device).manual_seed(d.AUGMENTATION_SEED+10)
    updates = labeled = unlabeled = identity = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with (folder/'training.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, d.EPOCHS+1):
            model.train()
            order = shuffle.permutation(len(clean))
            # Independent RNG makes the clean/satellite mask identical across
            # architectures, including diffusion's additional random draws.
            mask = np.random.default_rng(d.AUGMENTATION_SEED+epoch).random(len(clean)) < .2
            sums = {}
            for offset in range(0, len(order), d.BATCH_SIZE):
                selected = order[offset:offset+d.BATCH_SIZE]
                x = array_tensor(clean[selected], device)
                y = array_tensor(received[selected], device)
                clean_mask = torch.tensor(mask[selected].tolist(), device=device)
                y = torch.where(clean_mask[:, None, None], x, y)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    target_views = extractor(x)
                if isinstance(model, (ConditionalDiffusion, ConditionalFlow)):
                    estimate, velocity_loss = model.training_prediction(x, y, diffusion_rng)
                    reconstruction, terms = objective(estimate, x, extractor, target_views)
                    loss = velocity_loss + .2*reconstruction
                    terms['velocity'] = velocity_loss
                else:
                    estimate = model(y)
                    loss, terms = objective(estimate, x, extractor, target_views)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite restoration loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                updates += 1
                labeled += int((roles[selected] == 'L_s').sum())
                unlabeled += int((roles[selected] == 'U_s').sum())
                identity += int(mask[selected].sum())
                terms = dict(terms, total=loss, gradient_norm=norm)
                term_values = torch.stack([value.detach() for value in terms.values()]).cpu().tolist()
                for key, value in zip(terms, term_values):
                    sums[key] = sums.get(key, 0.)+value*len(selected)
                if updates == 1:
                    # First launcher step: a real source-batch optimization,
                    # real checkpoint, strict fresh-model roundtrip, no query.
                    smoke_path = folder/'initial_source_smoke.pt'
                    save_checkpoint(smoke_path, model, optimizer, epoch, updates, source_cache)
                    fresh, _ = load_checkpoint(smoke_path, name, device)
                    before_training = model.training
                    model.eval()
                    with torch.no_grad():
                        a = predict_batch(model, y[:2], ids[selected][:2])
                        b = predict_batch(fresh, y[:2], ids[selected][:2])
                    if not torch.equal(a, b) or not torch.isfinite(b).all():
                        raise ValueError('Real-checkpoint no-query smoke differs')
                    write_json(folder/'smoke.json', dict(status='PASS', real_source_samples=len(selected),
                               optimizer_updates=1, strict_checkpoint_roundtrip=True, target_reads=0,
                               TX_label_reads=0, inference_shape=list(b.shape)))
                    model.train(before_training)
                    del fresh
            row = dict(event='EPOCH_COMPLETE', candidate=name, epoch=epoch, updates=updates,
                       losses={key: value/len(clean) for key, value in sums.items()},
                       L_exposures=labeled, U_exposures=unlabeled, clean_identity_exposures=identity,
                       seconds=time.perf_counter()-started)
            log.write(json.dumps(row, allow_nan=False)+'\n')
            log.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
    save_checkpoint(folder/'final.pt', model, optimizer, d.EPOCHS, updates, source_cache)
    write_json(folder/'training_complete.json', dict(status='TRAINING_COMPLETE', candidate=name,
               epochs=d.EPOCHS, updates=updates, parameters=sum(p.numel() for p in model.parameters()),
               train_distinct_L=int((roles == 'L_s').sum()), train_distinct_U=int((roles == 'U_s').sum()),
               L_exposures=labeled, U_exposures=unlabeled, clean_identity_exposures=identity,
               TX_labels_read=0, source_only=True, target_reads=0,
               seconds=time.perf_counter()-started,
               peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
    del model, extractor, optimizer
    torch.cuda.empty_cache()
    predict_row(name, cache/'validation', folder, device, frozen_bundle, source_cache)
    subprocess.run([sys.executable, '-X', 'utf8', '-m', 'experiments.pamf_restoration.runner',
                    'score', '--reference', str(cache/'validation'), '--row-output', str(folder),
                    '--source-cache', str(source_cache)],
                   cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)


@torch.no_grad()
def predict_row(name, reference, folder, device, frozen_bundle, source_cache):
    from .frozen import load_readers
    model, checkpoint = load_checkpoint(folder/'final.pt', name, device)
    readers = load_readers(frozen_bundle, source_cache, device)
    if checkpoint['epoch'] != d.EPOCHS:
        raise ValueError('Fixed final checkpoint required')
    ids = np.load(reference/'physical_ids.npy', allow_pickle=False)
    prediction = folder/'prediction'
    prediction.mkdir()
    elapsed = {}
    restoration_elapsed = {}
    stochastic_seconds = {}
    # Predictor deliberately does not load the clean *reference*. Clean view
    # is explicitly an identity-preservation input; it is never paired with
    # the other views inside the predictor or used for adaptation.
    for view in ('clean', *SCENES):
        received = np.load(reference/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        out = np.lib.format.open_memmap(prediction/(view+'.npy'), mode='w+', dtype=np.float32, shape=received.shape)
        logits = {reader: {mode: np.lib.format.open_memmap(prediction/f'{view}.{reader}.{mode}.logits.npy',
                           mode='w+', dtype=np.float32, shape=(len(ids), 6)) for mode in ('restored', 'identity')}
                  for reader in readers}
        embeddings = {reader: {mode: np.lib.format.open_memmap(prediction/f'{view}.{reader}.{mode}.features.npy',
                           mode='w+', dtype=np.float32, shape=(len(ids), 160)) for mode in ('restored', 'identity')}
                  for reader in readers}
        started = time.perf_counter()
        restoration_seconds = 0.
        for offset in range(0, len(received), d.BATCH_SIZE):
            y = array_tensor(received[offset:offset+d.BATCH_SIZE], device)
            torch.cuda.synchronize()
            restoration_start = time.perf_counter()
            restored = predict_batch(model, y, ids[offset:offset+len(y)])
            torch.cuda.synchronize()
            restoration_seconds += time.perf_counter()-restoration_start
            if not torch.isfinite(restored).all():
                raise FloatingPointError('Nonfinite restoration prediction')
            out[offset:offset+len(y)] = tensor_array(restored)
            for reader_name, reader in readers.items():
                for mode, value in (('restored', restored), ('identity', y)):
                    feature = reader.features(value)
                    scores = reader.classify_features(feature)
                    if not torch.isfinite(scores).all() or not torch.isfinite(feature).all():
                        raise FloatingPointError('Nonfinite frozen default-head prediction')
                    logits[reader_name][mode][offset:offset+len(y)] = tensor_array(scores)
                    embeddings[reader_name][mode][offset:offset+len(y)] = tensor_array(feature)
        out.flush()
        for arrays in (logits, embeddings):
            for reader_arrays in arrays.values():
                for array in reader_arrays.values():
                    array.flush()
        if isinstance(model, (ConditionalDiffusion, ConditionalFlow)):
            extra_started = time.perf_counter()
            count = min(128, len(ids))
            y = array_tensor(received[:count], device)
            for draw, delta in enumerate((101, 202), 1):
                alternate = model(y, fixed_noise(ids[:count], d.EVALUATION_SEED+delta, y.shape, device))
                if not torch.isfinite(alternate).all():
                    raise FloatingPointError('Nonfinite fixed stochastic diagnostic')
                np.save(prediction/f'{view}.draw{draw}.npy', tensor_array(alternate), allow_pickle=False)
                for reader_name, reader in readers.items():
                    scores = reader(alternate)
                    if not torch.isfinite(scores).all():
                        raise FloatingPointError('Nonfinite diagnostic default-head output')
                    np.save(prediction/f'{view}.{reader_name}.draw{draw}.logits.npy', tensor_array(scores), allow_pickle=False)
            stochastic_seconds[view] = time.perf_counter()-extra_started
        elapsed[view] = time.perf_counter()-started
        restoration_elapsed[view] = restoration_seconds
        print(json.dumps(dict(event='VIEW_PREDICTION_COMPLETE', candidate=name, view=view, count=len(out))), flush=True)
    np.save(prediction/'physical_ids.npy', ids, allow_pickle=False)
    write_json(prediction/'complete.json', dict(status='PREDICTIONS_COMPLETE', candidate=name,
               views=['clean', *SCENES], count=len(ids), per_view_seconds=elapsed,
               restoration_only_seconds=restoration_elapsed, frozen_readers=list(readers),
               stochastic_diagnostic_seconds=stochastic_seconds,
               stochastic_diagnostic_draws=3 if stochastic_seconds else 1,
               stochastic_diagnostic_count=min(128, len(ids)) if stochastic_seconds else 0,
               one_noise_draw_per_physical_sample=True, TX_truth_reads=0, state_updates=0))


def score_row(reference, folder, source_cache=None, target_truth=None):
    done = read_json(folder/'prediction/complete.json')
    if done['status'] != 'PREDICTIONS_COMPLETE' or done['views'] != ['clean', *SCENES]:
        raise ValueError('All registered predictions must be fixed before scoring')
    ids = np.load(reference/'physical_ids.npy', allow_pickle=False)
    pred_ids = np.load(folder/'prediction/physical_ids.npy', allow_pickle=False)
    if not np.array_equal(ids, pred_ids) or len(ids) != done['count']:
        raise ValueError('Prediction physical coverage differs')
    # Validate every class prediction before opening any TX labels/truth.
    for view in ('clean', *SCENES):
        for reader in done.get('frozen_readers', []):
            for mode in ('restored', 'identity'):
                scores = np.load(folder/'prediction'/f'{view}.{reader}.{mode}.logits.npy', mmap_mode='r', allow_pickle=False)
                if scores.shape != (len(ids), 6) or not np.isfinite(scores).all():
                    raise ValueError('Incomplete default-head prediction; truth remains closed')
            for draw in range(1, done.get('stochastic_diagnostic_draws', 1)):
                scores = np.load(folder/'prediction'/f'{view}.{reader}.draw{draw}.logits.npy', allow_pickle=False)
                if scores.shape != (done['stochastic_diagnostic_count'], 6) or not np.isfinite(scores).all():
                    raise ValueError('Incomplete stochastic prediction; truth remains closed')
    if source_cache is not None and target_truth is not None:
        raise ValueError('One scoring role per independent process')
    labels = None
    if source_cache is not None:
        source_ids = np.load(source_cache/'V/physical_ids.npy', allow_pickle=False)
        source_labels = np.load(source_cache/'V/labels.npy', allow_pickle=False)
        lookup = dict(zip(source_ids.tolist(), source_labels.tolist()))
        if not set(ids.tolist()) <= lookup.keys():
            raise ValueError('Scorer IDs must belong to V')
        labels = np.array([lookup[str(pid)] for pid in ids], np.int64)
    elif target_truth is not None:
        truth = read_json(target_truth)
        labels = np.array([truth[str(pid)]['label'] for pid in ids], np.int64)
    clean = np.load(reference/'clean.npy', mmap_mode='r', allow_pickle=False)
    rx = np.load(reference/'rx.npy', allow_pickle=False)
    day = np.load(reference/'day.npy', allow_pickle=False)
    if target_truth is not None:
        rx = np.asarray([str(truth[str(pid)]['receiver']) for pid in ids])
        day = np.asarray([str(truth[str(pid)].get('day', 'unavailable')) for pid in ids])
    extractor = PAMFViews().eval()
    results = {}
    with torch.no_grad():
        for view in ('clean', *SCENES):
            received = np.load(reference/(view+'.npy'), mmap_mode='r', allow_pickle=False)
            restored = np.load(folder/'prediction'/(view+'.npy'), mmap_mode='r', allow_pickle=False)
            if restored.shape != clean.shape:
                raise ValueError('Incomplete prediction shape')
            accumulators = {'restored': {}, 'identity': {}}
            for offset in range(0, len(clean), d.BATCH_SIZE):
                x = array_tensor(clean[offset:offset+d.BATCH_SIZE], 'cpu')
                target = extractor(x)
                for method, array in (('restored', restored), ('identity', received)):
                    values = metrics(array_tensor(array[offset:offset+len(x)], 'cpu'), x, extractor, target)
                    for key, value in values.items():
                        if not torch.isfinite(value).all():
                            raise FloatingPointError('Nonfinite scoring metric')
                        accumulators[method].setdefault(key, []).append(tensor_array(value))
            values = {method: {key: np.concatenate(chunks) for key, chunks in result.items()}
                      for method, result in accumulators.items()}
            summary = {method: {key: float(v.mean()) for key, v in result.items()} for method, result in values.items()}
            strata = []
            for receiver in sorted(set(rx.tolist())):
                for date in sorted(set(day.tolist())):
                    mask = (rx == receiver) & (day == date)
                    if mask.any():
                        strata.append(dict(rx=str(receiver), day=str(date), count=int(mask.sum()),
                          metrics={method: {key: float(v[mask].mean()) for key, v in result.items()} for method, result in values.items()}))
            classification = {}
            if labels is not None:
                from .frozen import classification_statistics
                for reader in done['frozen_readers']:
                    a = np.load(folder/'prediction'/f'{view}.{reader}.restored.logits.npy', allow_pickle=False)
                    b = np.load(folder/'prediction'/f'{view}.{reader}.identity.logits.npy', allow_pickle=False)
                    item = classification_statistics(labels, a.argmax(1), b.argmax(1))
                    z = np.load(folder/'prediction'/f'{view}.{reader}.restored.features.npy', allow_pickle=False)
                    ref_z = np.load(folder/'prediction'/f'clean.{reader}.identity.features.npy', allow_pickle=False)
                    item['mean_cosine_distance_to_original_clean_features'] = float(
                        (1-(z*ref_z).sum(1)/np.maximum(np.linalg.norm(z, axis=1)*np.linalg.norm(ref_z, axis=1), 1e-8)).mean())
                    item['receiver'] = {str(receiver): classification_statistics(labels[rx == receiver], a[rx == receiver].argmax(1), b[rx == receiver].argmax(1)) for receiver in sorted(set(rx.tolist()))}
                    item['per_tx'] = {str(tx): classification_statistics(labels[labels == tx], a[labels == tx].argmax(1), b[labels == tx].argmax(1)) for tx in sorted(set(labels.tolist()))}
                    competitors = a.copy()
                    competitors[np.arange(len(labels)), labels] = -np.inf
                    item['true_TX_margin_mean'] = float((a[np.arange(len(labels)), labels]-competitors.max(1)).mean())
                    unit = z/np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
                    reference_unit = ref_z/np.maximum(np.linalg.norm(ref_z, axis=1, keepdims=True), 1e-8)
                    similarities = unit@reference_unit.T
                    paired = similarities.diagonal().copy()
                    similarities[labels[:, None] == labels[None, :]] = -np.inf
                    margin = paired-similarities.max(1)
                    item['truthside_paired_clean_feature_vs_other_TX_margin'] = dict(
                        mean=float(margin.mean()), own_pair_higher_percent=float((margin > 0).mean()*100),
                        inference_or_selection_use=False)
                    if done.get('stochastic_diagnostic_draws', 1) > 1:
                        count = done['stochastic_diagnostic_count']
                        predictions = [a[:count].argmax(1)]
                        for draw in range(1, done['stochastic_diagnostic_draws']):
                            predictions.append(np.load(folder/'prediction'/f'{view}.{reader}.draw{draw}.logits.npy', allow_pickle=False).argmax(1))
                        draws = np.stack(predictions)
                        item['stochastic_diagnostic'] = dict(count=count, draws=len(draws),
                            any_class_switch_percent=float((draws != draws[:1]).any(0).mean()*100),
                            per_draw_accuracy=[float((pred == labels[:count]).mean()*100) for pred in predictions],
                            main_draw=0, selection_or_averaging=False)
                    classification[reader] = item
            results[view] = dict(count=len(clean), summary=summary, strata=strata, classification=classification)
    result = dict(status='SCORED', candidate=done['candidate'], views=results,
                  checkpoint='final.pt', prediction_complete_before_truth=True, training_TX_labels_read=0,
                  scored_TX_labels=len(ids) if labels is not None else 0,
                  scorer_TX_labels_read=len(source_labels) if source_cache is not None else len(truth) if target_truth is not None else 0,
                  identity_accuracy='frozen existing cleanF and AsF default heads; no fitted head',
                  probe='Original analytic transforms; default initialized Sinc bank; no learned identity features',
                  interpretation='Raw IQ and scalar-aligned IQ are separate; clean is original received ground IQ')
    write_json(folder/'metrics.json', result)
    print(json.dumps(dict(event='ROW_SCORED', candidate=done['candidate'])), flush=True)


def source_decision(output):
    results = []
    rules = d.configuration()['source_thresholds']
    for name in NAMES:
        row = read_json(output/name/'metrics.json')
        def mean(method, metric):
            return float(np.mean([row['views'][s]['summary'][method][metric] for s in SCENES]))
        ratios = {metric: mean('restored', metric)/max(mean('identity', metric), 1e-12)
                  for metric in ('iq_nmse', 'aligned_iq_nmse', 'fft_log_mse', 'reference_nmse')}
        clean = row['views']['clean']['summary']['restored']['iq_nmse']
        low = row['views']['practical_low_urban']['classification']['AsF']
        original_clean = row['views']['clean']['classification']['AsF']
        gain = low['accuracy']-low['baseline_accuracy']
        drop = original_clean['baseline_accuracy']-original_clean['accuracy']
        eligible = (gain >= rules['low_urban_accuracy_gain_pp_min'] and
                    drop <= rules['clean_accuracy_drop_pp_max'] and
                    ratios['iq_nmse'] <= rules['raw_iq_mean_ratio_max'] and clean <= rules['clean_iq_nmse_max'])
        results.append(dict(candidate=name, eligible=bool(eligible), ratios=ratios,
                       low_urban_accuracy=low['accuracy'], low_urban_gain_pp=gain,
                       clean_accuracy_drop_pp=drop, clean_iq_nmse=clean,
                       mean_aligned_iq_nmse=mean('restored', 'aligned_iq_nmse')))
    eligible = [row for row in results if row['eligible']]
    selected = max(eligible or results, key=lambda row: row['low_urban_accuracy'])['candidate']
    result = dict(status='SOURCE_SELECTION_FROZEN', selected=selected, rows=results,
                  source_qualified=bool(eligible), negative_confirmation=not bool(eligible),
                  target_reads_before_decision=0, training_TX_labels=False, selection_uses_source_V_TX_labels=True,
                  policy=d.configuration()['selection'], primary_reader='AsF', thresholds=rules)
    write_json(output/'source_decision.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    run = sub.add_parser('run')
    run.add_argument('--source-cache', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--target-capsule', type=Path)
    run.add_argument('--frozen-bundle', type=Path, required=True)
    run.add_argument('--expected-commit', required=True)
    dispatch = sub.add_parser('dispatch')
    dispatch.add_argument('--source-cache', type=Path, required=True)
    dispatch.add_argument('--output', type=Path, required=True)
    dispatch.add_argument('--target-capsule', type=Path, required=True)
    dispatch.add_argument('--frozen-bundle', type=Path, required=True)
    dispatch.add_argument('--expected-commit', required=True)
    dispatch.add_argument('--gpus', default='0,1,5')
    worker = sub.add_parser('worker')
    worker.add_argument('--source-cache', type=Path, required=True)
    worker.add_argument('--output', type=Path, required=True)
    worker.add_argument('--frozen-bundle', type=Path, required=True)
    worker.add_argument('--candidate', choices=NAMES, required=True)
    worker.add_argument('--expected-commit', required=True)
    confirmation = sub.add_parser('confirm')
    confirmation.add_argument('--source-cache', type=Path, required=True)
    confirmation.add_argument('--output', type=Path, required=True)
    confirmation.add_argument('--target-capsule', type=Path, required=True)
    confirmation.add_argument('--frozen-bundle', type=Path, required=True)
    confirmation.add_argument('--candidate', choices=NAMES, required=True)
    confirmation.add_argument('--expected-commit', required=True)
    score = sub.add_parser('score')
    score.add_argument('--reference', type=Path, required=True)
    score.add_argument('--row-output', type=Path, required=True)
    score.add_argument('--source-cache', type=Path)
    score.add_argument('--target-truth', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.mode == 'score':
        score_row(args.reference, args.row_output, args.source_cache, args.target_truth)
        return
    if args.mode == 'dispatch':
        from .dispatcher import dispatch
        dispatch(args)
        return
    commit = code_commit()
    if commit != args.expected_commit:
        raise ValueError('Pinned code commit differs')
    if args.mode == 'worker':
        train_row(args.candidate, args.output/'cache', args.output, 'cuda:0', args.source_cache, args.frozen_bundle)
        return
    if args.mode == 'confirm':
        from .confirmation import confirm
        confirm(args.candidate, args.target_capsule, args.output, args.frozen_bundle, args.source_cache)
        return
    if args.output.exists():
        raise FileExistsError('Immutable run root exists; inspect original state before any recovery')
    args.output.mkdir(parents=True)
    write_json(args.output/'resolved_config.json', d.configuration())
    write_json(args.output/'launch.json', dict(status='STARTING', pid=os.getpid(), cwd=str(ROOT),
               argv=sys.argv, python=sys.executable, commit=commit, physical_gpu=0,
               device=torch.cuda.get_device_name(0), source_cache=str(args.source_cache),
               target_capsule=None if args.target_capsule is None else str(args.target_capsule),
               launch_owner='codex/root/pamf-restoration', max_gpu_workers=1, started=time.time()))
    try:
        from .frozen import load_readers
        readers = load_readers(args.frozen_bundle, args.source_cache, 'cuda:0')
        source = IQSource(args.source_cache, 'L_s')
        x = array_tensor(source.iq[:2], 'cuda:0')
        with torch.no_grad():
            if any(not torch.isfinite(reader(x)).all() for reader in readers.values()):
                raise ValueError('Real frozen PAMF no-query smoke failed')
        write_json(args.output/'reference_smoke.json', dict(status='PASS', readers=list(readers),
                   physical_source_samples=2, labels_read=0, target_reads=0,
                   data_contract='Exact original physical roles; full AsF scratch20+own10 lineage'))
        del readers, x, source
        torch.cuda.empty_cache()
        cache = args.output/'cache'
        prepare_cache(args.source_cache, cache, d.TRAIN_L_PER_SESSION, d.TRAIN_U_PER_SESSION,
                      d.VALIDATION_PER_SESSION, d.DATA_SEED)
        for name in NAMES:
            train_row(name, cache, args.output, 'cuda:0', args.source_cache, args.frozen_bundle)
        decision = source_decision(args.output)
        if decision['selected'] is not None and args.target_capsule is not None:
            from .confirmation import confirm
            confirm(decision['selected'], args.target_capsule, args.output, args.frozen_bundle, args.source_cache)
        status = 'SOURCE_SCREEN_COMPLETE_NO_ELIGIBLE_MODEL' if decision['selected'] is None else (
            'SOURCE_SCREEN_COMPLETE_TARGET_INPUT_MISSING' if args.target_capsule is None else 'ANALYZED')
        write_json(args.output/'complete.json', dict(status=status, source_rows=list(NAMES),
                   selected=decision['selected'], identity_accuracy_measured=True,
                   reconstruction_threshold_met=decision['source_qualified'],
                   target_confirmation=decision['selected'] is not None and args.target_capsule is not None))
    except BaseException as error:
        write_json(args.output/'failure.json', dict(status='FAILED_TECHNICAL', error=repr(error),
                   traceback=traceback.format_exc(), automatic_retry=False, artifacts_preserved=True))
        raise


if __name__ == '__main__':
    main()
