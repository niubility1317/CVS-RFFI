"""CPU-only source research; calibration freezes before V is opened.

No TX labels, F outputs or source V are used to set the scalar threshold.
All seven prediction views finish before a fresh CPU scorer opens V labels.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT/'code') not in sys.path:
    sys.path.append(str(ROOT/'code'))

import numpy as np
import torch
from .model import POLICY, PreserveSmallChanges, calibrated_threshold, relative_change
from experiments.pamf_restoration.data import IQSource, write_json
from experiments.pamf_restoration.frozen import classification_statistics
from experiments.pamf_restoration.runner import array_tensor, tensor_array
from experiments.pamf_residual_restore.runtime import load_checkpoint
from experiments.pamf_fcrossrx_readout.frozen import load_reader, REFERENCE_RUN, REFERENCE_COMMIT
from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference, VIEWS

READOUTS = ('identity', 'original_R03', 'preserved_R03')


def load_restorer(checkpoint):
    model, payload = load_checkpoint(checkpoint, 'residual_deterministic', 'cpu')
    if payload['epoch'] != 20 or payload['updates'] != 4940:
        raise ValueError('Fixed final R03 E20 after frozen R02 E20 required')
    return model.eval().requires_grad_(False)


def calibration_selection(counts):
    if tuple(counts) != (6300, 56700):
        raise ValueError('Original complete L/U roles required')
    return np.random.default_rng(POLICY['calibration_seed']).choice(sum(counts),
                    size=POLICY['calibration_samples'], replace=False)


@torch.no_grad()
def calibrate(source_cache, restorer_checkpoint, output, batch_size=64):
    # Intentionally no reference/V or reader/F argument exists here.
    roles = [IQSource(source_cache, role) for role in ('L_s', 'U_s')]
    selected = calibration_selection([len(role.ids) for role in roles])
    values = []
    counts = [0, 0]
    model = load_restorer(restorer_checkpoint)
    for offset in range(0, len(selected), batch_size):
        chosen = selected[offset:offset+batch_size]
        batch = np.empty((len(chosen), 2, 256), np.float32)
        for group, role in enumerate(roles):
            mask = chosen < 6300 if group == 0 else chosen >= 6300
            index = chosen[mask] if group == 0 else chosen[mask]-6300
            batch[mask] = role.iq[index]
            counts[group] += len(index)
        x = array_tensor(batch, 'cpu')
        values.append(relative_change(x, model(x)))
        if (offset+len(chosen)) % 2048 == 0:
            print(json.dumps(dict(event='SOURCE_CALIBRATION_PROGRESS',
                                  count=offset+len(chosen), total=len(selected))), flush=True)
    changes = torch.cat(values)
    threshold = calibrated_threshold(changes)
    result = dict(status='SOURCE_THRESHOLD_FROZEN', policy=POLICY, threshold=threshold,
                  sample_count=len(selected), role_counts=dict(zip(('L_s', 'U_s'), counts)),
                  maximum_clean_change=float(changes.max()),
                  median_clean_change=float(changes.median()),
                  cap_active=float(changes.max()) > POLICY['threshold_cap'],
                  calibration_source_V_reads=0, TX_label_reads=0, F_reads=0, target_reads=0,
                  optimizer_updates=0, parent='R03 residual_deterministic after frozen R02',
                  parent_checkpoint=str(Path(restorer_checkpoint).resolve()))
    write_json(Path(output)/'calibration.json', result)
    # The launcher first performs a real-parent-checkpoint no-query smoke.
    # Continue directly on PASS; no receipt or additional approval is created.
    sample = array_tensor(roles[0].iq[:2], 'cpu')
    wrapper = PreserveSmallChanges(model, threshold)
    with torch.no_grad():
        expected, _, _ = wrapper.from_restored(sample, model(sample))
        fresh = PreserveSmallChanges(load_restorer(restorer_checkpoint), threshold)
        if not torch.equal(expected, fresh(sample)) or not torch.isfinite(expected).all():
            raise ValueError('Real-parent-checkpoint no-query smoke differs')
    write_json(Path(output)/'smoke.json', dict(status='PASS', real_source_samples=2,
               original_R03_strict_load=True, original_default_head_unchanged=True,
               input_IQ_only=True, TX_label_reads=0, source_V_reads=0, target_reads=0,
               optimizer_updates=0, exact_wrapper_roundtrip=True))
    return wrapper


@torch.no_grad()
def predict(reference, source_cache, reader_folder, restorer_checkpoint, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This preregistered source diagnostic uses CPU only')
    torch.set_num_threads(2)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve existing run output')
    output.mkdir(parents=True)
    write_json(output/'execution.json', dict(pid=os.getpid(), cwd=str(Path.cwd()), argv=sys.argv,
               role='source_only', GPU_workers=0, CPU_threads=2, policy=POLICY,
               readouts=list(READOUTS), views=list(VIEWS), started_at=time.time()))
    wrapper = calibrate(source_cache, restorer_checkpoint, output)
    # V and the classifier are opened only after threshold and smoke complete.
    ids, _, _ = source_reference(reference, source_cache)
    reader = load_reader(reader_folder, source_cache)
    before = {k: v.clone() for k, v in wrapper.state_dict().items()}
    saved = output/'prediction'
    saved.mkdir()
    gate_records = []
    for view in VIEWS:
        started = time.perf_counter()
        observed = np.load(Path(reference)/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        if observed.shape != (len(ids), 2, 256):
            raise ValueError('Fixed source-V shape differs')
        predictions = {k: [] for k in READOUTS}
        decisions, magnitudes = [], []
        restored_iq = np.lib.format.open_memmap(saved/(view+'_restored.npy'), mode='w+',
                         dtype=np.float32, shape=(len(ids), 2, 256))
        for offset in range(0, len(ids), 64):
            x = array_tensor(observed[offset:offset+64], 'cpu')
            restored = wrapper.restorer(x)
            guarded, keep, change = wrapper.from_restored(x, restored)
            restored_iq[offset:offset+len(x)] = tensor_array(restored)
            for name, value in zip(READOUTS, (x, restored, guarded)):
                logits = reader(value)
                if logits.shape != (len(x), 6) or not torch.isfinite(logits).all():
                    raise FloatingPointError('Invalid original default-head output')
                predictions[name].extend(logits.argmax(1).tolist())
            decisions.extend(keep.tolist())
            magnitudes.extend(change.tolist())
        restored_iq.flush()
        with (saved/(view+'.npz')).open('xb') as stream:
            np.savez_compressed(stream, ids=ids,
                  **{k: np.asarray(v, np.uint8) for k, v in predictions.items()},
                  preserve=np.asarray(decisions, bool), change=np.asarray(magnitudes, np.float64))
        record = dict(view=view, count=len(ids), preserved=int(sum(decisions)),
                      threshold=float(wrapper.threshold), change_min=min(magnitudes),
                      change_max=max(magnitudes), seconds=time.perf_counter()-started)
        gate_records.append(record)
        print(json.dumps(dict(event='SOURCE_VIEW_COMPLETE', **record)), flush=True)
    if any(not torch.equal(value, before[k]) for k, value in wrapper.state_dict().items()):
        raise ValueError('Frozen restoration/threshold state changed')
    write_json(output/'gate_diagnostic.json', dict(records=gate_records, TX_label_reads=0,
               F_or_TX_used_to_route=False, source_selection=False))
    write_json(saved/'complete.json', dict(status='SOURCE_PREDICTIONS_COMPLETE', count=len(ids),
               views=list(VIEWS), readouts=list(READOUTS), role='source_V', target_reads=0,
               TX_label_reads=0, state_updates=0, threshold_frozen_before_V=True))


def completed_predictions(output, ids):
    saved = Path(output)/'prediction'
    complete = json.loads((saved/'complete.json').read_text(encoding='utf-8'))
    expected = dict(status='SOURCE_PREDICTIONS_COMPLETE', count=len(ids), views=list(VIEWS),
                    readouts=list(READOUTS), role='source_V', target_reads=0, TX_label_reads=0,
                    state_updates=0, threshold_frozen_before_V=True)
    if any(complete.get(k) != v for k, v in expected.items()):
        raise ValueError('Complete fixed source predictions required before labels')
    result = {}
    for view in VIEWS:
        with np.load(saved/(view+'.npz'), allow_pickle=False) as packet:
            if set(packet.files) != {'ids', *READOUTS, 'preserve', 'change'} or not np.array_equal(packet['ids'], ids):
                raise ValueError('Source physical predictions differ')
            result[view] = {}
            for name in READOUTS:
                values = packet[name]
                if (values.shape != (len(ids),) or values.dtype.kind not in 'iu'
                        or (values < 0).any() or (values >= 6).any()):
                    raise ValueError('Malformed frozen default-head prediction')
                result[view][name] = values.copy()
    return result


def score(reference, source_cache, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scoring required')
    ids, source, indices = source_reference(reference, source_cache)
    predictions = completed_predictions(output, ids)
    labels = np.load(Path(source_cache)/'V/labels.npy', allow_pickle=False)[indices]
    clean = np.load(Path(reference)/'clean.npy', allow_pickle=False)
    energy = np.maximum(np.square(clean).mean((1, 2)), 1e-4)
    waveform = []
    for view in VIEWS:
        raw = np.load(Path(reference)/(view+'.npy'), allow_pickle=False)
        restored = np.load(Path(output)/'prediction'/(view+'_restored.npy'), allow_pickle=False)
        with np.load(Path(output)/'prediction'/(view+'.npz'), allow_pickle=False) as packet:
            mask = packet['preserve']
        if (restored.shape != clean.shape or not np.isfinite(restored).all()
                or mask.shape != (len(ids),) or mask.dtype.kind != 'b'):
            raise ValueError('Malformed preserved waveform or per-packet decision')
        preserved = np.where(mask[:, None, None], raw, restored)
        errors = {name: float((np.square(value-clean).mean((1, 2))/energy).mean())
                  for name, value in zip(READOUTS, (raw, restored, preserved))}
        waveform.append(dict(view=view, preserved=int(mask.sum()), count=len(ids), NMSE=errors))
    records = []
    for view in VIEWS:
        for name in READOUTS:
            after = predictions[view][name]
            for baseline_name in ('identity', 'original_R03'):
                baseline = predictions[view][baseline_name]
                groups = [('overall', 'all', np.ones(len(ids), bool))]
                for dimension, values in (('TX', labels), ('RX', source.rx[indices]), ('day', source.day[indices])):
                    groups.extend((dimension, str(value), values == value) for value in np.unique(values))
                for dimension, value, mask in groups:
                    metrics = classification_statistics(labels[mask], after[mask], baseline[mask])
                    records.append(dict(view=view, readout=name, baseline=baseline_name,
                                        dimension=dimension, value=value, **metrics))
    write_json(Path(output)/'metrics.json', dict(status='ANALYZED', source_only=True,
               truth_last=True, reader_run=REFERENCE_RUN, reader_commit=REFERENCE_COMMIT,
               count=len(ids), records=records, waveform=waveform, target_reads=0, head_fit_updates=0,
               limitation='Single source panel/seed, no target improvement claim; no candidate selection'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('predict', 'score'))
    for name in ('reference', 'source-cache', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('reader-folder', 'restorer-checkpoint'):
        p.add_argument('--'+name, type=Path)
    a = p.parse_args()
    if a.mode == 'predict':
        if a.reader_folder is None or a.restorer_checkpoint is None:
            p.error('Frozen F50 and original R03 checkpoints required')
        predict(a.reference, a.source_cache, a.reader_folder, a.restorer_checkpoint, a.output)
    else:
        score(a.reference, a.source_cache, a.output)


if __name__ == '__main__':
    main()
