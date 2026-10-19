"""Fixed source-V comparison, with labels opened only by a separate scorer.

This complements physical-operator research. It neither selects a restorer nor
reads target IQ/truth, and never changes the original F or either restorer.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .frozen import load_reader, REFERENCE_RUN, REFERENCE_COMMIT
from experiments.pamf_restoration.data import IQSource, SCENES, write_json
from experiments.pamf_restoration.frozen import classification_statistics
from experiments.pamf_restoration.runner import array_tensor

VIEWS = ('clean', *SCENES)
READOUTS = ('identity', 'coarse', 'residual')


def source_reference(reference, source_cache):
    """Only the existing source-V physical IDs; no label files are opened."""
    reference = Path(reference)
    source = IQSource(source_cache, 'V')
    ids = np.load(reference/'physical_ids.npy', allow_pickle=False)
    if (ids.ndim != 1 or len(ids) != 1800 or len(set(ids.tolist())) != len(ids)
            or not set(ids.tolist()) <= set(source.ids.tolist())):
        raise ValueError('Fixed 1800 source-V IDs required; target/training IDs forbidden')
    positions = {str(v): i for i, v in enumerate(source.ids)}
    index = np.array([positions[str(v)] for v in ids], dtype=np.int64)
    for name, expected in (('rx', source.rx[index]), ('day', source.day[index])):
        if not np.array_equal(np.load(reference/(name+'.npy'), allow_pickle=False), expected):
            raise ValueError('Source-V physical metadata differs')
    return ids, source, index


@torch.no_grad()
def predict(reference, source_cache, reader_folder, coarse_checkpoint,
            residual_checkpoint, output, batch_size=64):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or batch_size <= 0:
        raise ValueError('This fixed diagnostic uses CPU only and positive batches')
    ids, _, _ = source_reference(reference, source_cache)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve existing diagnostic output')
    from experiments.pamf_reception_restore.runner import load_checkpoint as load_coarse
    from experiments.pamf_residual_restore.runtime import load_checkpoint as load_residual
    reader = load_reader(reader_folder, source_cache)
    coarse, c = load_coarse(coarse_checkpoint, 'tf_unet_mixed', 'cpu')
    residual, r = load_residual(residual_checkpoint, 'residual_deterministic', 'cpu')
    if any(p['epoch'] != 20 or p['updates'] != 4940 for p in (c, r)):
        raise ValueError('Fixed final E20 coarse/corrector checkpoints required')
    coarse.requires_grad_(False)
    residual.requires_grad_(False)
    torch.set_num_threads(2)
    output.mkdir(parents=True)
    write_json(output/'execution.json', dict(pid=os.getpid(), cwd=str(Path.cwd()),
               argv=__import__('sys').argv, device='cpu', threads=2, role='source_V',
               reader_run=REFERENCE_RUN, reader_commit=REFERENCE_COMMIT,
               views=list(VIEWS), readouts=list(READOUTS), count=len(ids),
               optimizer_updates=0, head_fit_updates=0, TX_label_reads=0, target_reads=0))
    total_started = time.perf_counter()
    for view in VIEWS:
        observed = np.load(Path(reference)/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        if observed.shape != (len(ids), 2, 256):
            raise ValueError('Malformed fixed source view')
        decisions = {name: [] for name in READOUTS}
        started = time.perf_counter()
        for offset in range(0, len(ids), batch_size):
            y = array_tensor(observed[offset:offset+batch_size], 'cpu')
            if not torch.isfinite(y).all():
                raise ValueError('Nonfinite source IQ')
            for name, x in (('identity', y), ('coarse', coarse(y)), ('residual', residual(y))):
                logits = reader(x)
                if logits.shape != (len(y), 6) or not torch.isfinite(logits).all():
                    raise ValueError('Invalid frozen original default-head output')
                decisions[name].extend(logits.argmax(1).tolist())
        with (output/(view+'.npz')).open('xb') as stream:
            np.savez_compressed(stream, ids=ids, **{k: np.asarray(v, np.uint8) for k, v in decisions.items()})
        print(json.dumps(dict(event='SOURCE_VIEW_COMPLETE', view=view, count=len(ids),
                              seconds=time.perf_counter()-started)), flush=True)
    write_json(output/'complete.json', dict(status='SOURCE_PREDICTIONS_COMPLETE',
               role='source_V', views=list(VIEWS), readouts=list(READOUTS), count=len(ids),
               target_reads=0, TX_label_reads=0, state_updates=0,
               seconds=time.perf_counter()-total_started, source_selection=False))


def prediction_arrays(output, ids):
    """Validate all views before the scorer can open any source TX labels."""
    output = Path(output)
    complete = json.loads((output/'complete.json').read_text(encoding='utf-8'))
    expected = dict(status='SOURCE_PREDICTIONS_COMPLETE', role='source_V',
                    views=list(VIEWS), readouts=list(READOUTS), count=len(ids),
                    target_reads=0, TX_label_reads=0, state_updates=0)
    if any(complete.get(k) != v for k, v in expected.items()):
        raise ValueError('All fixed source predictions required before labels')
    result = {}
    for view in VIEWS:
        with np.load(output/(view+'.npz'), allow_pickle=False) as p:
            if set(p.files) != {'ids', *READOUTS} or not np.array_equal(p['ids'], ids):
                raise ValueError('Source prediction IDs/fields differ')
            result[view] = {}
            for name in READOUTS:
                values = p[name]
                if (values.shape != (len(ids),) or values.dtype.kind not in 'iu'
                        or (values < 0).any() or (values >= 6).any()):
                    raise ValueError('Malformed default-head decisions')
                result[view][name] = values.copy()
    return result


def score(reference, source_cache, output, metrics_path):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Separate CPU source scorer required')
    ids, source, index = source_reference(reference, source_cache)
    predictions = prediction_arrays(output, ids)
    # The only TX-label open in this module, after all views pass coverage.
    labels = np.load(Path(source_cache)/'V'/'labels.npy', allow_pickle=False)[index]
    records = []
    for view in VIEWS:
        for name in READOUTS:
            baseline = predictions[view]['identity']
            after = predictions[view][name]
            groups = [('overall', 'all', np.ones(len(ids), bool))]
            for dimension, values in (('TX', labels), ('RX', source.rx[index]), ('day', source.day[index])):
                groups += [(dimension, str(value), values == value) for value in np.unique(values)]
            for dimension, value, mask in groups:
                stat = classification_statistics(labels[mask], after[mask], baseline[mask])
                records.append(dict(view=view, readout=name, dimension=dimension, value=value, **stat))
    result = dict(status='ANALYZED', source_only=True, source_V_count=len(ids),
                  reader_run=REFERENCE_RUN, reader_commit=REFERENCE_COMMIT,
                  reader_seed=2026092701, original_default_head=True, head_fit_updates=0,
                  target_reads=0, truth_last=True, source_selection=False, records=records,
                  limitation='One fixed source panel and one F seed; not target improvement or a restorer selection')
    write_json(metrics_path, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('predict', 'score'))
    for name in ('reference', 'source-cache', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('reader-folder', 'coarse-checkpoint', 'residual-checkpoint', 'metrics'):
        p.add_argument('--'+name, type=Path)
    a = p.parse_args()
    if a.mode == 'predict':
        if any(v is None for v in (a.reader_folder, a.coarse_checkpoint, a.residual_checkpoint)):
            p.error('Prediction requires the three fixed checkpoints')
        predict(a.reference, a.source_cache, a.reader_folder, a.coarse_checkpoint,
                a.residual_checkpoint, a.output)
    else:
        if a.metrics is None:
            p.error('Score requires an exclusive metrics output path')
        score(a.reference, a.source_cache, a.output, a.metrics)


if __name__ == '__main__':
    main()
