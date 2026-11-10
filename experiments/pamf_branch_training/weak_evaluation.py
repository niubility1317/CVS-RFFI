"""Fixed weak-view predictions followed by separate truth-side scoring."""
from pathlib import Path
import numpy as np

from experiments.cric_dura_comparison.io import read, tensor, write
from .plan import WEAK_VIEWS


def index(root):
    root = Path(root)
    manifest = read(root / 'manifest.json')
    if (manifest['status'] != 'VALIDATED_ONCE' or manifest['count'] != 168000
            or manifest['scenes'] != list(WEAK_VIEWS) or manifest['truth_read']
            or manifest['query_fit']):
        raise ValueError('Weak-view data contract differs')
    with np.load(root / 'index.npz', allow_pickle=False) as z:
        ids = z['ids'].copy()
    with np.load(manifest['physical_ids_ref'], allow_pickle=False) as z:
        if not np.array_equal(ids, z['ids']):
            raise ValueError('Weak-view physical IDs differ from original capsule')
    if len(ids) != 168000 or len(set(ids.tolist())) != len(ids):
        raise ValueError('Weak-view physical ID coverage differs')
    return ids


def predict(c):
    import torch
    from experiments.cric_dura_comparison.evaluation import reject_truth
    from experiments.pamf_paper_completion.model import load_reader
    from experiments.pamf_channel_pl.restored_source import snapshot, unchanged
    reject_truth(c)
    ids = index(c['weak_root'])
    reader, _ = load_reader(c['source_output'], read(c['source_contract']), c['paper_row'], 'cuda:0')
    before = snapshot(reader)
    out = Path(c['output_root'])
    out.mkdir(parents=True, exist_ok=False)
    values = {}
    for view in WEAK_VIEWS:
        array = np.load(Path(c['weak_root']) / (view + '.npy'), mmap_mode='r', allow_pickle=False)
        if array.shape != (len(ids), 2, 256):
            raise ValueError('Weak-view IQ shape differs')
        parts = []
        with torch.inference_mode():
            for begin in range(0, len(ids), 128):
                x = tensor(np.ascontiguousarray(array[begin:begin + 128], np.float32), 'cuda:0')
                logits = reader.classify_features(reader.features(x))
                if logits.shape != (len(x), 6) or not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError('Invalid weak-view logits')
                parts.extend(logits.argmax(1).cpu().tolist())
        values[view] = np.asarray(parts, dtype=np.int64)
        print('WEAK_PREDICTION', c['row_id'], view, len(ids), flush=True)
    if not unchanged(reader, before):
        raise ValueError('Frozen model state changed')
    np.savez(out / 'predictions.npz', ids=ids, **values)
    write(out / 'complete.json', dict(status='ALL_WEAK_PREDICTIONS_COMPLETE', row_id=c['row_id'],
          count=len(ids), views=list(WEAK_VIEWS), truth_read=False, query_fit=False))


def score(c):
    from experiments.cric_dura_comparison.evaluation import handle, metrics
    if any(c.get(k) for k in ('source_output', 'source_cache', 'frozen_sources', 'restored_caches')):
        raise ValueError('Independent scorer must not receive source/model handles')
    ids = index(c['weak_root'])
    prediction = Path(c['prediction_output'])
    done = read(prediction / 'complete.json')
    if (done['status'] != 'ALL_WEAK_PREDICTIONS_COMPLETE' or done['row_id'] != c['row_id']
            or done['truth_read'] or done['query_fit'] or done['views'] != list(WEAK_VIEWS)):
        raise ValueError('Truth closed: weak predictions incomplete')
    with np.load(prediction / 'predictions.npz', allow_pickle=False) as z:
        if set(z.files) != {'ids', *WEAK_VIEWS} or not np.array_equal(ids, z['ids']):
            raise ValueError('Truth closed: weak prediction IDs differ')
        values = {view: z[view].copy() for view in WEAK_VIEWS}
    for p in values.values():
        if p.shape != ids.shape or p.dtype.kind not in 'iu' or p.min() < 0 or p.max() >= 6:
            raise ValueError('Truth closed: weak decisions invalid')
    _, both, _ = handle(c['double_cross_views_root'], c['classes'])
    mask = np.isin(ids, both)
    if len(both) != 42000 or int(mask.sum()) != 42000:
        raise ValueError('Truth closed: double-cross population mismatch')
    truth = read(c['truth'])
    y = np.asarray([truth[str(i)]['label'] for i in ids], dtype=np.int64)
    rx = np.asarray([str(truth[str(i)]['receiver']) for i in ids])
    day = np.asarray([str(truth[str(i)]['day']) for i in ids])
    if y.min() < 0 or y.max() >= 6:
        raise ValueError('Truth class map differs')
    result = []
    for population, selection in (('combined_target', np.ones(len(ids), bool)),
                                  ('cross_rx', ~mask), ('cross_both', mask)):
        groups = [('overall', 'ALL', selection)]
        for dim, group in (('receiver', rx), ('date', day), ('transmitter', y)):
            groups.extend((dim, str(key), selection & (group == key)) for key in np.unique(group)
                          if np.any(selection & (group == key)))
        for view, predicted in values.items():
            for dim, name, chosen in groups:
                result.append(dict(test=population, readout='raw', view=view, dimension=dim,
                                   stratum=name, **metrics(y[chosen], predicted[chosen], 6)))
    out = Path(c['output_root'])
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'scores.json', dict(status='SCORED', row_id=c['row_id'], results=result,
                                   truth_last=True, target_feedback_forbidden=True))
    write(out / 'complete.json', dict(status='SCORED_WEAK_COMPLETE', row_id=c['row_id'],
                                     count=len(ids), views=list(WEAK_VIEWS), truth_last=True))
