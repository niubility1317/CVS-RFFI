"""Copy this run's trusted, already-frozen class IDs into safe numeric archives.

This is a technical recovery for the known NumPy ABI failure. The object arrays
are from this exact run's own recorded outputs, not external input. No IQ, truth,
model, or new prediction is read or computed here.
"""
import json
import numbers
from pathlib import Path
import sys

import numpy as np

project = Path('/home/szu2070436088/2510044040/CV-SincNet')
run_id = '20261128-phase1-shared-confidence-route-s2026112925-r01'
source = project / 'runs' / run_id
target = project / 'runs' / (run_id + '-numeric-score-recovery-r01')
release = project / 'releases' / (run_id + '-release')
views = ('clean', 'practical_low_urban')

if len(sys.argv) != 1:
    raise ValueError('This recovery is fixed to its own recorded run')
if target.exists():
    raise FileExistsError(target)
prepared = {}
for dataset, count, classes in (('lora', 60, 10), ('wisig', 360, 6)):
    folder = source / dataset
    mark = json.loads((folder / 'screen-frozen.json').read_text(encoding='utf-8'))
    expected_modes = (['direct', 'lora_native_r17', 'real_gain', 'lora_r69_adapt', 'fusion']
                      if dataset == 'lora' else ['direct', 'wisig_r17', 'real_gain', 'dura_v1', 'fusion'])
    if (mark['status'], mark['views'], mark['modes'], mark['count'], mark['source_truth_reads'], mark['target_reads']) != (
            'PREDICTIONS_FROZEN', list(views), expected_modes, count, 0, 0):
        raise ValueError('Expected the complete prediction-only frozen marker')
    # Trusted producer is the immutable original release of this exact run.
    with np.load(folder / 'screen.npz', allow_pickle=True) as archive:
        ids = archive['ids'].copy()
        if ids.dtype.kind not in 'US' or len(ids) != count or len(set(ids.tolist())) != count:
            raise ValueError('Unexpected source physical IDs')
        arrays = {}
        for view in views:
            values = archive[view]
            if values.shape != (5, count) or values.dtype != object:
                raise ValueError('Expected the recorded object-array failure')
            flat = values.ravel().tolist()
            if not all(isinstance(x, numbers.Integral) and 0 <= x < classes for x in flat):
                raise ValueError('Class predictions must be in-range integer values')
            numeric = np.asarray(flat, dtype=np.int16).reshape(values.shape)
            if numeric.tolist() != values.tolist():
                raise ValueError('Prediction value changed during conversion')
            arrays[view] = numeric
        prepared[dataset] = (ids, arrays, mark)

target.mkdir()
for dataset, (ids, arrays, mark) in prepared.items():
    folder = target / dataset
    folder.mkdir()
    with (folder / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=ids, **arrays)
    (folder / 'screen-frozen.json').write_text(json.dumps(mark, indent=2), encoding='utf-8')
    with np.load(folder / 'screen.npz', allow_pickle=False) as check:
        if not np.array_equal(check['ids'], ids) or any(not np.array_equal(check[v], arrays[v]) for v in views):
            raise ValueError('Numeric archive readback mismatch')
runtime = json.loads((release / 'runtime.json').read_text(encoding='utf-8'))
runtime['run_root'] = str(target)
(target / 'runtime.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
print('NUMERIC_RECOVERY_FROZEN', target, 'two datasets; no truth read', flush=True)
