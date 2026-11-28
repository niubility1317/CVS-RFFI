"""Independent paired source-V analysis after both fixed experiments finish."""
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from experiments.cric_dura_comparison.bootstrap import activate
from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.dura_real_gain_fit import LORA_GAIN, activate_wisig_archive
from experiments.pamf_lora_main.dura_real_gain_screen import paired
from experiments.pamf_lora_main.shared_dura_realgain_run import REFERENCE, VIEWS

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise ValueError('Paired analysis runs only in an independent CPU process')

project = Path('/home/szu2070436088/2510044040/CV-SincNet')
if len(sys.argv) == 1:
    teacher = '20261128-phase1-shared-teacher-selector-s2026112937-r01'
    control = '20261128-phase1-shared-matched-mse-s2026112937-r01'
elif len(sys.argv) == 3:
    teacher, control = sys.argv[1:]
else:
    raise ValueError('Expected either no run IDs or a teacher and matched control run ID')
release = project / 'releases' / (teacher + '-release')
rt = read(release / 'runtime.json')
out = project / 'runs' / (teacher + '-paired-mse-analysis-r01')
if out.exists():
    raise FileExistsError(out)

# Freeze and ID checks for both runs and both datasets precede all truth reads.
frozen = {}
for dataset, count in (('lora', 60), ('wisig', 360)):
    rows = []
    for run in (teacher, control):
        folder = project / 'runs' / run / dataset
        mark = read(folder / 'screen-frozen.json')
        if (mark['status'], mark['views'], mark['count'], mark['source_truth_reads'], mark['target_reads']) != (
                'PREDICTIONS_FROZEN', list(VIEWS), count, 0, 0):
            raise ValueError('Both complete source prediction matrices must already be frozen')
        with np.load(folder / 'screen.npz', allow_pickle=False) as archive:
            ids = archive['ids'].copy()
            values = {view: archive[view].copy() for view in VIEWS}
        if len(set(ids.tolist())) != count or any(x.shape != (5, count) for x in values.values()):
            raise ValueError('Invalid source prediction matrix')
        rows.append((ids, values, mark))
    if not np.array_equal(rows[0][0], rows[1][0]) or rows[0][2]['modes'] != rows[1][2]['modes']:
        raise ValueError('Experiments must have identical packet IDs and controls')
    if any(not np.array_equal(rows[0][1][v][:-1], rows[1][1][v][:-1]) for v in VIEWS):
        raise ValueError('Experiments must have identical frozen controls')
    frozen[dataset] = rows

results = {}
for dataset, count in (('lora', 60), ('wisig', 360)):
    rows = frozen[dataset]
    ids = rows[0][0]
    if dataset == 'lora':
        old = read(project / 'runs' / LORA_GAIN / 'frequency_inphase_native' / 'fit-inputs.json')
        with np.load(Path(old['source_root']) / 'V_metadata.npz', allow_pickle=False) as meta:
            lookup = dict(zip(meta['ids'].tolist(), meta['y'].tolist(), strict=True))
    else:
        activate(rt)
        ref = project / 'releases' / REFERENCE
        activate_wisig_archive(ref)
        original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
        _, _, source, _, all_ids, _, positions = original.core.baseline.handles(read(ref / 'runtime.json'))
        labels = np.load(source / 'V' / 'labels.npy', allow_pickle=False)[positions]
        lookup = dict(zip(all_ids.tolist(), labels.tolist(), strict=True))
    truth = np.asarray([lookup[item] for item in ids.tolist()], dtype=np.int64)
    permutation = np.random.default_rng(2026112852).permutation(count)
    result = {}
    for view in VIEWS:
        a, b = rows[0][1][view][-1], rows[1][1][view][-1]
        pair = paired(a, b, truth, permutation, 1 if dataset == 'lora' else 2)
        groups = {str(int(t)): dict(n=int((truth == t).sum()),
                  teacher_correct=int(((truth == t) & (a == truth)).sum()),
                  mse_correct=int(((truth == t) & (b == truth)).sum())) for t in np.unique(truth)}
        result[view] = dict(teacher_correct=int((a == truth).sum()), mse_correct=int((b == truth).sum()),
                            paired=pair, source_tx=groups)
    results[dataset] = result

out.mkdir()
(out / 'paired.json').write_text(json.dumps(dict(status='SOURCE_V_PAIRED_ANALYSIS', truth_last=True,
    teacher=teacher, control=control, exposure='repeatedly exposed source V', target_reads=0,
    caveat='Same seed/data generator and budget; different GPUs may yield numerical differences',
    results=results), indent=2), encoding='utf-8')
print('SOURCE_V_PAIRED_ANALYSIS', out / 'paired.json', flush=True)
