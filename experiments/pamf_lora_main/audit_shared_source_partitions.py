"""Describe already-scored shared-method source predictions, with no target access."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
RUNS = ('20261128-phase1-shared-r17-realgain-s2026112893-r01',
        '20261128-phase1-shared-twoprior-realgain-s2026112901-r01',
        '20261128-phase1-shared-convex-realgain-s2026112905-r01',
        '20261128-phase1-shared-selector-realgain-s2026112909-r01')
LORA_GAIN = '20261128-phase1-lora-raw8192-frequency-quadrature-s2026113037-r01'
REFERENCE = '20261125-phase1-wisig-native-fir-source-m4-r01-release'
VIEWS = ('clean', 'practical_low_urban')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset', choices=('lora', 'wisig'))
parser.add_argument('run', choices=RUNS)
args = parser.parse_args()
dataset, RUN = args.dataset, args.run
folder = ROOT / 'runs' / RUN / dataset
mark = read(folder / 'screen-frozen.json')
count = 60 if dataset == 'lora' else 360
modes = (['direct', 'lora_native_r17', 'real_gain', 'fusion'] if dataset == 'lora' else
         ['direct', 'wisig_r17', 'real_gain', 'dura_v1', 'fusion'])
if dataset == 'lora' and RUN != RUNS[0]:
    modes.insert(-1, 'lora_r69_adapt')
if (mark['status'], mark['views'], mark['modes'], mark['count'],
        mark['source_truth_reads'], mark['target_reads']) != (
        'PREDICTIONS_FROZEN', list(VIEWS), modes, count, 0, 0):
    raise ValueError('Expected the complete fixed source prediction matrix')
with np.load(folder / 'screen.npz', allow_pickle=False) as data:
    ids, predictions = data['ids'].copy(), data['practical_low_urban'].copy()
if len(set(ids.tolist())) != count or predictions.shape != (len(modes), count):
    raise ValueError('Incorrect fixed IDs or prediction matrix')

# Source V truth is opened only after the complete frozen prediction checks.
if dataset == 'lora':
    old = read(ROOT / 'runs' / LORA_GAIN / 'frequency_inphase_native' / 'fit-inputs.json')
    with np.load(Path(old['source_root']) / 'V_metadata.npz', allow_pickle=False) as data:
        labels = dict(zip(data['ids'].tolist(), data['y'].tolist(), strict=True))
else:
    from experiments.cric_dura_comparison.bootstrap import activate
    from experiments.pamf_lora_main.dura_real_gain_fit import activate_wisig_archive
    activate(read(ROOT / 'releases' / (RUN + '-release') / 'runtime.json'))
    ref = ROOT / 'releases' / REFERENCE
    activate_wisig_archive(ref)
    original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
    _, _, source, _, all_ids, _, positions = original.core.baseline.handles(read(ref / 'runtime.json'))
    y = np.load(source / 'V' / 'labels.npy', allow_pickle=False)[positions]
    labels = dict(zip(all_ids.tolist(), y.tolist(), strict=True))
truth = np.asarray([labels[i] for i in ids.tolist()])
direct, r17, real, candidate = predictions[0], predictions[1], predictions[2], predictions[-1]


def summary(mask):
    correct = predictions[:, mask] == truth[mask]
    base, updated = correct[1], correct[-1]
    return dict(n=int(mask.sum()), correct=correct.sum(axis=1).tolist(),
                changed_from_r17=int((candidate[mask] != r17[mask]).sum()),
                repaired_vs_r17=int((updated & ~base).sum()), harmed_vs_r17=int((~updated & base).sum()))


groups = {'r17_same_real_gain': summary(r17 == real),
          'r17_differs_real_gain': summary(r17 != real),
          'candidate_differs_both': summary((candidate != r17) & (candidate != real)),
          'r17_correct': summary(r17 == truth), 'r17_wrong': summary(r17 != truth),
          'candidate_same_r17': summary(candidate == r17), 'candidate_changed_r17': summary(candidate != r17)}
classes = {str(int(value)): summary(truth == value) for value in np.unique(truth)}
if sum(row['n'] for row in classes.values()) != count:
    raise ValueError('Class partition does not cover the fixed panel')
real_correct, r17_correct = real == truth, r17 == truth
either = real_correct | r17_correct
order = np.random.default_rng(2026112852).permutation(count)
oracle = dict(both_correct=int((real_correct & r17_correct).sum()),
              only_real_gain=int((real_correct & ~r17_correct).sum()),
              only_r17=int((r17_correct & ~real_correct).sum()),
              neither_correct=int((~real_correct & ~r17_correct).sum()),
              oracle_correct=int(either.sum()),
              max_net_over_real_halves=[int((either[x].astype(np.int16)-real_correct[x].astype(np.int16)).sum())
                                        for x in np.array_split(order, 2)],
              max_net_over_r17_halves=[int((either[x].astype(np.int16)-r17_correct[x].astype(np.int16)).sum())
                                       for x in np.array_split(order, 2)],
              use='truth_dependent_oracle_not_deployable')
print(json.dumps(dict(status='POSTHOC_SOURCE_V_DESCRIPTION', run=RUN, dataset=dataset,
    modes=modes, observation='practical_low_urban', groups=groups, classes=classes, oracle=oracle,
    target_reads=0, use='diagnostic_only_no_deployment_gate'), ensure_ascii=False))
