"""Descriptive, truth-last RX6 agreement partitions for the frozen WiSig screen."""
import importlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN = '20261128-phase1-wisig-r17-spectral-gain-s2026112877-r01'
OLD = '20261127-phase1-lora-wisig-packet-dura-s2026113021-r02'
REFERENCE = '20261125-phase1-wisig-native-fir-source-m4-r01-release'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


folder = ROOT / 'runs' / RUN
mark = read(folder / 'screen-frozen.json')
assert (mark['status'], mark['source_RX'], mark['count'], mark['source_truth_reads'], mark['target_reads']) == (
    'PREDICTIONS_FROZEN', 6, 360, 0, 0)
assert mark['modes'] == ['direct', 'wisig_r17', 'real_gain', 'prior_fusion', 'fusion']
archived = ROOT / 'runs' / OLD / 'wisig'
old_mark = read(archived / 'frozen.json')
assert (old_mark['status'], old_mark['source_truth_reads'], old_mark['target_reads']) == (
    'PREDICTIONS_FROZEN', 0, 0)
with np.load(folder / 'screen.npz', allow_pickle=False) as data:
    ids, predictions = data['ids'].copy(), data['practical_low_urban'].copy()
with np.load(archived / 'predictions.npz', allow_pickle=False) as data:
    old_ids, old = data['ids'].copy(), data['practical_low_urban'].copy()
assert predictions.shape == (5, 360) and len(set(ids.tolist())) == 360
assert old.shape == (4, len(old_ids)) and len(set(old_ids.tolist())) == len(old_ids)
lookup = {key: i for i, key in enumerate(old_ids.tolist())}
indices = np.asarray([lookup[key] for key in ids.tolist()])
if not np.array_equal(predictions[0], old[0, indices]):
    raise ValueError('Historical and candidate direct predictions differ')

# Truth is read only after validating both frozen predictions and physical IDs.
from experiments.cric_dura_comparison.bootstrap import activate
from experiments.pamf_lora_main.dura_real_gain_fit import activate_wisig_archive
activate(read(ROOT / 'releases' / (RUN + '-release') / 'runtime.json'))
ref = ROOT / 'releases' / REFERENCE
activate_wisig_archive(ref)
original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
_, _, source, _, all_ids, rx, positions = original.core.baseline.handles(read(ref / 'runtime.json'))
assert set(ids.tolist()) <= set(all_ids[rx == 6].tolist())
labels = np.load(source / 'V' / 'labels.npy', allow_pickle=False)[positions]
truth_map = dict(zip(all_ids.tolist(), labels.tolist(), strict=True))
truth = np.asarray([truth_map[i] for i in ids.tolist()])
r17, r69, candidate = predictions[1], old[1, indices], predictions[-1]


def stats(mask):
    a, b, c = r17[mask], r69[mask], candidate[mask]
    y = truth[mask]
    return dict(n=int(mask.sum()), r17_correct=int((a == y).sum()), r69_correct=int((b == y).sum()),
                fusion_correct=int((c == y).sum()), repaired_vs_r17=int(((c == y) & (a != y)).sum()),
                harmed_vs_r17=int(((c != y) & (a == y)).sum()))


groups = {
    'r17_r69_same_label': stats(r17 == r69),
    'r17_r69_different_label': stats(r17 != r69),
    'r17_correct_r69_wrong': stats((r17 == truth) & (r69 != truth)),
    'r17_wrong_r69_correct': stats((r17 != truth) & (r69 == truth)),
    'both_wrong': stats((r17 != truth) & (r69 != truth)),
}
by_class = {str(int(y)): stats(truth == y) for y in np.unique(truth)}
assert sum(x['n'] for x in by_class.values()) == len(ids)
assert sum(x['fusion_correct'] for x in by_class.values()) == int((candidate == truth).sum())
print(json.dumps(dict(status='POSTHOC_SOURCE_V_DESCRIPTION', run=RUN, receiver=6,
                      observation='practical_low_urban', groups=groups, classes=by_class,
                      target_reads=0, use='diagnostic_only_no_deployment_gate'), ensure_ascii=False))
