"""Read-only paired source-screen audit against the pre-existing R69 baseline."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
OLD = '20261127-phase1-lora-wisig-packet-dura-s2026113021-r02'
REFERENCE = '20261125-phase1-wisig-native-fir-source-m4-r01-release'
VIEWS = ('clean', 'practical_low_urban')
CASES = {
    '20261128-phase1-wisig-r17-realgain-source-s2026112865-r01': (
        3, ('direct', 'wisig_r17', 'real_gain', 'prior_fusion', 'fusion'), 2026112868, 2026112873),
    '20261128-phase1-wisig-r17-joint-realgain-source-s2026112869-r01': (
        4, ('direct', 'wisig_r17', 'real_gain', 'first_fusion', 'second_fusion', 'joint_fusion'),
        2026112872, 2026112877),
    '20261128-phase1-wisig-r17-spectral-gain-s2026112877-r01': (
        6, ('direct', 'wisig_r17', 'real_gain', 'prior_fusion', 'fusion'),
        2026112881, 2026112886),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', choices=tuple(CASES))
RUN = parser.parse_args().run
RX, MODES, HALF_SEED, BOOTSTRAP_SEED = CASES[RUN]

def read(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)

folder = ROOT / 'runs' / RUN
mark = read(folder / 'screen-frozen.json')
assert (mark['status'], mark['modes'], mark['views'], mark['count'],
        mark['source_RX'], mark['source_truth_reads'], mark['target_reads']) == (
            'PREDICTIONS_FROZEN', list(MODES), list(VIEWS), 360, RX, 0, 0)
archived = ROOT / 'runs' / OLD / 'wisig'
old_mark = read(archived / 'frozen.json')
assert (old_mark['status'], old_mark['modes'], old_mark['source_truth_reads'],
        old_mark['target_reads']) == (
            'PREDICTIONS_FROZEN', ['direct', 'dura', 'packet_unet', 'hybrid'], 0, 0)
with np.load(folder / 'screen.npz', allow_pickle=False) as data:
    ids = data['ids'].copy()
    candidate = {view: data[view].copy() for view in VIEWS}
assert len(ids) == 360 and len(set(ids.tolist())) == 360
with np.load(archived / 'predictions.npz', allow_pickle=False) as data:
    old_ids = data['ids'].copy()
    old = {view: data[view].copy() for view in VIEWS}
assert len(set(old_ids.tolist())) == len(old_ids)
positions = {identifier: index for index, identifier in enumerate(old_ids.tolist())}
indices = np.asarray([positions[x] for x in ids.tolist()])
for view in VIEWS:
    assert candidate[view].shape == (len(MODES), len(ids))
    assert old[view].shape == (4, len(old_ids))
    if not np.array_equal(candidate[view][0], old[view][0, indices]):
        raise ValueError(f'Historical and current direct predictions differ: {view}')

# Join source-V truth only after both prediction artifacts and IDs are checked.
from experiments.cric_dura_comparison.bootstrap import activate
from experiments.pamf_lora_main.dura_real_gain_fit import activate_wisig_archive
rt = read(ROOT / 'releases' / (RUN + '-release') / 'runtime.json')
activate(rt)
ref = ROOT / 'releases' / REFERENCE
activate_wisig_archive(ref)
original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
_, _, source, _, all_ids, rx, source_positions = original.core.baseline.handles(read(ref / 'runtime.json'))
assert len(all_ids) == len(rx) and set(ids.tolist()) <= set(all_ids[rx == RX].tolist())
truth_all = np.load(source / 'V' / 'labels.npy', allow_pickle=False)[source_positions]
truth_map = dict(zip(all_ids.tolist(), truth_all.tolist(), strict=True))
truth = np.asarray([truth_map[x] for x in ids.tolist()], dtype=np.int64)
permutation = np.random.default_rng(HALF_SEED).permutation(len(ids))
results = {}
for view in VIEWS:
    new = candidate[view][-1] == truth
    reference = old[view][1, indices] == truth
    delta = new.astype(np.int16) - reference.astype(np.int16)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(delta), size=(2000, len(delta)))
    results[view] = dict(n=len(ids), fusion_correct=int(new.sum()), r69_correct=int(reference.sum()),
        repaired=int((new & ~reference).sum()), harmed=int((~new & reference).sum()),
        net=int(delta.sum()), half_nets=[int(delta[x].sum()) for x in np.array_split(permutation, 2)],
        ci95=np.quantile(delta[draws].sum(1), (.025, .975)).tolist())
print(json.dumps(dict(status='READ_ONLY_SOURCE_AUDIT', run=RUN, reference=OLD, receiver=RX,
    truth_last=True, direct_predictions_match=True, views=results), ensure_ascii=False))
