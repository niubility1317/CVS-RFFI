"""Paired direction/dose factorial on already scored, immutable source predictions."""

import json
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.channel_blind_source import panel
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.full_packet_views import source_contract
from experiments.pamf_lora_main.stage_splice_source import MODES as OLD_MODES, RUN as OLD
from experiments.pamf_lora_main.stage_dose_source import MODES


runtime = read(Path('runtime.json'))
root = Path(runtime['run_root'])
if read(root / 'state.json')['status'] != 'SOURCE_SCORED':
    raise ValueError('Both prediction routes and independent scoring must finish first')
checked = read(root / 'clean-complete-scores.json')
assert checked['summary']['n'] == 12000 and checked['truth_after_predictions'] and not checked['target_reads']
routes = ('full', 'residual')
current = {route: read_frozen(root, route, MODES) for route in routes}
assert current['full'][0] == current['residual'][0]
mapping = (('direct', 'direct'), ('reference', 'reference'), ('rank_half', 'rank_half'),
           ('direction_fine_half', 'direction_direction'), ('lag_fine_half', 'lag_lag'))
diff = {}
for route, (ids, prediction, _) in current.items():
    old_ids, old, _ = read_frozen(root.parent / OLD, route, OLD_MODES)
    assert ids == old_ids
    diff[route] = {view: {current_name: int((prediction[view][MODES.index(current_name)]
                            != old[view][OLD_MODES.index(old_name)]).sum()) for current_name, old_name in mapping}
                   for view in VIEWS}
source = Path(runtime['source_root'])
keys = np.asarray(source_contract(source)['role_ids']['V'])
positions = {value: i for i, value in enumerate(keys)}
selected = np.asarray([positions[v] for v in current['full'][0]])
with np.load(source / 'V_metadata.npz', allow_pickle=False) as truth:
    receivers = np.asarray(SOURCE)[truth['rx_i']]
    assert np.array_equal(selected, panel(receivers))
    assert np.array_equal(keys[selected], truth['ids'][selected])
    labels = truth['y'][selected]
rng = np.random.default_rng(2026116679)
strata = [np.flatnonzero(receivers[selected] == rx) for rx in SOURCE]
draws = np.concatenate([rng.choice(group, (1600, len(group)), replace=True) for group in strata], axis=1)
# Order is coarse/coarse, fine/fine, fine/coarse, coarse/fine.
cells = [MODES.index(name) for name in ('lag_coarse_half','lag_fine_half','lag_fine_at_coarse','lag_coarse_at_fine')]
coefficient = {'fine_direction': [-.5, .5, .5, -.5],
               'fine_dose': [-.5, .5, -.5, .5],
               'interaction': [1, 1, -1, -1]}
result = {}
for route, (_, prediction, _) in current.items():
    hits = np.stack([prediction[view][cells] == labels for view in VIEWS[1:]], axis=1)
    row = {}
    for key, definition in coefficient.items():
        effects = np.tensordot(definition, hits, axes=1)
        six = effects.sum(0)
        low = effects[-1]
        ci = lambda values: np.quantile(values[draws].sum(1), (.025, .975)).tolist()
        row[key] = dict(six=float(six.sum()), six_ci=ci(six), low=float(low.sum()), low_ci=ci(low),
                        per_view=effects.sum(1).tolist())
    result[route] = row
print(json.dumps(dict(status='CHECKED', n=len(selected), target_reads=0, F_reads=0,
                      source_V_truth_read_after_complete_predictions=True, bootstrap=1600,
                      endpoint_mismatches=diff, factorial=result)), flush=True)
