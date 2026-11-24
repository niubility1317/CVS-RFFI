"""Verify the unchanged DURA controls before interpreting the projection study."""
import json
from pathlib import Path
import numpy as np
from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import VIEWS
from experiments.pamf_lora_main.dura_prior_width_source import MODES as OLD_MODES, RUN as OLD_RUN
from experiments.pamf_lora_main.dura_private_projection_source import MODES, RUN
from experiments.pamf_lora_main.spectral_unroll import ROUTES


def main():
    rt = read(Path('runtime.json'))
    root = Path(rt['run_root'])
    previous = root.parent / OLD_RUN
    names = ('direct','reference','wide_half')
    if root.name != RUN or read(root/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Requires completed registered source experiment')
    changes = {}
    for route in ROUTES:
        ids, found, _ = read_frozen(root, route, MODES)
        expected_ids, expected, _ = read_frozen(previous, route, OLD_MODES)
        if ids != expected_ids:
            raise ValueError('Mismatched physical source IDs')
        changes[route] = {
            view:{mode:int(np.count_nonzero(found[view][MODES.index(mode)] != expected[view][OLD_MODES.index(mode)]))
                  for mode in names} for view in VIEWS}
    with np.load(root/'clean-reserve-predictions.npz',allow_pickle=False) as found, \
            np.load(previous/'clean-reserve-predictions.npz',allow_pickle=False) as expected:
        if not np.array_equal(found['ids'],expected['ids']):
            raise ValueError('Mismatched physical clean reserve IDs')
        changes['clean_reserve'] = {mode:int(np.count_nonzero(
            found['rows'][MODES.index(mode)] != expected['rows'][OLD_MODES.index(mode)])) for mode in names}
    zero = all(d == 0 for route in ROUTES for row in changes[route].values() for d in row.values())
    zero = zero and all(d == 0 for d in changes['clean_reserve'].values())
    print(json.dumps(dict(status='VERIFIED' if zero else 'MISMATCH',
                          target_reads=0, changes=changes, parent=OLD_RUN),allow_nan=False))


if __name__ == '__main__':
    main()
