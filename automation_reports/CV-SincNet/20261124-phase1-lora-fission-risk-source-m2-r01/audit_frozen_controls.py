"""Compare frozen endpoint predictions against the same physical historical panel."""

import json
from pathlib import Path

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.private_direction_risk_source import MODES as OLD_MODES, RUN as OLD
from experiments.pamf_lora_main.data import VIEWS
from experiments.pamf_lora_main.fission_risk_source import MODES


root = Path(read(Path('runtime.json'))['run_root'])
if read(root / 'state.json')['status'] != 'SOURCE_SCORED':
    raise ValueError('Expected terminal current predictions')
controls = (('direct', 'direct'), ('reference', 'reference'), ('rank_half', 'rank_parent_half'),
            ('direction_half', 'direction_risk_half'))
mismatches = {}
for route in ('full', 'residual'):
    ids, prediction, _ = read_frozen(root, route, MODES)
    old_ids, old, _ = read_frozen(root.parent / OLD, route, OLD_MODES)
    if ids != old_ids:
        raise ValueError('Control audit requires matching physical identifiers and order')
    mismatches[route] = {view: {key: int((prediction[view][MODES.index(key)]
                                      != old[view][OLD_MODES.index(parent_key)]).sum())
                              for key, parent_key in controls}
                         for view in VIEWS}
print(json.dumps(dict(status='CHECKED', n=900, target_reads=0, TX_reads=0,
                      history=OLD, mismatches=mismatches)))
