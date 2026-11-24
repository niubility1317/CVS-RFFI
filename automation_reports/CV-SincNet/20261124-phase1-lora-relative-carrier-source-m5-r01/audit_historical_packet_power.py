"""Audit historical endpoints and isolate output power under a frozen LoRa restorer."""
import json
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.carrier_coupled_source import MODES as OLD_MODES
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.full_packet_views import source_contract
from experiments.pamf_lora_main.relative_carrier_source import LEGACY_RUN, MODES


def main():
    runtime = read(Path('runtime.json'))
    root = Path(runtime['run_root'])
    source = Path(runtime['source_root'])
    history = root.parent/LEGACY_RUN
    if any(read(r/'state.json')['status']!='SOURCE_SCORED' for r in (root,history)):
        raise ValueError('Both source matrices must be closed')
    routes=('full','residual')
    current = {r:read_frozen(root,r,MODES) for r in routes}
    past = {r:read_frozen(history,r,OLD_MODES) for r in routes}
    ids=current['full'][0]
    if any(bundle[0]!=ids for bundle in (*current.values(),*past.values())):
        raise ValueError('Mismatched physical LoRa packets in historical comparator')
    controls=('direct','reference','rank_parent_half','coupled_gain_coarse','coupled_gain_full')
    differences={r:{view:{arm:int(np.sum(current[r][1][view][MODES.index(arm)]
                                          != past[r][1][view][OLD_MODES.index(arm)]))
                           for arm in controls} for view in VIEWS} for r in routes}
    if any(v for r in differences.values() for row in r.values() for v in row.values()):
        raise ValueError('An unchanged endpoint was classified differently')
    positions={key:i for i,key in enumerate(source_contract(source)['role_ids']['V'])}
    chosen=[positions[key] for key in ids]
    with np.load(source/'V_metadata.npz',allow_pickle=False) as known:
        if known['ids'][chosen].tolist()!=ids:
            raise ValueError('Truth order differs from already frozen physical IDs')
        labels=known['y'][chosen]
    per=len(ids)//len(SOURCE)
    rng=np.random.default_rng(2026114069)
    batch=np.concatenate([rng.integers(i*per,(i+1)*per,(4000,per)) for i in range(len(SOURCE))],axis=1)

    def stats(a,b):
        change=(a.astype(np.int8)-b).sum(axis=0)
        return dict(net=int(change.sum()), interval=np.quantile(change[batch].sum(axis=1),(.025,.975)).tolist(),
                    repaired=int((a&~b).sum()),damaged=int((~a&b).sum()),
                    by_rx=change.reshape(len(SOURCE),per).sum(1).tolist())

    results = {}
    for r in routes:
        then = np.stack([past[r][1][v][OLD_MODES.index('coupled_gain_half')]==labels for v in VIEWS[1:]])
        results[r] = {}
        for arm in ('coupled_gain_half','relative_coupled_half'):
            now = np.stack([current[r][1][v][MODES.index(arm)]==labels for v in VIEWS[1:]])
            results[r][arm]=dict(six=stats(now,then),low=stats(now[-1:],then[-1:]))
    print(json.dumps(dict(status='VERIFIED',n=len(ids),targets_read=0,truth_after_predictions=True,
                          prior=LEGACY_RUN,unchanged_endpoints=differences,against_original_half=results)))


if __name__=='__main__':
    main()
