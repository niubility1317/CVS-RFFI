"""Measure isolated same-packet coverage of fixed LoRa recovery proposals."""

import argparse
import json
from pathlib import Path

import numpy as np


PARENTS = {
    'wiener': ('20261123-phase1-lora-local-wiener-source-m2-r01', 'ffbfb47e2392c3b5c447ead6c92a98ff63754c1b',
               ('diagonal_half','covariance_half','covariance_full')),
    'masked_prior': ('20261123-phase1-lora-spectral-block-source-m4-r01', '50c16f57afb0696c0508d9d0f09886f46bc906eb',
                     ('unmasked_coarse','unmasked_full','unmasked_half','masked_coarse','masked_full','masked_half')),
    'repeat_protection': ('20261123-phase1-lora-repeat-protection-source-m4-r01', '2004887db268dd8e4673e3205331af96d425a773',
                          ('scalar_coarse','scalar_full','scalar_half','protected_coarse','protected_full','protected_half')),
    'empirical': ('20261124-phase1-lora-empirical-chirp-source-m2-r01', '1b71d4ddb24c216f95cb3884f2e6d004178e2c4d',
                  ('all_half','cross_half','agreement_half','agreement_full')),
}
VIEWS = ('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban',
         'practical_mid_urban','practical_low_urban')
ROUTES = ('full','residual')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def frozen(project):
    source = project/'releases'/'20261122-phase1-lora-full-aligned-data-r01-release'/'data'/'source'
    contract = read(source/'contract.json')
    if contract['protocol']!='lora_full_packets_v1' or not contract['source_only']:
        raise ValueError('Expected the registered source-only panel')
    common_ids = None
    result = {route:{} for route in ROUTES}
    for name, (run, commit, accepted) in PARENTS.items():
        root = project/'runs'/run
        config = read(project/'releases'/f'{run}-release'/'runtime.json')
        if (config['code_commit'], config['run_root'], config['source_root'])!=(commit,str(root),str(source)):
            raise ValueError('Incompatible registered prediction source')
        if read(root/'state.json')['status']!='SOURCE_SCORED':
            raise ValueError('Both routes of every parent must be complete')
        for route in ROUTES:
            row = read(root/f'{route}-predictions.json')
            ids = row['ids']
            predictions = np.stack([np.asarray(row['rows'][view]) for view in VIEWS])
            if (row['modes']!=['direct','reference',*accepted] or set(row['rows'])!=set(VIEWS)
                    or len(ids)!=900 or len(set(ids))!=900 or predictions.shape!=(7,len(accepted)+2,900)
                    or predictions.dtype.kind not in 'iu' or (predictions<0).any() or (predictions>9).any()):
                raise ValueError('Expected fixed observable-only proposal responses')
            common_ids = ids if common_ids is None else common_ids
            if ids!=common_ids:
                raise ValueError('Source physical panel changed')
            other = next(iter(result[route].values()),None)
            if other is not None and not np.array_equal(other[:,:2],predictions[:,:2]):
                raise ValueError('Freeze baselines differ across same-route runs')
            result[route][name] = predictions
    positions = {pid:i for i,pid in enumerate(contract['role_ids']['V'])}
    chosen = np.asarray([positions[pid] for pid in common_ids])
    return source, contract, common_ids, chosen, result


def aggregate(routes, truth, receiver):
    rng = np.random.default_rng(2026112407)
    groups = [np.flatnonzero(receiver==j) for j in np.unique(receiver)]
    draw = np.concatenate([rows[rng.integers(len(rows),size=(1600,len(rows)))] for rows in groups],axis=1)
    summary = {}
    for route, candidate in routes.items():
        base = next(iter(candidate.values()))[:,:2] == truth
        direct, reference = base[:,0],base[:,1]
        hit = {name:(decision[:,2:]==truth).any(1) for name,decision in candidate.items()}
        union = direct | np.stack([*hit.values()]).any(0)
        delta = (union.astype(int)-reference.astype(int))[1:].sum(0)
        summary[route] = dict(direct=direct.sum(1).tolist(),reference=reference.sum(1).tolist(),
            coverage=union.sum(1).tolist(), repair=(union & ~reference).sum(1).tolist(),
            harm=(~union & reference).sum(1).tolist(),
            coverage_by_receiver=[union[:,rows].sum(1).tolist() for rows in groups],
            six_ci_vs_reference=np.quantile(delta[draw].sum(1),(.025,.975)).tolist(),
            unique={name:(union & ~(direct | np.stack([v for k,v in hit.items() if k!=name]).any(0))).sum(1).tolist()
                    for name in hit})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project',type=Path)
    args = parser.parse_args()
    source, contract, ids, chosen, routes = frozen(args.project)
    with np.load(source/'V_metadata.npz',allow_pickle=False) as file:
        rx = file['rx_i'][chosen]
        if file['ids'][chosen].tolist()!=ids or np.bincount(rx).tolist()!=[180]*5:
            raise ValueError('Restored source order or RX panel differs')
        truth = file['y'][chosen]
    output = aggregate(routes,truth,rx)
    print(json.dumps(dict(status='VERIFIED', diagnostic_only=True, target_reads=0, truth_last=True,
                    packets=len(ids),modes=1+sum(len(cols) for _,_,cols in PARENTS.values()),
                    source_receivers=contract['source_receivers'],views=VIEWS,routes=output),allow_nan=False))


if __name__=='__main__':
    main()
