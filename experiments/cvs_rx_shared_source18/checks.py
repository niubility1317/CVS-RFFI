"""Focused source-permission and scientific-control checks; no real IQ or NN."""
from copy import deepcopy
import numpy as np
from . import design as d
from .core import cube, evaluate
from .permissions import select


class ShapeOnly:
    def __init__(self, n):
        self.shape = (n, 2, 256)


class SourceOnly(dict):
    def __getitem__(self, key):
        if key == 'unlabeled':
            raise AssertionError('U container must never be read')
        return super().__getitem__(key)


def synthetic_roles():
    packed = SourceOnly()
    role_ids = {'U_s': ['opaque-hidden-U-id']}
    for key, handle, n in (('labeled', 'L_s', 70), ('validation', 'V', 300)):
        rows = [(t, r, date) for t in range(6) for r in d.RX for date in (1, 2, 3) for _ in range(n)]
        tx, rx, day = (np.array([v[j] for v in rows]) for j in range(3))
        ids = [handle + '-opaque-' + str(i) for i in range(len(rows))]
        role_ids[handle] = ids
        packed[key] = dict(x=ShapeOnly(len(rows)), y=tx, rx=rx, day=day, ids=ids)
    contract = dict(role_ids=role_ids, native_role_comparison='EXACT_MATCH')
    packed['contract'] = deepcopy(contract)
    prepared = dict(status='SOURCE_GEOMETRY_AUDIT_COMPLETE', target_access=False, source_contract=deepcopy(contract))
    return packed, prepared, contract


def run():
    negatives = 0
    def reject(fn):
        nonlocal negatives
        try:
            fn()
        except ValueError:
            negatives += 1
        else:
            raise AssertionError('Protocol/scientific negative accepted')

    p, ready, c = synthetic_roles()
    selected = select(p, ready, c)
    if set(selected) != {'L_s', 'V'}:
        raise AssertionError('Original L/V roles required')
    for mutation in ('overlap', 'mismatch', 'missing_RX', 'bad_shape', 'bad_ready', 'target_ready'):
        pp, rr, cc = deepcopy(p), deepcopy(ready), deepcopy(c)
        if mutation == 'overlap':
            cc['role_ids']['U_s'].append(cc['role_ids']['L_s'][0])
            pp['contract'] = deepcopy(cc)
            rr['source_contract'] = deepcopy(cc)
        elif mutation == 'mismatch':
            pp['contract']['role_ids']['V'][0] = 'different-id'
        elif mutation == 'missing_RX':
            pp['validation']['rx'][:] = 3
        elif mutation == 'bad_shape':
            pp['labeled']['x'].shape = (6300, 2, 255)
        elif mutation == 'bad_ready':
            rr['status'] = 'UNKNOWN'
        else:
            rr['target_access'] = True
        reject(lambda: select(pp, rr, cc))
    for key, value in (('RX', [3, 4, 6, 8]), ('target_access', True), ('V', 720), ('window', [20, 180])):
        config = d.config()
        config[key] = value
        reject(lambda: d.validate(config))
    arrays = [np.zeros((4, 3, 6)), np.full((5, 3, 6), np.nan)]
    for a in arrays:
        reject(lambda: evaluate(a, np.zeros((5, 3, 6))))
    reject(lambda: cube(np.zeros(89), np.zeros(89), np.ones(89), np.ones(89), 1))
    reject(lambda: cube(np.full(90, np.inf), np.zeros(90), np.ones(90), np.ones(90), 1))
    metadata = p['labeled']
    values = np.array([.02 + .01 * r for r in metadata['rx']])
    means, cells = cube(values, metadata['y'], metadata['rx'], metadata['day'], 70)
    if len(cells) != 90 or not np.allclose(means[:, 0, 0], [.02+.01*r for r in d.RX]):
        raise AssertionError('All5RX cell scalar reduction wrong')
    stable = np.broadcast_to(np.array([.03, .03, .12, .12, .15])[:, None, None], (5, 3, 6)).copy()
    distinct = np.broadcast_to(np.linspace(.01, .19, 5)[:, None, None], (5, 3, 6)).copy()
    controls = dict(shared_RX=evaluate(stable, stable), distinct_with_zero_RX_contrast=evaluate(distinct, distinct),
                    TX_only=evaluate(np.full((5, 3, 6), .1), np.full((5, 3, 6), .1)))
    reversal = stable - stable.mean(axis=0, keepdims=True)
    controls['RX_day_reversal'] = evaluate(.1 + reversal*np.array([-1., 0., 1.])[None, :, None],
                                           .1 + reversal*np.array([-1., 0., 1.])[None, :, None])
    controls['RX_TX_reversal'] = evaluate(.1 + reversal*np.array([-1.,-1.,-1.,1.,1.,1.])[None,None,:],
                                          .1 + reversal*np.array([-1.,-1.,-1.,1.,1.,1.])[None,None,:])
    controls['L_V_reversal'] = evaluate(stable, 2*stable.mean()-stable)
    expected = {'shared_RX': True, 'distinct_with_zero_RX_contrast': True, 'TX_only': False,
                'RX_day_reversal': False, 'RX_TX_reversal': False, 'L_V_reversal': False}
    for key, result in controls.items():
        if all(result['conditions'].values()) != expected[key] or result['wrong_RX_is_promotion_condition']:
            raise AssertionError('Known scalar control verdict wrong: '+key)
    # V must change only readout errors, never the fitted L predictions.
    changed = evaluate(stable, stable + .003 * np.arange(5)[:, None, None])
    if [f['prediction'] for f in changed['folds']] != [f['prediction'] for f in controls['shared_RX']['folds']]:
        raise AssertionError('V entered L fitted prediction')
    return dict(status='PASS', negatives=negatives, scientific_cases={k:v['decision'] for k,v in controls.items()},
                synthetic_unique_metadata_packets=33300, scalar_packets_reduced=6300,
                scientific_evaluate_calls=7, cell_means_crossfit_evaluated=1260,
                real_IQ=0, synthetic_IQ=0, CP=0, NN=0, updates=0, target=False)


if __name__ == '__main__':
    import json
    print(json.dumps(run(), allow_nan=False))
