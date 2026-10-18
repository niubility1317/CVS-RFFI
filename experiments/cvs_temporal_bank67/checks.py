"""Protocol negatives and the actual native bank's tensor-only toy cases."""
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d
from .temporal import FIELDS, TemporalTrace, load_native_function


def native_source():
    native = sys.modules.get('SSDG.train_ssdg')
    if native is not None and getattr(native, '__file__', None):
        return Path(native.__file__)
    from experiments.cvs_phase1_overlay.model import NATIVE
    return NATIVE / 'SSDG/train_ssdg.py'


def run(native_path=None):
    negative = 0

    def bad(fn):
        nonlocal negative
        try:
            fn()
        except (ValueError, TypeError, KeyError, AttributeError):
            negative += 1
        else:
            raise AssertionError('An invalid temporal-bank67 contract was accepted')

    assert tuple(r['arm'] for r in d.rows()) == ('B00', 'B01')
    parent_roots = []
    for row in d.rows():
        c = d.config(row)
        d.validate(c)
        parent_roots.append(c['parent_source'])
        assert c['As'] == (row['arm'] == 'B01')
        expected = d.old.make_args(d.old_config(row['arm']), 'cpu')
        actual = d.make_args(c, 'cpu')
        def equal(a, b):
            return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
        differences = {k for k in set(vars(expected)) | set(vars(actual)) if not equal(getattr(expected, k, None), getattr(actual, k, None))}
        assert differences - {'run_id', 'candidate_id', 'output_dir'} == {'pseudo_temporal_mode'}
        assert expected.pseudo_temporal_mode == 'batch_neighbor'
        assert actual.pseudo_temporal_mode == 'epoch_bank'
        assert actual.pseudo_temporal_bank_min_streak == 2 and actual.pseudo_temporal_min_conf == .8
        for key, value in [('target_access', True), ('As', not c['As']), ('new_Ag', True),
                           ('epochs', 20), ('updates', 4440), ('model_seed', 2026092702),
                           ('parent_source', 'unknown'), ('label_epochs', 13),
                           ('temporal_mode', 'batch_neighbor'), ('temporal_min_streak', 1)]:
            changed = copy.deepcopy(c)
            changed[key] = value
            bad(lambda changed=changed: d.validate(changed))
        bad(lambda row=row: d.config(row, 20))
    assert len(set(parent_roots)) == 2
    native = load_native_function(native_path or native_source())
    args = SimpleNamespace(pseudo_temporal_mode='epoch_bank', pseudo_temporal_gate=True,
        pseudo_temporal_min_conf=.8, pseudo_temporal_bank_min_streak=2)
    records = []
    trace = TemporalTrace(records.append, steps_per_epoch=3)
    bank = {}
    A, B, C, D = (1, 1, 1, 10, 10), (3, 1, 1, 10, 10), (1, 1, 1, 30, 30), (1, 1, 1, 40, 40)

    def call(keys, predictions, confidence):
        step = trace.calls + 1
        meta = {name: [key[i] for key in keys] for i, name in enumerate(FIELDS)}
        return trace.call(native, torch.tensor(predictions, dtype=torch.long),
            torch.tensor(confidence, dtype=torch.float64), (None, meta), args, 'cpu',
            epoch=(step - 1) // 3 + 1, bank=bank, step=step).tolist()

    assert call([A, B, C, D], [0, 1, 2, 3], [.99, .99, .79, .8]) == [False] * 4
    assert len(bank) == 3 and A in bank and B in bank and C not in bank
    assert call([A, A], [0, 0], [.99, .99]) == [False, False]
    assert bank[A] == (0, 1, 1)
    assert call([B], [2], [.99]) == [False] and bank[B] == (2, 1, 1)
    assert call([D, A, B, C], [3, 0, 1, 2], [.99] * 4) == [True, True, False, False]
    assert call([A], [0], [.79]) == [False] and bank[A] == (0, 2, 2)
    assert call([A], [0], [.99]) == [True] and bank[A] == (0, 2, 2)
    assert call([B], [1], [.99]) == [True]
    assert call([D], [3], [.99]) == [True]
    assert call([C], [2], [.99]) == [True]
    assert call([A], [0], [.99]) == [False] and bank[A] == (0, 4, 1)
    assert trace.summary()['epoch_summaries'][0]['temporal_selected'] == 0
    assert trace.summary()['no_same_epoch_streak_growth'] and trace.summary()['bank_owned_exclusively']
    extra = (None, {name: [A[i]] for i, name in enumerate(FIELDS)})
    p, q = torch.tensor([0]), torch.tensor([.99], dtype=torch.float64)
    bad(lambda: TemporalTrace(lambda r: None).call(None, p, q, extra, args, 'cpu', epoch=1, bank={}, step=1))
    bad(lambda: TemporalTrace(lambda r: None).call(native, p, q, extra, args, 'cpu', epoch=1, bank={A: (0, 1, 1)}, step=1))
    bad(lambda: trace.call(native, p, q, extra, args, 'cpu', epoch=4, bank=dict(bank), step=11))
    missing = copy.deepcopy(extra)
    missing[1].pop('base_index')
    bad(lambda: TemporalTrace(lambda r: None).call(native, p, q, missing, args, 'cpu', epoch=1, bank={}, step=1))
    invalid_args = copy.copy(args)
    invalid_args.pseudo_temporal_mode = 'batch_neighbor'
    bad(lambda: TemporalTrace(lambda r: None).call(native, p, q, extra, invalid_args, 'cpu', epoch=1, bank={}, step=1))
    bad(lambda: TemporalTrace(lambda r: None).call(native, p, torch.tensor([float('nan')]), extra, args, 'cpu', epoch=1, bank={}, step=1))
    return dict(status='PASS', protocol_negatives=negative, native_temporal_toy_calls=len(records),
        actual_native_function_source=str(native_path or native_source()),
        first_epoch_all_false=True, shuffled_physical_IDs_confirm=True,
        same_base_index_different_RX_separate=True, same_epoch_duplicates_do_not_increment=True,
        changed_prediction_resets=True, low_confidence_does_not_update=True, epoch_gap_resets=True,
        one_native_argument_change='pseudo_temporal_mode', physical_key=list(FIELDS),
        real_IQ=0, checkpoint_reads=0, CVS_NN=0, optimizer_updates=0, target_access=False)


if __name__ == '__main__':
    print(json.dumps(run()))
