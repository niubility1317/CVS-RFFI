"""CPU checks for the fixed epoch-local replay and unchanged supervised path."""
from copy import deepcopy
import math

from .replay import EpochReplay


def _records(step, ids=None):
    epoch = (step - 1) // 222 + 1
    ids = ids or [(1, 1, 1, i, i) for i in range(128)]
    return [dict(sample_id=pid, namespace=f'FcrossRX/train/e{epoch}/step{step}/L_S',
                 session_id=f'rx{pid[1]}:day{pid[2]}:eq{pid[0]}',
                 scene='practical_low_urban', state_end='BLOCKED',
                 receiver_lock_status='unlocked') for pid in ids]


def replay_checks():
    import torch
    raw = torch.linspace(-1., 1., 128 * 512).reshape(128, 2, 256)
    fresh = raw + .25
    cache = EpochReplay()
    first, records, initial = cache.apply(fresh, _records(1), raw, epoch=1, step=1)
    if not torch.equal(first, fresh) or initial['miss_packets'] != 128 or initial['hit_packets']:
        raise AssertionError('First complete fullAs batch must be used exactly')
    # A clone is retained: mutation of caller-owned tensors/records cannot edit
    # the cached observation.  This is an in-memory correctness check only.
    first.zero_(); records[0]['scene'] = 'changed'
    second, used, summary = cache.apply(fresh + 1., _records(2), raw, epoch=1, step=2)
    if (not torch.equal(second, fresh) or summary['hit_packets'] != 128
            or summary['hit_changed_packets'] != 128
            or summary['source_steps'] != [1] * 128
            or summary['source_slots'] != list(range(128))
            or not all(summary['cacheflags']) or used[0]['scene'] != 'practical_low_urban'
            or second.requires_grad or second.grad_fn is not None):
        raise AssertionError('Cross-update first-IQ reuse/source coordinates differ')
    for step in range(3, 223):
        cache.apply(fresh + 1., _records(step), raw, epoch=1, step=step)
    refreshed, _, boundary = cache.apply(fresh + 2., _records(223), raw, epoch=2, step=223)
    if (not torch.equal(refreshed, fresh + 2.) or boundary['hit_packets']
            or boundary['source_steps'] != [223] * 128
            or boundary['reset_count'] != 1 or len(cache.cache) != 128):
        raise AssertionError('New epoch did not discard every old IQ realization')
    empty = deepcopy(cache)
    if empty.cache or empty.steps or any(empty.summary()[k] for k in (
            'reset_count', 'epochs_started', 'cumulative_generated', 'cumulative_used',
            'miss_packets', 'hit_packets')):
        raise AssertionError('Own EMA deepcopy must start with an empty zero-count cache')
    dup_raw, dup_fresh = raw.clone(), fresh.clone()
    ids = [tuple(v['sample_id']) for v in _records(1)]
    ids[1] = ids[0]; dup_raw[1] = dup_raw[0]; dup_fresh[1] = dup_fresh[0] + 3.
    duplicate = EpochReplay()
    value, _, dup = duplicate.apply(dup_fresh, _records(1, ids), dup_raw, epoch=1, step=1)
    if (not torch.equal(value[1], dup_fresh[0]) or dup['source_slots'][1] != 0
            or dup['miss_packets'] != 127 or dup['hit_packets'] != 1):
        raise AssertionError('Within-batch duplicate must use first physical slot')
    negatives = 0
    def reject(fn):
        nonlocal negatives
        try:
            fn()
        except ValueError:
            negatives += 1
        else:
            raise AssertionError('Invalid replay input accepted')
    reject(lambda: EpochReplay().apply(fresh, _records(2), raw, epoch=1, step=2))
    reject(lambda: EpochReplay().apply(fresh, _records(1), raw, epoch=2, step=1))
    reject(lambda: EpochReplay().apply(fresh, _records(1), raw, epoch=0, step=1))
    for pid in ((0, 1, 1, 0, 0), (1, 2, 1, 0, 0), (1, 1, 0, 0, 0),
                (1, 1, 1, -1, 0), (1, 1, 1, 0, -1), (1, 1, 1, 0),
                (1, 1, 1, 0., 0)):
        bad = _records(1); bad[0]['sample_id'] = pid
        reject(lambda bad=bad: EpochReplay().apply(fresh, bad, raw, epoch=1, step=1))
    for key, value in (('namespace', 'source_eval/fixed'), ('namespace', 'target_fixed_v3'),
                       ('session_id', 'rx7:day1:eq1'), ('scene', 'clean'),
                       ('state_end', None), ('receiver_lock_status', None)):
        bad = _records(1); bad[0][key] = value
        reject(lambda bad=bad: EpochReplay().apply(fresh, bad, raw, epoch=1, step=1))
    for tensor in ('fresh', 'raw'):
        broken = fresh.clone() if tensor == 'fresh' else raw.clone()
        broken[0, 0, 0] = float('nan')
        reject(lambda tensor=tensor, broken=broken: EpochReplay().apply(
            broken if tensor == 'fresh' else fresh, _records(1),
            broken if tensor == 'raw' else raw, epoch=1, step=1))
    reject(lambda: EpochReplay().apply(fresh[:1], _records(1), raw[:1], epoch=1, step=1))
    reject(lambda: EpochReplay().apply(fresh.double(), _records(1), raw.double(), epoch=1, step=1))
    reject(lambda: EpochReplay().apply(fresh, _records(1)[:1], raw, epoch=1, step=1))
    changed = raw.clone(); changed[0, 0, 0] += 1.
    witness = EpochReplay(); witness.apply(fresh, _records(1), raw, epoch=1, step=1)
    before = witness.summary()
    reject(lambda: witness.apply(fresh, _records(2), changed, epoch=1, step=2))
    if before != witness.summary():
        raise AssertionError('Rejected raw mismatch consumed a native training step')
    reject(lambda: witness.apply(fresh, _records(1), raw, epoch=1, step=1))
    return dict(status='PASS', replay_protocol_negatives=negatives,
                cross_step_first_IQ_reused=True, each_epoch_refreshes=True,
                duplicate_batch_first_slot=True, no_graph_or_persistent_model_state=True,
                deepcopy_empty=True, rejected_step_does_not_mutate=True,
                synthetic_replay_successes=225, real_channel_packets=0,
                NN_packets=0, optimizer_updates=0)


def learning_checks(pair, draw_and_use, ordinary_factory):
    import torch
    labels = torch.arange(128) % 6
    raw = torch.linspace(-1., 1., 128 * 512).reshape(128, 2, 256)
    calls, records, witnesses = [], [], []
    class Native:
        def __call__(self, value, *, labels, no_pa):
            calls.append((len(value), no_pa))
            return value + torch.rand_like(value)
    def original():
        return Native()
    wrapper = ordinary_factory(original, records,
        lambda before, ordinary, labels, step:
            witnesses.append((before.clone(), ordinary.clone(), labels.clone(), step)))()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        expected = original()(raw.clone(), labels=labels, no_pa=False)
        expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(73)
        actual = wrapper(raw, labels=labels, no_pa=False)
        if (not torch.equal(actual, expected)
                or not torch.equal(expected_rng, torch.random.get_rng_state())
                or not torch.equal(witnesses[0][0], raw)
                or not torch.equal(witnesses[0][1], expected)
                or witnesses[0][3] != 1 or len(records) != 1):
            raise AssertionError('Native ordinary draw/first input/RNG changed')
    hidden_negatives = 0
    for call in (lambda: draw_and_use(Native(), raw, torch.full_like(labels, -1)),
                 lambda: wrapper(raw, labels=torch.full_like(labels, -1), no_pa=False)):
        before = len(calls)
        try:
            call()
        except ValueError:
            hidden_negatives += 1
        else:
            raise AssertionError('Hidden U admitted to labeled ordinary wrapper')
        if len(calls) != before:
            raise AssertionError('Hidden U reached the native labeled augmentation')
    first = torch.linspace(-1., 1., 128 * 6).reshape(128, 6).requires_grad_()
    second = first.detach().flip(1).clone().requires_grad_()
    loss, _, _ = pair(first, second, labels)
    one, two = torch.autograd.grad(loss, (first, second))
    target = torch.nn.functional.one_hot(labels, 6).float() * .99 + .01 / 6
    if (not torch.allclose(one, .5 * (first.softmax(1) - target) / 128, atol=1e-7)
            or not torch.allclose(two, .5 * (second.softmax(1) - target) / 128, atol=1e-7)):
        raise AssertionError('Both slots must supervise all128 true labels at weight0.5')
    return dict(all128_true_CE_gradient=True,
                original_ordinary_IQ_and_RNG_retained=True,
                hidden_U_rejected_before_augmentation=hidden_negatives,
                synthetic_logits_backward=1, optimizer_updates=0, NN_packets=0)


def run():
    from . import design as d
    from .model import pair
    from .source import draw_and_use, ordinary_factory
    result = replay_checks()
    result.update(learning_checks(pair, draw_and_use, ordinary_factory))
    negatives = 0
    for change in ({'epochs': 50}, {'model_seed': 2}, {'checkpoint_sources': ['old.pth']},
                   {'data_contract': {}}, {'head_fit_updates': 1}, {'training_mechanism': {}}):
        config = deepcopy(d.config(d.rows()[0])); config.update(change)
        try:
            d.validate(config)
        except ValueError:
            negatives += 1
        else:
            raise AssertionError('Fixed contract mutation accepted')
    reference = vars(d.base.make_args(d.base.config(next(
        row for row in d.base.rows() if row['arm'] == 'B1')), 'cpu'))
    for row in d.rows():
        config = d.config(row); actual = vars(d.make_args(config, 'cpu'))
        def equal(a, b):
            return a == b or (isinstance(a, float) and isinstance(b, float)
                              and math.isnan(a) and math.isnan(b))
        difference = {key for key in set(actual) | set(reference)
                      if not equal(actual.get(key), reference.get(key))}
        if difference - {'run_id', 'candidate_id', 'output_dir'}:
            raise AssertionError('Original B1 native data/U/architecture args differ: ' + str(difference))
        if (config['training_mechanism']['L_forwards_per_sample'] != 2
                or config['training_mechanism']['teacher_packets_per_step'] != 0
                or not actual['use_aug']):
            raise AssertionError('Original ordinary/U/external-teacher budget differs')
    result.update(fixed_config_negatives=negatives,
                  native_arguments_match_B1_except_run_identity=True)
    return result


if __name__ == '__main__':
    print(run())
