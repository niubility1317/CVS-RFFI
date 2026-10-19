"""Narrow fixed-contract and native-argument checks; actual input hooks tested separately."""
from copy import deepcopy
import math
from . import design as d


def run():
    import torch
    from .model import pair
    from .source import ordinary_factory
    # Check input organization with a stochastic fake native augmentor: consume
    # exactly the same draw, return raw, and never admit hidden U labels.
    x = torch.linspace(-1., 1., 128*2*256).reshape(128, 2, 256)
    y = torch.arange(128) % 6
    calls, records, witnesses = [], [], []
    class Native:
        def __call__(self, value, *, labels, no_pa):
            calls.append((len(value), no_pa))
            return value + torch.rand_like(value)
    def original():
        return Native()
    factory = ordinary_factory(original, records, lambda raw, labels, step:
                                witnesses.append((raw.clone(), labels.clone(), step)))
    wrapped = factory()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73); original()(x.clone(), labels=y, no_pa=False)
        expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(73); returned = wrapped(x, labels=y, no_pa=False)
        if not torch.equal(returned, x) or not torch.equal(expected_rng, torch.random.get_rng_state()):
            raise AssertionError('Discarded ordinary RNG/raw return differs')
    if len(records) != 1 or records[0]['full_native_draws'] != 1 or len(witnesses) != 1 or witnesses[0][2] != 1:
        raise AssertionError('Original ordinary draw/raw witness did not occur exactly once')
    try:
        wrapped(x, labels=torch.full_like(y, -1), no_pa=False)
    except ValueError:
        pass
    else:
        raise AssertionError('Hidden U was admitted to the L-only raw wrapper')
    first = torch.linspace(-1., 1., 128*6).reshape(128, 6).requires_grad_()
    second = first.detach().flip(1).clone().requires_grad_()
    total, _, _ = pair(first, second, y)
    g1, g2 = torch.autograd.grad(total, (first, second))
    target = torch.nn.functional.one_hot(y, 6).float()*.99+.01/6
    if not torch.allclose(g1, .5*(first.softmax(1)-target)/128, atol=1e-7) or not torch.allclose(
            g2, .5*(second.softmax(1)-target)/128, atol=1e-7):
        raise AssertionError('Both true-label CE slots must use all128/weight0.5')
    changes = ({'epochs': 50}, {'checkpoint_sources': ['old.pth']}, {'model_seed': 2},
               {'data_contract': {}}, {'head_fit_updates': 1}, {'training_mechanism': {}})
    negatives = 0
    for change in changes:
        c = deepcopy(d.config(d.rows()[0])); c.update(change)
        try:
            d.validate(c)
        except ValueError:
            negatives += 1
        else:
            raise AssertionError('Unauthorized fixed-contract mutation accepted')
    reference = vars(d.base.make_args(d.base.config(next(r for r in d.base.rows()
                                                       if r['arm'] == 'B1')), 'cpu'))
    for row in d.rows():
        c = d.config(row); actual = vars(d.make_args(c, 'cpu'))
        def equal(a, b):
            return a == b or (isinstance(a, float) and isinstance(b, float)
                              and math.isnan(a) and math.isnan(b))
        delta = {k for k in set(actual)|set(reference) if not equal(actual.get(k), reference.get(k))}
        if delta - {'run_id', 'candidate_id', 'output_dir'}:
            raise AssertionError('Native data/U/optimizer/architecture arguments changed: '+str(delta))
        m = c['training_mechanism']
        if m['L_forwards_per_sample'] != 2 or m['teacher_packets_per_step'] != 0 or not m['ordinary_draw_shadow'] or not actual['use_aug']:
            raise AssertionError('Matched L forward/RNG/original U recipe differs')
    return dict(status='PASS', protocol_negatives=negatives,
                raw_wrapper_discards_IQ_and_preserves_ordinary_RNG=True,
                hidden_U_rejected_before_augmentation=True, all128_true_CE_gradient=True,
                native_arguments_unchanged_except_run_identity=True,
                two_L128_forwards=True, ordinary_draw_shadow=True, original_U_unchanged=True)


if __name__ == '__main__':
    print(run())
