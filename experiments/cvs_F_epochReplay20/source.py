"""Observe the native ordinary L call and return its exact ordinary output."""
import torch


def draw_and_use(augmentor, x, labels):
    if (x.shape != (128, 2, 256) or x.dtype != torch.float32
            or not bool(torch.isfinite(x).all()) or labels.shape != (128,)
            or labels.dtype != torch.long
            or bool(((labels < 0) | (labels > 5)).any())):
        raise ValueError('Finite visible L128 IQ256 and labels0..5 required')
    raw = x.detach().clone()
    ordinary = augmentor(x, labels=labels, no_pa=False)
    if (ordinary.shape != raw.shape or ordinary.dtype != raw.dtype
            or not bool(torch.isfinite(ordinary).all()) or not torch.equal(raw, x)):
        raise ValueError('Native ordinary must preserve raw and return finite original IQ')
    return raw, ordinary, dict(
        ordinary_raw_MSE=float((ordinary.detach() - raw).square().mean()),
        ordinary_changed_packets=int((ordinary.detach() != raw).flatten(1).any(1).sum()),
        returned_ordinary_MSE=0., full_native_draws=1, count=128,
        original_input_unchanged=True, full_native_aug_computation=True,
        ordinary_output_used_by_student=True, ordinary_output_returned_unmodified=True,
    )


class Wrapper:
    def __init__(self, native, records, before_return):
        object.__setattr__(self, '_native', native)
        object.__setattr__(self, '_records', records)
        object.__setattr__(self, '_before_return', before_return)

    def __getattr__(self, key):
        return getattr(self._native, key)

    def __setattr__(self, key, value):
        setattr(self._native, key, value)

    def __call__(self, x, *args, **kwargs):
        if (args or set(kwargs) != {'labels', 'no_pa'}
                or kwargs['no_pa'] is not False or kwargs['labels'] is None):
            raise ValueError('Only native visible L labels/no_paFalse call may be wrapped')
        raw, ordinary, record = draw_and_use(self._native, x, kwargs['labels'])
        step = len(self._records) + 1
        self._before_return(raw, ordinary, kwargs['labels'], step)
        self._records.append(dict(step=step, **record))
        return ordinary


def ordinary_factory(original, records, before_return):
    def create(*args, **kwargs):
        create.calls += 1
        if create.calls != 1:
            raise ValueError('Exactly one native ordinary L augmentor is supported')
        return Wrapper(original(*args, **kwargs), records, before_return)
    create.calls = 0
    return create


def train(c, design=None, model_api=None, observer=None):
    if observer is None:
        from .observer import train as run
        return run(c)
    from experiments.cvs_feature_sequence50.source import train as shared_train
    return shared_train(c, design=design, model_api=model_api, observer=observer)
