"""Use raw for the unique native L call after its original ordinary shadow draw."""
import torch


def draw_and_discard(augmentor, x, labels):
    """Retain the reviewed LRaw input scope and original ordinary RNG work."""
    if (x.shape != (128, 2, 256) or x.dtype != torch.float32
            or not bool(torch.isfinite(x).all()) or labels.shape != (128,)
            or labels.dtype != torch.long
            or bool(((labels < 0) | (labels > 5)).any())):
        raise ValueError('Finite visible L128 IQ256 and labels0..5 required')
    raw = x.detach().clone()
    discarded = augmentor(raw.clone(), labels=labels, no_pa=False)
    if (discarded.shape != raw.shape or discarded.dtype != raw.dtype
            or not bool(torch.isfinite(discarded).all())
            or not torch.equal(raw, x)):
        raise ValueError('Original ordinary shadow changed input or became invalid')
    return raw, dict(
        discardedordinary_MSE=float((discarded - raw).square().mean()),
        discardedordinary_changed_packets=int((discarded != raw).flatten(1).any(1).sum()),
        returned_raw_MSE=float((raw - x).square().mean()),
        full_native_draws=1, count=128, input_clone=True,
        original_input_unchanged=True, full_native_aug_computation=True,
        ordinary_output_used_by_student=False,
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
            raise ValueError('Only the original visible L labels/no_paFalse call is wrapped')
        raw, record = draw_and_discard(self._native, x, kwargs['labels'])
        step = len(self._records) + 1
        self._before_return(raw, kwargs['labels'], step)
        self._records.append(dict(step=step, **record))
        return raw


def ordinary_factory(original, records, before_return):
    def create(*args, **kwargs):
        create.calls += 1
        if create.calls != 1:
            raise ValueError('Exactly one original ordinary L augmentor is supported')
        return Wrapper(original(*args, **kwargs), records, before_return)
    create.calls = 0
    return create


def train(c, design=None, model_api=None, observer=None):
    """Same source trainer; only the explicit observer scopes L raw return."""
    if observer is None:
        from .observer import train as run
        return run(c)
    from experiments.cvs_feature_sequence50.source import train as shared_train
    return shared_train(c, design=design, model_api=model_api, observer=observer)
