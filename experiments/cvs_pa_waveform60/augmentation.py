"""Consume every native PA draw, but return only the unchanged pre-PA waveform."""
import torch

class Wrapper:
    """Used only by native make_augmentor for the original visible L call."""
    def __init__(self, native, records):
        object.__setattr__(self, '_native', native)
        object.__setattr__(self, '_records', records)

    def __getattr__(self, key):
        return getattr(self._native, key)

    def __setattr__(self, key, value):
        setattr(self._native, key, value)

    def __call__(self, x, *args, **kwargs):
        if (args or set(kwargs) != {'labels', 'no_pa'} or kwargs['no_pa'] is not False
                or kwargs['labels'] is None or x.shape != (128, 2, 256)
                or x.dtype != torch.float32 or not bool(torch.isfinite(x).all())
                or kwargs['labels'].shape != (128,) or kwargs['labels'].dtype != torch.long
                or not bool(((kwargs['labels'] >= 0) & (kwargs['labels'] < 6)).all())):
            raise ValueError('Only original visible L128 ordinary augmentation; no U/As/raw/defect calls')
        aug = self._native
        if aug.p_dac != 0 or aug.enable_class_signature or aug.p_pa <= 0:
            raise ValueError('Keep original enabled PA, disabled DAC/class signature')
        original_pa, original_channel = aug.simulate_pa, aug._apply_general_channel
        trace = dict(prefix=None, shadow_calls=0, selected=0, changed=0, mse_sum=0., strength_sum=0.)

        def prefix(*a, **kw):
            result = original_channel(*a, **kw)
            if trace['prefix'] is not None:
                raise ValueError('Exactly one ordinary non-PA prefix required')
            trace['prefix'] = result.detach().clone()
            return result

        def shadow(v, *a, **kw):
            if trace['shadow_calls']:
                raise ValueError('At most one native normal PA subset call')
            trace['shadow_calls'] += 1
            # Execute the original chain, including Python and Torch draws and
            # any future NumPy draws, on a disposable clone. Never reseed it.
            hypothetical, strength = original_pa(v.clone(), *a, **kw)
            if hypothetical.shape != v.shape or not bool(torch.isfinite(hypothetical).all()):
                raise ValueError('Native shadow PA produced invalid waveform')
            trace['selected'] = len(v)
            trace['changed'] = int((hypothetical != v).flatten(1).any(1).sum())
            trace['mse_sum'] = float((hypothetical - v).square().mean((1, 2)).sum())
            trace['strength_sum'] = float(strength.sum())
            return v, torch.zeros_like(strength)

        aug.simulate_pa, aug._apply_general_channel = shadow, prefix
        try:
            result = aug(x, **kwargs)
        finally:
            aug.simulate_pa, aug._apply_general_channel = original_pa, original_channel
        if trace['prefix'] is None or not torch.equal(result, trace['prefix']):
            raise ValueError('Returned waveform must exactly equal its native non-PA prefix')
        self._records.append(dict(
            step=len(self._records) + 1, count=len(x),
            effective_p_pa=float(aug.p_pa), effective_p_dac=float(aug.p_dac),
            native_no_pa=False, native_class_signature=False,
            shadow_PA_calls=trace['shadow_calls'], shadow_PA_selected_packets=trace['selected'],
            shadow_PA_changed_packets=trace['changed'],
            shadow_PA_MSE_sum=trace['mse_sum'], shadow_PA_strength_sum=trace['strength_sum'],
            actual_PA_injected_packets=0, exact_native_nonPA_prefix=True,
            prefix_raw_MSE=float((result - x).square().mean()),
            same_native_draw=True, scope='ordinary_L_only', U_changed=False, As_changed=False))
        return result
