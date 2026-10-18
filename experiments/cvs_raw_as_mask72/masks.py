"""Call-local masks at the fixed native fusion Dropout; no new network state."""
from contextlib import contextmanager
import hashlib
import json

import torch
from torch import nn

TARGET = 'id_backbone.encoder.core.id_backbone.fuse.2'
MASK_SEED = 20261019072
MODES = ('INDEPENDENT_FUSE', 'SHARED_RAW_AS_FUSE')


def _seed(cumulative_step, pair, role):
    """Private named generator seed; contains no label or physical identity."""
    value = json.dumps([MASK_SEED, cumulative_step, pair, role], separators=(',', ':')).encode('ascii')
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'little') & ((1 << 63) - 1)


class Recorder:
    def __init__(self, step, cumulative_step, method, smask):
        self.step, self.cumulative_step, self.method = step, cumulative_step, method
        self.smask = smask.detach().clone()
        self.p = .45; self.calls = 0; self.call_records = []; self._masks = {}
        self.mask_seeds = {role + str(pair): _seed(cumulative_step, pair, role)
                           for role in ('As', 'raw') for pair in (0, 1)}
        self._device = self._dtype = None

    def _initialize(self, value):
        if self._masks:
            if value.device != self._device or value.dtype != self._dtype:
                raise ValueError('Fixed fusion input dtype/device must agree across five calls')
            return
        self._device, self._dtype = value.device, value.dtype
        # Both arms make all four private draws, including unused raw masks in S1.
        for key, seed in self.mask_seeds.items():
            generator = torch.Generator(device=value.device).manual_seed(seed)
            self._masks[key] = torch.rand((64, 160), device=value.device,
                                          dtype=torch.float32, generator=generator) >= self.p

    def hook(self, module, inputs, output):
        index = self.calls; self.calls += 1
        if index >= 5:
            raise ValueError('Only five existing identity forwards belong to this scope')
        if not module.training or not torch.is_grad_enabled():
            raise ValueError('Fusion masks are restricted to gradient-enabled L training')
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
            raise ValueError('Native fusion Dropout tensor input/output required')
        before = inputs[0]
        if before.shape != (128, 160) or output.shape != before.shape or not before.is_floating_point() or before.device != self.smask.device:
            raise ValueError('Fixed native fusion shape128x160 and source mask device required')
        self._initialize(before)
        if index == 0:
            self.call_records.append(dict(index=index, role='ordinary', overridden_rows=0,
                                          native_dropout_executed=True))
            return output
        role, pair = ('As', index - 1) if index <= 2 else ('raw', index - 3)
        used_role = 'As' if role == 'raw' and self.method == 'SHARED_RAW_AS_FUSE' else role
        selected = self._masks[used_role + str(pair)].to(before.dtype) / (1. - self.p)
        result = output.clone()
        # The native draw already happened; overwrite from its INPUT, not output.
        result[self.smask] = before[self.smask] * selected
        self.call_records.append(dict(index=index, role=role, pair=pair, overridden_rows=64,
                                      native_dropout_executed=True, used_mask_role=used_role,
                                      used_mask_seed=self.mask_seeds[used_role + str(pair)],
                                      independent_role_seed=self.mask_seeds[role + str(pair)]))
        return result

    def summary(self):
        if self.calls != 5 or len(self._masks) != 4:
            raise ValueError('Only a complete five-call scope has a mask summary')
        a0, a1 = self._masks['As0'], self._masks['As1']
        r0, r1 = self._masks['raw0'], self._masks['raw1']
        used_r0, used_r1 = (a0, a1) if self.method == 'SHARED_RAW_AS_FUSE' else (r0, r1)
        values = [self._masks[key].sum() for key in self.mask_seeds]
        values += [(a0 != used_r0).sum(), (a1 != used_r1).sum(),
                   (a0 != a1).sum(), (used_r0 != used_r1).sum(), (r0 != r1).sum()]
        counts = torch.stack(values).detach().cpu().tolist()
        hamming = counts[4:6]
        return dict(method=self.method, step=self.step, cumulative_step=self.cumulative_step,
                    target=TARGET, dropout_p=self.p, calls=self.calls,
                    private_draws=4, private_draw_slots=4 * 64 * 160,
                    overridden_calls=4, overridden_rows_per_call=64, overridden_slots=4 * 64 * 160,
                    native_dropout_calls=5, native_dropout_slots=5 * 128 * 160,
                    mask_seeds=dict(self.mask_seeds),
                    private_kept_counts=dict(zip(self.mask_seeds, counts[:4])),
                    As_raw_pair_hamming=hamming, As_RD_hamming=counts[6],
                    raw_RD_hamming=counts[7], unused_raw_RD_hamming=counts[8],
                    paired_same=all(value == 0 for value in hamming),
                    RD_different=counts[6] > 0 and counts[7] > 0,
                    call_records=[dict(record) for record in self.call_records],
                    rng='four named private generators; original native dropout draws retained',
                    new_parameters=0, persistent_model_state=0)


@contextmanager
def scope(model, *, step, cumulative_step, method, smask):
    """Cover ordinary1/mixed0/mixed1/raw0/raw1 within the existing L hook."""
    if not model.training or not torch.is_grad_enabled():
        raise ValueError('Mask scope requires gradient-enabled L student training')
    if method not in MODES or type(step) is not int or type(cumulative_step) is not int or step <= 0 or cumulative_step < step:
        raise ValueError('Declared method and positive local/cumulative step required')
    if not isinstance(smask, torch.Tensor) or smask.dtype != torch.bool or smask.shape != (128,):
        raise ValueError('Original source L128 boolean As slot mask required')
    expected = (torch.arange(128, device=smask.device) + step).remainder(2) != 0
    if not torch.equal(smask, expected):
        raise ValueError('Source As64 slot parity differs from original P43')
    target = model.get_submodule(TARGET)
    if type(target) is not nn.Dropout or not target.training or abs(float(target.p) - .45) > 1e-12 or target.inplace:
        raise ValueError('Exact native non-inplace fusion Dropout(.45) required')
    recorder = Recorder(step, cumulative_step, method, smask)
    handle = target.register_forward_hook(recorder.hook)
    try:
        yield recorder
    except BaseException:
        raise  # Preserve the original failure instead of hiding it with call count.
    else:
        if recorder.calls != 5:
            raise ValueError('Mask scope must observe exactly five existing identity forwards')
    finally:
        handle.remove()
