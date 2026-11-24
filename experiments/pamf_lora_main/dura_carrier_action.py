"""Constrain DURA's learned observation correction to the tracked carrier subspace."""

import torch

from .dura_prior_width import PacketDura


CARRIER_BINS = (-2, -1, 0, 1, 2)


class CarrierRestrictedAction:
    """Represent I + P(A - I)P in the rotated five-bin basis of the observation."""

    def __init__(self, action, rotation):
        count = rotation.shape[-1]
        frequency = rotation.real.new_tensor(CARRIER_BINS)[:, None]
        time = torch.arange(count, device=rotation.device, dtype=rotation.real.dtype)
        angle = 2 * torch.pi * frequency * time / count
        waves = torch.polar(torch.ones_like(angle), angle) / count**.5
        self.basis = rotation[:, None, :] * waves
        filtered = torch.stack([action.filter(wave) for wave in self.basis.unbind(1)], dim=-1)
        dual = self.basis.conj()
        eye = torch.eye(len(CARRIER_BINS), dtype=rotation.dtype, device=rotation.device)
        self.direct_delta = dual @ (action.main[..., None] * filtered) - eye
        self.image = dual @ (action.image[..., None] * filtered.conj())

    def __call__(self, value):
        coefficients = self.basis.conj() @ value[..., None]
        delta = self.direct_delta @ coefficients + self.image @ coefficients.conj()
        return value + (self.basis.transpose(-2, -1) @ delta).squeeze(-1)

    def adjoint(self, value):
        coefficients = self.basis.conj() @ value[..., None]
        delta = self.direct_delta.mH @ coefficients + self.image.transpose(-2, -1) @ coefficients.conj()
        return value + (self.basis.transpose(-2, -1) @ delta).squeeze(-1)


class CarrierRestrictedDura(PacketDura):
    def __init__(self):
        super().__init__(wide=True)

    def solve(self, native, _observed, rotation):
        coarse, action, variance = self.restorer.estimate_context(native)
        restricted = CarrierRestrictedAction(action, rotation.flatten(0, 1))
        return self.restorer.stage_outputs(native, context=(coarse, restricted, variance))
