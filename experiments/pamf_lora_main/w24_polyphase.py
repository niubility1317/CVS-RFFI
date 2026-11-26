"""Recover W24 center IQ with access to discarded native polyphase projections."""

import torch

from .w24_packet_unet import CENTER, PacketUNet
from .w24_probe_transport import phase_probe


OFFSETS = (-3, -1, 1, 3)


def orthogonal_taps(taps):
    """Remove overlap with the center aperture from four nearby resampling kernels."""
    samples = torch.arange(taps.shape[-1], device=taps.device)
    shifted = []
    for offset in OFFSETS:
        valid = (samples >= max(0, offset)) & (samples < len(samples) + min(0, offset))
        shifted.append(taps.roll(offset, dims=-1) * valid)
    shifted = torch.stack(shifted, dim=-1)
    center = taps.unsqueeze(-1)
    power = center.square().sum(-2, keepdim=True).clamp_min(1e-12)
    residual = shifted - center * ((shifted * center).sum(-2, keepdim=True) / power)
    return residual * (power / residual.square().sum(-2, keepdim=True).clamp_min(1e-12)).sqrt()


def native_context_fields(raw, tracker, phase):
    windows = raw[:, tracker.indices]
    taps = tracker.taps
    center = (windows * taps).sum(-1)
    rms = center.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    residual = torch.einsum('bctw,twk->bctk', windows, orthogonal_taps(taps).to(raw.dtype))
    # Reflected windows can repeat a physical sample. Omit these narrow boundary
    # contexts so that the stated orthogonality is in actual sample coordinates.
    valid = (tracker.indices.diff(dim=-1) == 1).all(-1)
    return residual * (valid * phase.conj() / rms).unsqueeze(-1)


def matching_aperture_fields(y):
    samples = torch.arange(CENTER, device=y.device)[:, None]
    indices = (samples + samples.new_tensor(OFFSETS)).clamp(0, CENTER - 1)
    return y[..., indices] - y[..., None]


class NativePolyphaseUNet(PacketUNet):
    def __init__(self, *, native_context):
        super().__init__(across_chirps=True, receive_fields=2 * len(OFFSETS))
        self.native_context = native_context

    def propose_native(self, tracker, y, phase, quality, rms, edges, raw):
        fields = (native_context_fields(raw, tracker, phase)
                  if self.native_context else matching_aperture_fields(y))
        fields = torch.cat((fields.real, fields.imag), dim=-1).transpose(-1, -2).clamp(-4., 4.)
        return self(y, quality, rms, edges, fields=fields),


def native_probe_views(raw, tracker, condition, rng, weak, *, with_identity):
    """Replay the same channel draw for the paired original and pre-channel probe."""
    original = raw.unflatten(-1, (8, 1024))
    modified = phase_probe(raw, weak)
    replay = torch.Generator(device=raw.device)
    replay.set_state(rng.get_state())
    a, b = [condition(original, rng).flatten(-2) for _ in range(2)]
    probe = condition(modified, replay).flatten(-2)
    clean_probe = modified.flatten(-2)
    if with_identity:
        a[-2:] = b[-2:] = raw[-2:]
        probe[-2:] = clean_probe[-2:]
    native = torch.stack((a, b, probe), dim=1)
    apertures = tracker.aperture(native.flatten(0, 1)).unflatten(0, native.shape[:2])
    return apertures, tracker.aperture(clean_probe), native
