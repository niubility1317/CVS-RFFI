"""Two-stage observation-constrained restoration at the original LoRa sample rate."""

import torch

from .channel_blind import ChannelBlind, corruption_score
from .rf_noise_transport import colored_observation


class NativeRestorer(ChannelBlind):
    """Run both learned stages over all eight chirps, including the W24 boundaries."""

    def __init__(self):
        super().__init__(widths=(12, 16, 28, 40))

    @staticmethod
    def frame(received):
        if received.ndim != 3 or received.shape[1:] != (8, 1024) or not received.is_complex():
            raise ValueError('Eight aligned native 1 MHz complex LoRa chirps are required')
        rms = received.abs().square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        wave = received / rms
        reference = wave[:, :1]
        cross = (wave * reference.conj()).mean(-1, keepdim=True)
        magnitude = cross.abs()
        rotation = torch.where(magnitude > 1e-6, cross / magnitude.clamp_min(1e-6),
                               torch.ones_like(cross))
        energy = wave.abs().square().mean(-1, keepdim=True)
        confidence = magnitude / (energy * energy[:, :1]).clamp_min(1e-12).sqrt()
        return wave * rotation.conj(), confidence.clamp(0, 1), rms * rotation

    def stages(self, received):
        aligned, confidence, scale = self.frame(received)
        coarse, refined = self(aligned, confidence)
        return coarse * scale, refined * scale

    @torch.no_grad()
    def restore(self, received, threshold, *, amount=.5):
        aligned, confidence, scale = self.frame(received)
        active = corruption_score(aligned) > threshold
        corrected = received.clone()
        if active.any():
            _, refined = self(aligned[active], confidence[active])
            corrected[active] = torch.lerp(received[active], refined * scale[active], amount)
        return corrected


def seam_product(packet):
    """Complex cross-products spanning the seven physical chirp boundaries."""
    return packet[:, 1:, 0] * packet[:, :-1, -1].conj()


def source_loss(model, clean, rng, *, identity=False):
    """Supervise native waveform and boundary preservation using source IQ only."""
    received = clean if identity else colored_observation(clean, rng)
    coarse, refined = model.stages(received)
    return waveform_loss(coarse, refined, clean)


def waveform_loss(coarse, refined, clean):
    power = clean.abs().square().mean((1, 2)).clamp_min(1e-6)
    error = (refined - clean).abs().square()
    waveform = (error.mean((1, 2)) / power).mean()
    first_stage = ((coarse - clean).abs().square().mean((1, 2)) / power).mean()
    boundaries = (error[..., :24].mean((1, 2)) + error[..., -24:].mean((1, 2))) / (2 * power)
    seam_error = (seam_product(refined) - seam_product(clean)).abs().square().mean(1) / power.square()
    return waveform + .20 * first_stage + .15 * boundaries.mean() + .10 * seam_error.mean()
