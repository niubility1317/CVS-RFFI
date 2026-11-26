"""A separately trained correction conditioned on a frozen first LoRa recovery."""

import torch
from torch import nn

from experiments.pamf_lora_main.w24_packet_unet import PacketUNet


class SecondStage(nn.Module):
    def __init__(self, parent, *, conditioned, seed):
        super().__init__()
        self.parent = parent.eval().requires_grad_(False)
        self.conditioned = conditioned
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.refiner = PacketUNet(across_chirps=True, with_state=True)
        for head in (self.refiner.time_edit, self.refiner.spectral[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def propose(self, y, quality, rms, edges):
        with torch.no_grad():
            (_, first_full), = self.parent.propose(y, quality, rms, edges)
        state = y + .5 * (first_full - y) if self.conditioned else y
        _, second_full = self.refiner(y, quality, rms, edges, state=state)
        # The common renderer applies 0.5, giving 0.5 of G1 and 0.75 of the new correction.
        return (first_full, first_full + 1.5 * (second_full - y)),
