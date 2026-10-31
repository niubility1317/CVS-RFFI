"""A full-rate IQ residual path beside the original SIG-conditioned decoder."""
import torch
from torch import nn
from .signature_context_prior import SignatureContextPrior, BOUND

WIDTH = 12
DILATIONS = (2, 4, 8)
SEED_OFFSET = 2026103131
ADDED = {'plain': 2068, 'conditioned': 25252}
PARAMETERS = {'plain': 346363, 'conditioned': 369547}
MACS = {'plain': 25599528, 'conditioned': 25622568}


class FullRateResidual(nn.Module):
    def __init__(self, seed, conditioned):
        super().__init__()
        self.conditioned = bool(conditioned)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.input = nn.Sequential(nn.Conv1d(9, WIDTH, 3, padding=1), nn.GroupNorm(3, WIDTH), nn.SiLU())
            self.blocks = nn.ModuleList(nn.Sequential(
                nn.Conv1d(WIDTH, WIDTH, 3, padding=d, dilation=d, groups=WIDTH, bias=False),
                nn.Conv1d(WIDTH, WIDTH, 1), nn.GroupNorm(3, WIDTH)) for d in DILATIONS)
            self.output = nn.Conv1d(WIDTH, 2, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
            self.conditioner = nn.Linear(160, 2 * WIDTH * len(DILATIONS)) if self.conditioned else None
            if self.conditioner is not None:
                nn.init.zeros_(self.conditioner.weight)
                nn.init.zeros_(self.conditioner.bias)

    def forward(self, value, signature):
        modulation = self.conditioner(signature) if self.conditioned else None
        hidden = self.input(value)
        for index, block in enumerate(self.blocks):
            delta = block(hidden)
            if modulation is not None:
                offset = index * 2 * WIDTH
                scale = 1 + BOUND * modulation[:, offset:offset + WIDTH].tanh()
                shift = BOUND * modulation[:, offset + WIDTH:offset + 2 * WIDTH].tanh()
                delta = delta * scale[:, :, None] + shift[:, :, None]
            hidden = hidden + torch.nn.functional.silu(delta)
        return self.output(hidden)


class FullRateSignaturePrior(SignatureContextPrior):
    def __init__(self, prior, seed, mode):
        nn.Module.__init__(self)
        if type(prior) is not SignatureContextPrior or not prior.decoder_context or mode not in ADDED:
            raise ValueError('Original main SIG context prior and fixed full-rate mode required')
        self.encoder, self.decode, self.out = prior.encoder, prior.decode, prior.out
        self.signature_projection, self.signature_conditioner = prior.signature_projection, prior.signature_conditioner
        self.decoder_context = True
        self.conditional = False
        self.full_rate_mode = mode
        self.full_rate = FullRateResidual(seed, mode == 'conditioned')

    def forward_with_signature(self, value):
        if value.ndim != 3 or value.shape[1:] != (9, 256):
            raise ValueError('Actual prior Bx9x256 input required')
        features = self.encoder(value)
        signature = self.signature_projection(torch.nn.functional.adaptive_avg_pool1d(features[1], 8).flatten(1))
        # The model applies its existing SCALE to the sum, before its existing CG.
        return self._decode(features, signature) + self.full_rate(value, signature), signature


def attach(model, seed, mode):
    if mode not in ADDED or len(model.prior) != 2 or sum(p.numel() for p in model.parameters()) != 344295:
        raise ValueError('One original main two-prior SIG-context G required')
    model.prior = nn.ModuleList(FullRateSignaturePrior(prior, int(seed) + SEED_OFFSET + i, mode)
                               for i, prior in enumerate(model.prior))
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[mode]:
        raise ValueError('Full-rate G parameter accounting differs')
    model.initialization_report = dict(model.initialization_report,
        full_rate_parameters=ADDED[mode], full_rate_mode=mode, full_rate_width=WIDTH,
        full_rate_dilations=list(DILATIONS), full_rate_zero_output_init=True,
        full_rate_conditioner_zero_init=mode == 'conditioned', full_rate_conditioner_bound=BOUND,
        full_rate_seed_offset=SEED_OFFSET, full_rate_CPU_CUDA_rng_preserved=True,
        full_rate_input='same9channelpriorinput', full_rate_inside_existing_SCALE=True)
    return model
