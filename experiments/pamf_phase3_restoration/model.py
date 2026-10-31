"""Direct time/frequency IQ restoration, without a DURA model dependency."""
import torch
from torch import nn
from torch.nn import functional as F

LENGTH = 256
WIDTH = 64
DILATIONS = (1, 2, 4, 8, 16, 32)
PARAMETERS = 196898
CONV_LINEAR_MAC = 19646464


class SpectralFilter(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.real = nn.Parameter(torch.empty(width, LENGTH // 2 + 1))
        # DC and Nyquist have no imaginary degree of freedom for a real signal.
        self.imaginary = nn.Parameter(torch.empty(width, LENGTH // 2 - 1))
        nn.init.normal_(self.real, std=.02)
        nn.init.normal_(self.imaginary, std=.02)

    def weight(self):
        return torch.complex(self.real, F.pad(self.imaginary, (1, 1)))

    def forward(self, value):
        spectrum = torch.fft.rfft(value, n=LENGTH, dim=-1, norm='ortho')
        return torch.fft.irfft(spectrum * self.weight(), n=LENGTH, dim=-1, norm='ortho')


class DualDomainBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.norm = nn.GroupNorm(8, WIDTH)
        self.time_depthwise = nn.Conv1d(WIDTH, WIDTH, 5, padding=2*dilation, dilation=dilation, groups=WIDTH)
        self.time_projection = nn.Conv1d(WIDTH, 2*WIDTH, 1)
        self.spectral = SpectralFilter(WIDTH)
        self.frequency_projection = nn.Conv1d(WIDTH, WIDTH, 1)
        self.gate = nn.Sequential(nn.Linear(WIDTH, WIDTH//4), nn.SiLU(), nn.Linear(WIDTH//4, 2*WIDTH))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, value):
        normalized = self.norm(value)
        time = F.glu(self.time_projection(F.silu(self.time_depthwise(normalized))), dim=1)
        frequency = self.frequency_projection(self.spectral(normalized))
        gate = (2*self.gate(normalized.mean(2)).sigmoid())[:, :, None]
        return value + .1*(gate[:, :WIDTH]*time + gate[:, WIDTH:]*frequency)


class TimeFrequencyRestorer(nn.Module):
    def __init__(self, seed):
        super().__init__()
        # Seed only the CPU construction stream; preserve all caller RNG states.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.stem = nn.Sequential(nn.Conv1d(2, WIDTH, 7, padding=3), nn.GroupNorm(8, WIDTH), nn.SiLU())
            self.blocks = nn.ModuleList(DualDomainBlock(dilation) for dilation in DILATIONS)
            self.output_norm = nn.GroupNorm(8, WIDTH)
            self.output = nn.Conv1d(WIDTH, 2, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        self.initialization_report = dict(architecture='P3-Restore-TF', model_seed=int(seed),
            checkpoint_sources=[], ancestors=[], training_updates=0, IQ_examples=0,
            TX_labels_read=0, F_reads=0, V_reads=0, OE_reads=0, target_reads=0,
            initialization='Random independent graph with zero residual output; no learned ancestor',
            identity_output_at_initialization=True, CPU_CUDA_rng_preserved=True)
        if sum(p.numel() for p in self.parameters()) != PARAMETERS:
            raise ValueError('Complete independent G parameter count differs')

    def forward(self, received):
        if received.ndim != 3 or received.shape[0] == 0 or received.shape[1:] != (2, LENGTH) or received.dtype != torch.float32:
            raise ValueError('Nonempty FP32[N,2,256] received IQ required')
        hidden = self.stem(received)
        for block in self.blocks:
            hidden = block(hidden)
        return received + self.output(F.silu(self.output_norm(hidden)))


def scratch_model(seed):
    return TimeFrequencyRestorer(seed).train().requires_grad_(True)


def architecture():
    return dict(name='P3-Restore-TF', parameters=PARAMETERS, ConvLinear_MAC=CONV_LINEAR_MAC,
        width=WIDTH, dilations=list(DILATIONS), packet_length=LENGTH, downsampling=False,
        rFFT_calls=6, irFFT_calls=6, FFT_length=LENGTH, transformed_real_channels_per_packet=768,
        CG_updates=0, learned_action_estimator=False, deployment_inputs='receivedIQonly',
        GN_activation_FFT_gate_cost_excluded_from_ConvLinear_MAC=True,
        initialization_checkpoint_loads=0, training_stages=1,
        performance_verified=False, training_and_inference_speed_verified=False)
