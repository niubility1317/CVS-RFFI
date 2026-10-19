"""Six independent 1-D conditional restorers, without pretrained weights.

Every model returns one IQ estimate. FFT/PA/Sinc/reference views are derived
from that estimate, so independently predicted views cannot contradict it.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

NAMES = ('rescnn', 'unet', 'dual_domain', 'diffusion', 'flow', 'unrolled', 'transformer')


def pair(z):
    return torch.stack((z.real, z.imag), 1)


def complex_iq(x):
    return torch.complex(x[:, 0], x[:, 1])


def zero_output(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class Block(nn.Module):
    def __init__(self, a, b, dilation=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(a, b, 5, padding=2*dilation, dilation=dilation),
                                 nn.GroupNorm(4, b), nn.SiLU(),
                                 nn.Conv1d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU())
        self.skip = nn.Conv1d(a, b, 1) if a != b else nn.Identity()

    def forward(self, x):
        return (self.net(x) + self.skip(x)) / math.sqrt(2)


class ResCNN(nn.Module):
    def __init__(self, incoming=2, width=32, outgoing=2, residual=True):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(incoming, width, 5, padding=2),
                                  *[Block(width, width, d) for d in (1, 2, 4, 8, 16)])
        self.out = zero_output(nn.Conv1d(width, outgoing, 1))
        self.residual = residual

    def forward(self, x):
        out = self.out(self.body(x))
        return x + out if self.residual else out


class UNet(nn.Module):
    def __init__(self, incoming=2, width=24, residual=True):
        super().__init__()
        self.enc0 = Block(incoming, width)
        self.enc1 = Block(width, width*2)
        self.middle = nn.Sequential(Block(width*2, width*4), Block(width*4, width*4, 2))
        self.dec1 = Block(width*6, width*2)
        self.dec0 = Block(width*3, width)
        self.out = zero_output(nn.Conv1d(width, 2, 1))
        self.residual = residual

    def forward(self, x):
        a = self.enc0(x)
        b = self.enc1(F.avg_pool1d(a, 2))
        h = self.middle(F.avg_pool1d(b, 2))
        h = self.dec1(torch.cat((F.interpolate(h, size=b.shape[-1], mode='linear', align_corners=False), b), 1))
        h = self.dec0(torch.cat((F.interpolate(h, size=a.shape[-1], mode='linear', align_corners=False), a), 1))
        out = self.out(h)
        return x + out if self.residual else out


class DualDomain(nn.Module):
    """Time residual plus a complex FFT residual, jointly mapped to one IQ."""
    def __init__(self):
        super().__init__()
        self.time = ResCNN(width=24)
        self.frequency = ResCNN(width=24, residual=False)

    def forward(self, x):
        spectrum = torch.fft.fft(complex_iq(x), norm='ortho')
        delta = self.frequency(pair(spectrum))
        return self.time(x) + pair(torch.fft.ifft(complex_iq(delta), norm='ortho'))


class Unrolled(nn.Module):
    """Learned circular FIR proxy plus three data-consistency/proximal steps.

    The native practical simulator uses a richer boundary/receiver model.
    This candidate's circular linear proxy is an explicit approximation,
    and receives no true channel kernel, noise level or simulator metadata.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(Block(2, 24), Block(24, 32, 2), nn.AdaptiveAvgPool1d(1))
        self.channel = zero_output(nn.Linear(32, 98))
        self.prox = nn.ModuleList(ResCNN(incoming=4, width=16, residual=False) for _ in range(3))
        self.step = nn.Parameter(torch.full((3,), -1.))

    def forward(self, y):
        h = self.channel(self.encoder(y).squeeze(-1)).reshape(-1, 2, 49)
        h = complex_iq(h)
        delta = torch.zeros_like(h)
        delta[:, 0] = 1
        transfer = torch.fft.fft(h + delta, n=y.shape[-1])
        observed = torch.fft.fft(complex_iq(y), norm='ortho')
        x = y
        lipschitz = transfer.abs().square().amax(-1, keepdim=True).clamp_min(1.)
        for eta, prox in zip(self.step, self.prox):
            residual = transfer * torch.fft.fft(complex_iq(x), norm='ortho') - observed
            grad = pair(torch.fft.ifft(transfer.conj()*residual/lipschitz, norm='ortho'))
            x = x - eta.sigmoid()*grad
            x = x + prox(torch.cat((x, y), 1))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Conv1d(2, 64, 4, stride=4)
        self.position = nn.Parameter(torch.randn(1, 64, 64)*.01)
        layer = nn.TransformerEncoderLayer(64, 4, 192, dropout=0., activation='gelu',
                                            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        self.out = zero_output(nn.Linear(64, 8))

    def forward(self, x):
        if x.shape[-1] != 256:
            raise ValueError('Fixed 256-sample, four-sample token contract')
        h = self.encoder(self.embed(x).transpose(1, 2) + self.position)
        delta = self.out(h).reshape(len(x), 64, 2, 4).permute(0, 2, 1, 3).reshape_as(x)
        return x + delta


class ConditionalDiffusion(nn.Module):
    """64-step cosine diffusion, v objective, 16-step deterministic DDIM.

    Sampling randomness is supplied explicitly by the predictor, once per
    physical sample. No averaging, best-of-N choice or test-time fitting.
    """
    def __init__(self, steps=64, sampling_steps=16):
        super().__init__()
        self.steps, self.sampling_steps = steps, sampling_steps
        self.denoiser = UNet(incoming=12, width=24, residual=False)
        t = torch.arange(steps+1, dtype=torch.float64)/steps
        curve = torch.cos((t+.008)/1.008*torch.pi/2).square()
        beta = (1-curve[1:]/curve[:-1]).clamp(0., .999)
        self.register_buffer('alpha_bar', (1-beta).cumprod(0).float())

    def velocity(self, noisy, received, t):
        frequencies = noisy.new_tensor([1., 2., 4., 8.])
        angle = (t.to(noisy.dtype)/self.steps)[:, None]*frequencies[None]*torch.pi
        planes = torch.cat((angle.sin(), angle.cos()), 1)[:, :, None].expand(-1, -1, noisy.shape[-1])
        return self.denoiser(torch.cat((noisy, received, planes), 1))

    def training_prediction(self, clean, received, generator=None):
        t = torch.randint(self.steps, (len(clean),), device=clean.device, generator=generator)
        noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator)
        a = self.alpha_bar[t, None, None].sqrt()
        b = (1-self.alpha_bar[t, None, None]).sqrt()
        noisy = a*clean + b*noise
        target = a*noise - b*clean
        velocity = self.velocity(noisy, received, t)
        return a*noisy-b*velocity, F.mse_loss(velocity, target)

    def forward(self, received, noise):
        if noise is None or noise.shape != received.shape:
            raise ValueError('One fixed noise tensor per received IQ sample required')
        x = noise
        indices = torch.linspace(self.steps-1, 0, self.sampling_steps).round().long().tolist()
        for i, index in enumerate(indices):
            t = torch.full((len(x),), index, dtype=torch.long, device=x.device)
            a = self.alpha_bar[index].sqrt()
            b = (1-self.alpha_bar[index]).sqrt()
            velocity = self.velocity(x, received, t)
            estimate = a*x-b*velocity
            epsilon = b*x+a*velocity
            if i == len(indices)-1:
                x = estimate
            else:
                following = self.alpha_bar[indices[i+1]]
                x = following.sqrt()*estimate + (1-following).sqrt()*epsilon
        return x


class ConditionalFlow(nn.Module):
    """Paired conditional flow from received+.02*noise to the clean endpoint.

    Same U-Net as diffusion, 16 fixed Euler evaluations. Unlike a general
    posterior flow, this source-paired straight path is a research adaptation.
    """
    def __init__(self, sampling_steps=16):
        super().__init__()
        self.network = ConditionalDiffusion()
        self.sampling_steps = sampling_steps
        self.sigma = .02

    def training_prediction(self, clean, received, generator=None):
        noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator)
        start = received+self.sigma*noise
        t = torch.rand((len(clean),), device=clean.device, generator=generator)
        point = (1-t[:, None, None])*start+t[:, None, None]*clean
        velocity = self.network.velocity(point, received, t*self.network.steps)
        return point+(1-t[:, None, None])*velocity, F.mse_loss(velocity, clean-start)

    def forward(self, received, noise):
        if noise is None or noise.shape != received.shape:
            raise ValueError('One fixed noise tensor per received IQ sample required')
        point = received+self.sigma*noise
        for index in range(self.sampling_steps):
            t = received.new_full((len(received),), index/self.sampling_steps*self.network.steps)
            point = point+self.network.velocity(point, received, t)/self.sampling_steps
        return point


def build(name):
    factories = dict(rescnn=ResCNN, unet=UNet, dual_domain=DualDomain,
                     unrolled=Unrolled, transformer=Transformer, diffusion=ConditionalDiffusion, flow=ConditionalFlow)
    if name not in factories:
        raise ValueError('Unregistered restoration candidate')
    return factories[name]()
