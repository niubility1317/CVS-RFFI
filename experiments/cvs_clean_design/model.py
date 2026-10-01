"""Physical inductive biases with no auxiliary objective or augmentation."""
import torch
from torch import nn
import torch.nn.functional as F
from experiments.cvs_identity_ce.model import IdentityOnlyCVS

BASELINES = ('native', 'cvcnn', 'real_cnn', 'resnet1d')
VARIANTS = ('native', 'orthogonal_pa', 'moment_pool', 'orthogonal_moment', 'shared_complex', 'cvcnn', 'real_cnn', 'resnet1d')


class RealCNN(nn.Module):
    """Standard real I/Q 1D CNN; same input and CE budget as the complex CNN."""
    def __init__(self):
        super().__init__();layers=[];cin=2
        for cout,kernel in ((64,7),(128,5),(256,3)):
            layers.extend([nn.Conv1d(cin,cout,kernel,padding=kernel//2,bias=False),
                           nn.BatchNorm1d(cout),nn.ReLU(),nn.AvgPool1d(2)])
            cin=cout
        self.trunk=nn.Sequential(*layers,nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.embedding=nn.Sequential(nn.Linear(256,128),nn.ReLU())
        self.classifier=nn.Linear(128,6)

    def forward(self,x):return self.classifier(self.embedding(self.trunk(x)))


class ResidualBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.path=nn.Sequential(nn.Conv1d(cin,cout,3,stride=stride,padding=1,bias=False),nn.BatchNorm1d(cout),nn.ReLU(),
                                nn.Conv1d(cout,cout,3,padding=1,bias=False),nn.BatchNorm1d(cout))
        self.skip=nn.Sequential(nn.Conv1d(cin,cout,1,stride=stride,bias=False),nn.BatchNorm1d(cout)) if cin!=cout or stride!=1 else nn.Identity()

    def forward(self,x):return F.relu(self.path(x)+self.skip(x))


class ResNet1D(nn.Module):
    """Six basic residual blocks with small widths; a project baseline, not ResNet18."""
    def __init__(self):
        super().__init__()
        self.trunk=nn.Sequential(nn.Conv1d(2,32,7,padding=3,stride=2,bias=False),nn.BatchNorm1d(32),nn.ReLU(),
            ResidualBlock(32,32),ResidualBlock(32,32),ResidualBlock(32,64,2),ResidualBlock(64,64),
            ResidualBlock(64,96,2),ResidualBlock(96,96),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.classifier=nn.Linear(96,6)

    def forward(self,x):return self.classifier(self.trunk(x))


class OrthogonalMemoryLift(nn.Module):
    """Packet-local Gram-Schmidt of odd PA bases; interleaved native layout."""
    def __init__(self, memory_depth=4, orders=(1, 3, 5), clip=2.0):
        super().__init__()
        self.memory_depth, self.orders, self.clip = memory_depth, tuple(orders), clip

    def forward(self, x):
        # Radial clipping commutes with global phase rotation, unlike I/Q clipping.
        power = x.square().sum(1, keepdim=True)
        z = x * (self.clip / power.clamp_min(1e-8).sqrt()).clamp(max=1.0)
        power = z.square().sum(1, keepdim=True)
        bases = []
        for order in self.orders:
            b = z * power.pow((order - 1) // 2)
            for previous in bases:
                # These odd monomials have real Hermitian inner products.
                coefficient = (b * previous).sum((1, 2), keepdim=True) / previous.square().sum((1, 2), keepdim=True).clamp_min(1e-8)
                b = b - coefficient * previous
            # Smooth energy floor limits amplification of nearly degenerate bases.
            b = b / (b.square().sum(1, keepdim=True).mean(-1, keepdim=True) + 1e-4).sqrt()
            bases.append(b)
        delayed = []
        for delay in range(self.memory_depth):
            for b in bases:
                delayed.append(F.pad(b, (delay, 0))[..., :x.shape[-1]])
        return torch.cat(delayed, dim=1)


class MomentPool(nn.Module):
    """Retain first and second moments without widening native projections."""
    def __init__(self, channels):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        deviation = ((x - mean).square().mean(-1, keepdim=True) + 1e-6).sqrt()
        return mean + self.mix.tanh() * deviation


class ComplexSeparable(nn.Module):
    """Complex depthwise temporal filter, complex pointwise mixer, radial gate."""
    def __init__(self, cin, cout, kernel=7, dilation=1, pool=2):
        super().__init__()
        self.cin = cin
        pad = (kernel // 2) * dilation
        self.dr = nn.Conv1d(cin, cin, kernel, padding=pad, dilation=dilation, groups=cin, bias=False)
        self.di = nn.Conv1d(cin, cin, kernel, padding=pad, dilation=dilation, groups=cin, bias=False)
        self.pr = nn.Conv1d(cin, cout, 1, bias=False)
        self.pi = nn.Conv1d(cin, cout, 1, bias=False)
        self.gain = nn.Parameter(torch.ones(1, cout, 1))
        self.offset = nn.Parameter(torch.zeros(1, cout, 1))
        self.pool = pool

    def forward(self, x):
        r, i = x.chunk(2, 1)
        r, i = self.dr(r) - self.di(i), self.dr(i) + self.di(r)
        r, i = self.pr(r) - self.pi(i), self.pr(i) + self.pi(r)
        energy = (r.square() + i.square()).mean((1, 2), keepdim=True)
        r, i = r / (energy + 1e-6).sqrt(), i / (energy + 1e-6).sqrt()
        magnitude = (r.square() + i.square() + 1e-6).sqrt()
        gate = torch.sigmoid(self.gain * magnitude + self.offset)
        y = torch.cat((r * gate, i * gate), 1)
        return F.avg_pool1d(y, self.pool) if self.pool > 1 else y


class SharedComplexCVS(nn.Module):
    """Compact shared PA/IQ trunk with phase-invariant relative-phase readout."""
    def __init__(self):
        super().__init__()
        self.lift = OrthogonalMemoryLift(memory_depth=1)
        self.blocks = nn.Sequential(ComplexSeparable(3, 32), ComplexSeparable(32, 64, dilation=2),
                                    ComplexSeparable(64, 96, dilation=3, pool=1))
        self.embedding = nn.Sequential(nn.LayerNorm(96 * 8), nn.Linear(96 * 8, 192), nn.SiLU())
        self.class_weight = nn.Parameter(torch.empty(6, 192))
        nn.init.xavier_uniform_(self.class_weight)

    def features(self, x):
        basis = self.lift(x)
        # Lift is interleaved I/Q, complex trunk requires all real then all imag.
        h = self.blocks(torch.cat((basis[:, 0::2], basis[:, 1::2]), 1))
        r, i = h.chunk(2, 1)
        amp = (r.square() + i.square() + 1e-6).sqrt()
        values = [amp.mean(-1), amp.std(-1, unbiased=False)]
        for lag in (1, 2):
            values += [(r[..., lag:] * r[..., :-lag] + i[..., lag:] * i[..., :-lag]).mean(-1),
                       (i[..., lag:] * r[..., :-lag] - r[..., lag:] * i[..., :-lag]).mean(-1)]
        # Complex channel cross-covariance retains relative phase across filters.
        rr, ii = r.roll(1, 1), i.roll(1, 1)
        values += [(r * rr + i * ii).mean(-1), (i * rr - r * ii).mean(-1)]
        return self.embedding(torch.cat(values, 1))

    def forward(self, x):
        return 30.0 * F.linear(F.normalize(self.features(x), dim=1), F.normalize(self.class_weight, dim=1))


def build(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered architecture: ' + str(variant))
    if variant == 'shared_complex':
        return SharedComplexCVS()
    if variant=='real_cnn':return RealCNN()
    if variant=='resnet1d':return ResNet1D()
    if variant == 'cvcnn':
        from baselines.cvcnn_ce.model import BasicCVCNN
        return BasicCVCNN(num_classes=6, base_channels=32, embedding_dim=128, dropout=0.0)
    model = IdentityOnlyCVS()
    backbone = model.id_backbone
    if variant in ('orthogonal_pa', 'orthogonal_moment'):
        old = backbone.pa_lift
        backbone.pa_lift = OrthogonalMemoryLift(old.memory_depth, old.orders, old.clip)
    if variant in ('moment_pool', 'orthogonal_moment'):
        for pool, projection in [('t_pool', 't_proj'), ('f_pool', 'f_proj'), ('pa_pool', 'pa_proj')]:
            proj = getattr(backbone, projection)
            # Native PA projection is Linear; preserve the original shape/head.
            linear = proj if isinstance(proj, nn.Linear) else next(m for m in proj.modules() if isinstance(m, nn.Linear))
            setattr(backbone, pool, MomentPool(linear.in_features))
    return model
