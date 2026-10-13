"""Source-L RX adversaries with first-order gradient reversal only."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from . import design as d
from experiments.cvs_report_alignment_clean.alignment import capture


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, coefficient):
        coefficient = float(coefficient)
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError('Finite nonnegative GRL coefficient required')
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def reverse(value, coefficient):
    return GradientReverse.apply(value, coefficient)


class AuxHeads(nn.Module):
    def __init__(self, dimensions=(96, 192)):
        super().__init__()
        self.levels = tuple(d.LEVELS)
        self.dimensions = tuple(dimensions)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dimension, 64), nn.ReLU(), nn.Linear(64, 30))
            for dimension in self.dimensions
        ])

    def forward(self, level, descriptor):
        if level not in self.levels:
            raise ValueError('Unregistered auxiliary feature level')
        index = self.levels.index(level)
        if descriptor.ndim != 2 or descriptor.shape[1] != self.dimensions[index]:
            raise ValueError('Auxiliary descriptor dimension differs')
        return self.layers[index](descriptor).reshape(len(descriptor), 6, 5)


def build(c, device):
    d.validate(c)
    # CPU-only initialization avoids torch.manual_seed's mutation of every CUDA
    # generator. Transfer to the selected device consumes no random numbers.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(c['model_seed'] + c['adversary']['seed_offset'])
        auxiliary = AuxHeads(c['adversary']['dimensions'])
    if sum(parameter.numel() for parameter in auxiliary.parameters()) != 22460:
        raise ValueError('Auxiliary parameter budget differs')
    if any(isinstance(module, nn.Dropout) for module in auxiliary.modules()):
        raise ValueError('Auxiliary initialization must not add stochastic forwards')
    return auxiliary.to(device)


def rx_classes(rx):
    if rx.ndim != 1 or rx.dtype != torch.long or not set(rx.tolist()) <= set(d.RX):
        raise ValueError('Only legal labeled source RX values are allowed')
    return (rx[:, None] == torch.tensor(d.RX, device=rx.device)).long().argmax(1)


def descriptor(h):
    if h.ndim != 3 or not h.is_floating_point() or not torch.isfinite(h).all():
        raise ValueError('Finite real native channel/time maps required')
    value = torch.cat((h.mean(-1), h.std(-1, unbiased=False)), 1)
    mean = value.detach().mean(0)
    scale = value.detach().std(0, unbiased=False).clamp_min(1e-4)
    return (value - mean) / scale


def domain_objective(auxiliary, maps, y, rx, c, epoch):
    d.validate(c)
    if not isinstance(epoch, int) or not 1 <= epoch <= 30:
        raise ValueError('Fixed epoch1..30 domain objective required')
    if tuple(maps) != tuple(d.LEVELS):
        raise ValueError('Actual shallow and middle feature levels differ')
    if y.ndim != 1 or y.dtype != torch.long or rx.shape != y.shape:
        raise ValueError('Legal current source-L TX/RX vectors required')
    if not len(y) or not set(y.tolist()) <= set(range(6)):
        raise ValueError('Only the six legal labeled source TX values are allowed')
    labels = rx_classes(rx)
    coefficient = 0. if c['arm'] == 'DETACHED_AUX' else d.coefficient(epoch)
    values = []; accuracies = []; details = []
    for level, dimension in zip(d.LEVELS, c['adversary']['dimensions']):
        h = maps[level]
        if len(h) != len(y) or h.device != y.device or rx.device != y.device:
            raise ValueError('Current source-L map and metadata devices/counts differ')
        z = descriptor(h)
        if z.shape != (len(y), dimension):
            raise ValueError('Actual descriptor dimension differs')
        routed = z.detach() if coefficient == 0. else reverse(z, coefficient)
        scores = auxiliary(level, routed)
        if c['arm'] == 'MARGINAL_GRL':
            selected = scores.reshape(-1, 5)
            targets = labels.repeat_interleave(6)
            group = 'all six heads equally; no TX routing'
        else:
            selected = scores[torch.arange(len(y), device=y.device), y]
            targets = labels
            group = 'legal source-L true TX selects its own head'
        value = F.cross_entropy(selected, targets)
        accuracy = (selected.argmax(1) == targets).float().mean()
        values.append(value); accuracies.append(accuracy)
        details.append(dict(level=level, map_shape=list(h.shape), descriptor_shape=list(z.shape),
                            head_shape=list(scores.shape), CE=float(value.detach()),
                            RX_accuracy=float(accuracy.detach()), TX_routing=group,
                            RX_order=list(d.RX), head_CE_count=len(targets),
                            features_detached=coefficient == 0., GRL_coefficient=coefficient))
    domain = torch.stack(values).mean()
    accuracy = torch.stack(accuracies).mean()
    metrics = dict(domain_CE=float(domain.detach()), domain_accuracy=float(accuracy.detach()),
                   GRL_coefficient=coefficient, auxiliary_CE_weight=1.,
                   domain_L_packets=len(y), domain_auxiliary_parameters=22460)
    return domain, metrics, dict(actual_layers=details, fit_role='L_only',
                                 auxiliary_query_state=False, U_labels_read=False,
                                 direct_cosW_path=False, first_order_GRL=True)


def _gradients(value, params):
    indices = [i for i, parameter in enumerate(params) if parameter.requires_grad]
    result = [None] * len(params)
    if indices and value.requires_grad:
        gradients = torch.autograd.grad(value, [params[i] for i in indices],
                                        allow_unused=True, retain_graph=True)
        for i, gradient in zip(indices, gradients):
            result[i] = gradient
    return result


def gradient_diagnostics(base, domain, mainparams, cosW):
    params = list(mainparams)
    base_gradients = _gradients(base, params)
    domain_gradients = _gradients(domain, params)
    groups = dict(main=lambda parameter: True,
                  encoder=lambda parameter: parameter is not cosW,
                  cosine_head=lambda parameter: parameter is cosW)
    result = {}
    for name, include in groups.items():
        bn = base.new_zeros(()); dn = base.new_zeros(()); dot = base.new_zeros(())
        count = 0
        for parameter, bg, dg in zip(params, base_gradients, domain_gradients):
            if not include(parameter):
                continue
            if bg is not None:
                bn = bn + bg.detach().square().sum()
            if dg is not None:
                dn = dn + dg.detach().square().sum()
                count += int(bool(dg.detach().abs().sum()))
            if bg is not None and dg is not None:
                dot = dot + (bg.detach() * dg.detach()).sum()
        cosine = float((dot / (bn * dn).sqrt().clamp_min(1e-20)).clamp(-1, 1)) if bn > 0 and dn > 0 else None
        result[name] = dict(base_norm=float(bn.sqrt()), domain_norm=float(dn.sqrt()),
                            net_added_norm=float(dn.sqrt()), ratio=float(dn.sqrt() / bn.sqrt().clamp_min(1e-12)),
                            cosine=cosine, angle_degrees=math.degrees(math.acos(cosine)) if cosine is not None else None,
                            nonzero_domain_parameter_count=count)
    head_gradient = _gradients(domain, [cosW])[0]
    result['direct_cosW'] = dict(gradient_is_none=head_gradient is None,
                               gradient_absmax=0. if head_gradient is None else float(head_gradient.detach().abs().max()),
                               path_absent_or_zero=head_gradient is None or not bool(head_gradient.detach().abs().sum()))
    return result
