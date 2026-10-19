"""ResNet default-head routes and source-domain regularizers.

The linear-head CE gradient is evaluated analytically, then tested against
per-example autograd. No fitted diagnostic head or backbone freezing is used.
"""
import torch
from torch import nn
import torch.nn.functional as F
from experiments.standard_ce_baselines.model import ResNet


def mix_statistics(x, y, rx, *, identity_protected, probability=.5, alpha=.1):
    if torch.rand((), device=x.device) >= probability:
        return x, 0
    candidates = rx[:, None] != rx[None, :]
    if identity_protected:
        candidates = candidates & (y[:, None] == y[None, :])
    eligible = candidates.any(1)
    random_scores = torch.rand(candidates.shape, device=x.device).masked_fill(~candidates, -1)
    paired = random_scores.argmax(1)
    paired = torch.where(eligible, paired, torch.arange(len(x), device=x.device))
    mean = x.mean(2, keepdim=True)
    sigma = (x.var(2, keepdim=True, unbiased=True) + 1e-6).sqrt()
    mean, sigma = mean.detach(), sigma.detach()
    beta = torch.distributions.Beta(alpha, alpha)
    weight = beta.sample((len(x), 1, 1)).to(x.device)
    mixed_mean = weight * mean + (1 - weight) * mean[paired]
    mixed_sigma = weight * sigma + (1 - weight) * sigma[paired]
    result = (x - mean) / sigma * mixed_sigma + mixed_mean
    return result, int(eligible.sum())


class RoutedResNet(ResNet):
    """The baseline architecture, with no added inference parameters."""
    def __init__(self, classes=6, route='erm'):
        super().__init__((2, 2, 2, 2))
        if classes != 6:
            # Cardinality is chosen before training, never replaced after fitting.
            self.classifier = nn.Linear(512, classes)
        self.route = route
        self.context = None
        self.mixstyle_counts = [0, 0]

    def encode(self, x):
        self.mixstyle_counts = [0, 0]
        for index, module in enumerate(self.features):
            x = module(x)
            if (index in (5, 7) and self.training and self.context is not None
                and self.route in ('mixstyle', 'mixstyle_id')):
                y, rx = self.context
                x, count = mix_statistics(x, y, rx,
                    identity_protected=self.route == 'mixstyle_id')
                self.mixstyle_counts[0 if index == 5 else 1] = count
        return x.mean(-1)

    def forward(self, x):
        z = self.encode(x)
        return z, self.classifier(z)


def linear_ce_sample_gradients(features, logits, labels):
    residual = logits.softmax(1) - F.one_hot(labels, logits.shape[1]).to(logits.dtype)
    weight = (residual[:, :, None] * features[:, None, :]).flatten(1)
    return torch.cat((weight, residual), 1)


class FishrVariance:
    def __init__(self, domains, ema=.95):
        self.domains = domains
        self.ema = ema
        self.previous = [None] * domains

    def __call__(self, features, logits, labels, rx):
        gradients = linear_ce_sample_gradients(features, logits, labels)
        current = []
        for domain in range(self.domains):
            g = gradients[rx == domain]
            if len(g) < 2:
                raise ValueError('Fishr needs at least two samples per source RX')
            variance = (g - g.mean(0, keepdim=True)).square().mean(0)
            prior = torch.zeros_like(variance) if self.previous[domain] is None else self.previous[domain]
            smoothed = self.ema * prior + (1 - self.ema) * variance
            self.previous[domain] = smoothed.detach().clone()
            current.append(smoothed / (1 - self.ema))
        matrix = torch.stack(current)
        return (matrix - matrix.mean(0, keepdim=True)).square().mean()


def source_coral(features, rx, domains):
    means, covariances = [], []
    for domain in range(domains):
        z = features[rx == domain]
        if len(z) < 2:
            raise ValueError('CORAL requires two samples per source domain')
        mean = z.mean(0)
        centered = z - mean
        means.append(mean)
        covariances.append(centered.T @ centered / (len(z) - 1))
    values = []
    width = features.shape[1]
    for i in range(domains):
        for j in range(i + 1, domains):
            values.append((means[i] - means[j]).square().mean() +
                (covariances[i] - covariances[j]).square().sum() / (4 * width * width))
    if not values:
        raise ValueError('CORAL needs multiple source RX')
    return torch.stack(values).mean()


def balanced_indices(labels, receivers, classes, domains, batch_size, generator):
    if batch_size % (classes * domains):
        raise ValueError('Budget must provide equal samples per TX and RX')
    quota = batch_size // (classes * domains)
    selected = []
    for receiver in range(domains):
        for label in range(classes):
            pool = torch.where((labels == label) & (receivers == receiver))[0]
            if not len(pool):
                raise ValueError('A source TX/RX cell has no labeled packet')
            draws = torch.randint(len(pool), (quota,), generator=generator)
            selected.append(pool[draws])
    return torch.cat(selected)
