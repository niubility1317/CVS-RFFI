"""Packet position features fused before each external method's original heads."""
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CHIRPS = tuple(range(8))
SAMPLES = 384
EDGE_GAIN = .10


def packet_chirps(x):
    if x.ndim != 4 or tuple(x.shape[1:]) != (len(CHIRPS), 2, SAMPLES):
        raise ValueError('Full preamble requires [B,8,2,384] physical IQ')
    return x.flatten(0, 1)


class PacketEdges(nn.Module):
    """Apply the two ordered boundary differences along the feature dimension."""

    def __init__(self, width):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, width),
                                     nn.LayerNorm(width))

    def forward(self, features, batch):
        chirps = features.unflatten(0, (batch, len(CHIRPS)))
        middle = chirps[:, 1:-1].mean(1)
        edges = torch.cat((chirps[:, 0] - middle, chirps[:, -1] - middle), dim=-1)
        return chirps.mean(1) + EDGE_GAIN * self.project(edges)


class PartitionedPacketEncoder(nn.Module):
    """Keep emitter and receiver features in their native disjoint spaces."""

    def __init__(self, encoder, emitter_dim, receiver_dim):
        super().__init__()
        self.encoder = encoder
        self.dims = (emitter_dim, receiver_dim)
        self.emitter_edges = PacketEdges(emitter_dim)
        self.receiver_edges = PacketEdges(receiver_dim)

    def forward(self, x):
        emitter, receiver = self.encoder(packet_chirps(x)).split(self.dims, dim=1)
        return torch.cat((self.emitter_edges(emitter, len(x)), self.receiver_edges(receiver, len(x))), dim=1)


class PacketFAN(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.edges = PacketEdges(encoder.cls_head.in_features)

    def forward(self, x):
        tokens, _ = self.encoder(packet_chirps(x))
        tokens = self.edges(tokens, len(x))
        return tokens, self.encoder.cls_head(tokens.mean(1))


class PacketSDG(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.edges = PacketEdges(encoder.fc.in_features)

    def forward(self, x):
        features, _ = self.encoder(packet_chirps(x))
        features = F.normalize(self.edges(features, len(x)))
        return features, self.encoder.fc(features)


def build(c):
    from experiments.phase1_dg_nativebudget import pipeline as native
    family, emitters, receivers = c['family'], c['num_classes'], c['num_domains']
    if family == 'riei':
        from baselines.riei_fd.model import RIEIModel
        net = RIEIModel(emitters, receivers, fed_variant='short_stem1d')
        net.fed = PartitionedPacketEncoder(net.fed, net.emitter_feature_dim, net.receiver_feature_dim)
        return net
    if family == 'drift':
        from baselines.drift.model import DRIFTModel
        net = DRIFTModel(emitters, receivers, domain_discriminator_layers=2)
        net.encoder = PartitionedPacketEncoder(net.encoder, net.split_dim, net.embedding_dim - net.split_dim)
        return net
    if family == 'ibrff':
        sys.path.insert(0, c['ibrff_model_root'])
        from backbones.FAN import FAN
        return PacketFAN(FAN(seq_len=SAMPLES, in_chans=2, patch_size=32, embed_dim=128, mlp_ratio=2.,
                            dr=.1, num_classes=emitters, bottleneck_type='DSCNN', use_SAF=True, use_FAF=True))
    if family == 'sdg':
        author = native.author_sdg(c)
        # Importing the author's module seeds its example program.
        native.seed(c['model_seed'])
        return PacketSDG(author.MACNN(in_channels=2, channels=8, num_classes=emitters))
    raise ValueError(f'Unknown full-preamble method: {family}')


def identity(net, family, x):
    if family == 'riei':
        return net.ec(net.fed(x)[:, :net.emitter_feature_dim])
    if family == 'drift':
        return net.tx_classifier(net.encoder(x)[:, :net.split_dim])
    return net(x)[1]


class SDGPacketViews(torch.utils.data.Dataset):
    """Keep packet position fixed and replay each sampled SDG IQ transform per chirp."""

    def __init__(self, source, author, seed):
        from experiments.practical_full_2x2.author_imports import compatible_sdg_author
        compatible_sdg_author(author)
        self.source = source
        self.author = author
        self.preprocess = author.transforms.ToTensor()
        self.rng = np.random.RandomState(seed).get_state()

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        x, y, rx = self.source[index]
        outer = np.random.get_state()
        main = []
        try:
            for iq in x.numpy():
                np.random.set_state(self.rng)
                main.append(torch.as_tensor(self.author.aug(iq.copy(), self.preprocess, [1], [])[0]))
            self.rng = np.random.get_state()
        finally:
            np.random.set_state(outer)
        return torch.stack((x, torch.stack(main))), y, rx


def optimizers(net, c):
    if c['family'] == 'riei':
        heads = list(net.ec.parameters()) + list(net.rc.parameters())
        return torch.optim.Adam(heads, lr=c['lr']), torch.optim.Adam(net.fed.parameters(), lr=c['lr'])
    return torch.optim.Adam(net.parameters(), lr=c['lr']), None


def train_batch(net, c, x, y, rx, optimizer, fed, stage, contrast=None, hsic=None):
    from experiments.phase1_dg_nativebudget.pipeline import finite
    family = c['family']
    net.train()
    optimizer.zero_grad()
    auxiliary = {}
    if family == 'riei':
        fed.zero_grad()
        z_e, z_r = net.fed(x).split((net.emitter_feature_dim, net.receiver_feature_dim), dim=1)
        ce = F.cross_entropy(net.ec(z_e), y)
        receiver = F.cross_entropy(net.rc(z_r), rx)
        auxiliary['receiver_ce'] = float(receiver.detach())
        loss = ce + receiver
    elif family == 'drift':
        from baselines.drift.losses import compute_drift_loss
        terms = compute_drift_loss(net(x), y, rx, lambda_grl=1., lambda_center=.01, lambda_mse=.02,
                                   normalize_features_for_mse=True, center_mode='batch')
        ce, loss = terms['loss_ce_tx'], terms['loss']
        auxiliary.update({k: float(v.detach()) for k, v in terms.items() if k.startswith('loss_')})
    elif family == 'ibrff':
        tokens, logits = net(x)
        ce = loss = F.cross_entropy(logits, y)
        if stage == 2:
            penalty = .04 * hsic(tokens.flatten(1), x.flatten(1))
            auxiliary['weighted_hsic'] = float(penalty.detach())
            loss = loss + penalty
    elif family == 'sdg':
        batch = len(y)
        features, logits = net(x.transpose(0, 1).flatten(0, 1))
        ce = F.cross_entropy(logits[batch:], y)
        identity_contrast = contrast(features.unsqueeze(1), y.repeat(2), adv=False)
        domains = torch.arange(2, device=y.device).repeat_interleave(batch)
        view_contrast = 100. * contrast(features.unsqueeze(1), domains, adv=True)
        auxiliary.update(identity_contrast=float(identity_contrast.detach()),
                         weighted_view_contrast=float(view_contrast.detach()))
        loss = ce + identity_contrast + view_contrast
    else:
        raise ValueError(f'Unknown full-preamble method: {family}')
    loss.backward()
    finite(net, loss)
    optimizer.step()
    loss_value = float(loss.detach())
    if family == 'riei':
        from baselines.riei_fd.losses import entropy_from_logits, mutual_independence_loss
        fed.step()
        heads = tuple(net.ec.parameters()) + tuple(net.rc.parameters())
        for param in heads:
            param.requires_grad_(False)
        try:
            fed.zero_grad()
            output = net(x)
            mi = mutual_independence_loss(output['z_e'], output['z_r'], mode='cosine_abs', reduction='mean')
            entropy = (entropy_from_logits(output['cross_emitter_logits'])
                       + entropy_from_logits(output['cross_receiver_logits']))
            disentanglement = 1.2 * mi - 1.2 * entropy
            disentanglement.backward()
            finite(net, disentanglement)
            fed.step()
            auxiliary.update(mutual_independence=float(mi.detach()), cross_head_entropy=float(entropy.detach()))
            loss_value += float(disentanglement.detach())
        finally:
            for param in heads:
                param.requires_grad_(True)
    return dict(loss=loss_value, identity_ce=float(ce.detach()), **auxiliary)
