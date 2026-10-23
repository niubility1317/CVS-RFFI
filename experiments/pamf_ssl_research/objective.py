"""EMA soft targets and labeled-only cross-receiver identity references."""
from copy import deepcopy
import torch
from torch.nn import functional as F


def confidence_weight(confidence, mean, variance, floor=.7):
    """Detached one-sided Gaussian, with a fixed low-confidence exclusion."""
    delta = (confidence - mean).clamp_max(0.)
    weight = torch.exp(-delta.square() / (2 * variance.clamp_min(1e-4)))
    return (weight * (confidence >= floor)).detach()


class ReceiverPrototypes:
    def __init__(self, receivers, classes, dimension, device, momentum=.9):
        self.receivers = tuple(receivers)
        self.centers = torch.zeros(len(receivers), classes, dimension, device=device)
        self.counts = torch.zeros(len(receivers), classes, dtype=torch.long, device=device)
        self.momentum = momentum

    @torch.no_grad()
    def update(self, features, labels, receivers):
        if (labels < 0).any() or (labels >= self.centers.shape[1]).any():
            raise ValueError('Only visible labeled source TX may update prototypes')
        if set(receivers.tolist()) - set(self.receivers):
            raise ValueError('Non-source receiver')
        features = F.normalize(features.detach(), dim=1)
        for j, rx in enumerate(self.receivers):
            for cls in range(self.centers.shape[1]):
                mask = (receivers == rx) & (labels == cls)
                if not mask.any():
                    continue
                center = features[mask].mean(0)
                if self.counts[j, cls]:
                    center = self.momentum * self.centers[j, cls] + (1 - self.momentum) * center
                self.centers[j, cls] = F.normalize(center, dim=0)
                self.counts[j, cls] += int(mask.sum())

    @torch.no_grad()
    def distribution(self, features, receivers, fallback, temperature=.1):
        if set(receivers.tolist()) - set(self.receivers):
            raise ValueError('Non-source receiver')
        ready = (self.counts > 0).all(1)
        other = torch.tensor(self.receivers, device=receivers.device)[None, :] != receivers[:, None]
        eligible = other & ready[None, :]
        valid = eligible.sum(1) >= 2
        similarity = torch.einsum('bd,rcd->brc', F.normalize(features, dim=1), self.centers)
        per_rx = (similarity / temperature).softmax(-1)
        probability = (per_rx * eligible[:, :, None]).sum(1) / eligible.sum(1).clamp_min(1)[:, None]
        probability = torch.where(valid[:, None], probability, fallback)
        centers = torch.einsum('br,rcd->bcd', eligible.float(), self.centers)
        centers = centers / eligible.sum(1).clamp_min(1)[:, None, None]
        return probability, centers, valid


class Objective:
    def __init__(self, student, config):
        self.c = config
        device = next(student.parameters()).device
        self.teacher = deepcopy(student.id_backbone.encoder).to(device).eval().requires_grad_(False)
        self.mean = torch.tensor(.95, device=next(student.parameters()).device)
        self.variance = torch.tensor(.01, device=self.mean.device)
        self.bank = ReceiverPrototypes(config['source_rxs'], 6, 160, self.mean.device,
                                       config['ssl']['prototype_momentum'])
        self.ema_updates = self.seen = self.physical_seen = self.calls = 0
        self.effective = 0.
        self.per_class = torch.zeros(6, device=self.mean.device)
        self.metrics = {}
        self.labeled = None
        self.view_anchor_calls = self.view_anchor_physical_seen = 0

    @torch.no_grad()
    def after_update(self, student):
        decay = self.c['ssl']['teacher_decay']
        source = student.id_backbone.encoder
        for teacher, value in zip(self.teacher.parameters(), source.parameters()):
            teacher.lerp_(value.detach(), 1. - decay)
        for teacher, value in zip(self.teacher.buffers(), source.buffers()):
            teacher.copy_(value.detach())
        self.ema_updates += 1

    def loss(self, student, device):
        try:
            batch = next(student._cd_U_iter)
        except StopIteration:
            student._cd_U_iter = iter(student._cd_U_loader)
            batch = next(student._cd_U_iter)
        if not bool((batch[1] == -1).all()):
            raise ValueError('U TX truth exposed')
        x = batch[0].to(device)
        domain = batch[2].to(device)
        meta = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch[3].items()}
        settings = self.c['ssl']
        rx = meta['rx_i'].long()
        with torch.no_grad():
            z = self.teacher.features(x)
            p = self.teacher.classify_features(z).softmax(1)
            confidence = p.max(1).values
            decay = settings['moment_decay']
            self.mean.lerp_(confidence.mean(), 1. - decay)
            self.variance.lerp_(confidence.var(unbiased=False), 1. - decay)
            weight = confidence_weight(confidence, self.mean, self.variance, settings['confidence_floor'])
            prototype_valid = torch.zeros(len(x), dtype=torch.bool, device=device)
            centers = None
            if self.c['ssl_method'] == 'RXPSM':
                if self.labeled is None:
                    raise ValueError('Current visible L batch required for prototype references')
                lx, ly, lrx = self.labeled
                lz = self.teacher.features(lx.detach())
                self.bank.update(lz, ly, lrx)
                proto, centers, prototype_valid = self.bank.distribution(z, rx, p, settings['prototype_temperature'])
                agreement = p.argmax(1) == proto.argmax(1)
                weight *= torch.where(prototype_valid, .25 + .75 * agreement.float(), 1.)
                p = torch.where(prototype_valid[:, None],
                                (1. - settings['prototype_mix']) * p + settings['prototype_mix'] * proto, p)
            target = p.pow(1. / settings['pseudo_temperature'])
            target /= target.sum(1, keepdim=True)
            # Strong views retain one source physical record and never consult its TX.
            n = len(x)
            mask = (torch.arange(n, device=device) + self.current_step).remainder(2).bool()
            strong = x + .015 * torch.randn_like(x)
            physical_meta = {k: v[mask] if torch.is_tensor(v) and v.ndim and len(v) == n else v
                             for k, v in meta.items()}
            strong[mask] = student._cd_practical(x[mask], physical_meta, self.current_epoch, self.current_step)
        student_forward = getattr(self, 'unlabeled_student_forward', None)
        output = (student_forward(student, strong, domain, target, weight, rx) if student_forward is not None
                  else student.forward_identity_only(strong, domain_labels=domain))
        ce = -(target * output['tx_logits'].log_softmax(1)).sum(1)
        feature = ce * 0.
        if centers is not None:
            anchor = (centers * target[:, :, None]).sum(1).detach()
            feature = (1 - F.cosine_similarity(output['z_id'], anchor, dim=1)) * prototype_valid
        ramp = min((self.current_epoch - self.c['U_start_epoch'] + 1) / settings['ramp_epochs'], 1.)
        loss = settings['unsupervised_weight'] * ramp * (weight * (ce + settings['prototype_feature_weight'] * feature)).mean()
        view_settings = self.c.get('view_anchor')
        view_loss = None
        if view_settings and self.current_epoch >= view_settings['start_epoch']:
            from .view_anchor import physical_alignment
            view_loss = view_settings['weight'] * physical_alignment(output['z_id'], z, weight, mask)
            loss = loss + view_loss
        extension = getattr(self, 'unlabeled_feature_loss', None)
        if extension is not None:
            loss = loss + extension(output, target, weight, rx)
        self.seen += n
        physical_count = int(mask.sum())
        self.physical_seen += physical_count
        if view_loss is not None:
            self.view_anchor_calls += 1
            self.view_anchor_physical_seen += physical_count
        self.effective += float(weight.sum())
        self.calls += 1
        self.per_class.scatter_add_(0, target.argmax(1), weight)
        student._cd_U_seen += n
        student._cd_U_accepted += int((weight >= .5).sum())
        self.metrics = dict(SSL_loss=loss.detach(), SSL_weight_mean=weight.mean(),
                            SSL_confidence=confidence.mean(), SSL_mean=self.mean.detach(),
                            SSL_variance=self.variance.detach(), SSL_ramp=ramp,
                            SSL_proto_coverage=prototype_valid.float().mean(),
                            SSL_physical_seen=self.physical_seen, SSL_seen=self.seen,
                            SSL_effective=self.effective, SSL_entropy_weight=0.)
        if view_settings:
            self.metrics.update(SSL_view_anchor_loss=0. if view_loss is None else view_loss.detach(),
                SSL_view_anchor_calls=self.view_anchor_calls,
                SSL_view_anchor_physical_seen=self.view_anchor_physical_seen)
        return loss

    def state_dict(self):
        value = dict(teacher=self.teacher.state_dict(), mean=self.mean, variance=self.variance,
                    centers=self.bank.centers, counts=self.bank.counts, ema_updates=self.ema_updates,
                    seen=self.seen, physical_seen=self.physical_seen, calls=self.calls,
                    effective=self.effective, per_class=self.per_class, method=self.c['ssl_method'])
        if self.c.get('view_anchor'):
            value.update(view_anchor=self.c['view_anchor'], view_anchor_calls=self.view_anchor_calls,
                         view_anchor_physical_seen=self.view_anchor_physical_seen)
        return value

    def load_state_dict(self, state):
        if state['method'] != self.c['ssl_method']:
            raise ValueError('Own SSL state belongs to another method')
        if state.get('view_anchor') and state['view_anchor'] != self.c.get('view_anchor'):
            raise ValueError('Physical feature state belongs to another recipe')
        self.teacher.load_state_dict(state['teacher'], strict=True)
        for name in ('mean', 'variance', 'per_class'):
            getattr(self, name).copy_(state[name])
        self.bank.centers.copy_(state['centers'])
        self.bank.counts.copy_(state['counts'])
        for name in ('ema_updates', 'seen', 'physical_seen', 'calls', 'effective'):
            setattr(self, name, state[name])
        self.view_anchor_calls = state.get('view_anchor_calls', 0)
        self.view_anchor_physical_seen = state.get('view_anchor_physical_seen', 0)
