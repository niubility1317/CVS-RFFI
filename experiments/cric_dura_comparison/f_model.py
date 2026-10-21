"""Original CRIC graph, generalized source classes and explicit mechanism ablations."""
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F

PREFIX = 'id_backbone.encoder.core.id_backbone.'


def contrast(maps, labels, receivers):
    if labels.ndim != 1 or receivers.shape != labels.shape or labels.dtype != torch.long:
        raise ValueError('Current labeled batch and physical RX required')
    same = labels[:, None] == labels[None, :]
    pos = same & (receivers[:, None] != receivers[None, :])
    neg = ~same
    valid = pos.any(1) & neg.any(1)
    results, details = [], []
    for name, h in maps.items():
        h = h if h.ndim == 2 else torch.cat((h.mean(-1), h.std(-1, unbiased=False)), 1)
        z = (h - h.detach().mean(0)) / h.detach().std(0, unbiased=False).clamp_min(1e-4)
        z = F.normalize(z, dim=1, eps=1e-4)
        if bool(valid.any()):
            scores = (z @ z.T / .2)[valid]
            positives = pos[valid]
            count = positives.sum(1)
            value = (torch.logsumexp(scores.masked_fill(~(pos | neg)[valid], -torch.inf), 1)
                - (scores * positives).sum(1) / count - count.to(z.dtype).log()).mean()
        else:
            value = z.sum() * 0.
        results.append(value)
        details.append(dict(level=name, valid_anchors=int(valid.sum()), total_anchors=len(labels)))
    return torch.stack(results).mean(), details


@contextmanager
def labeled_context(self):
    if not self.training:
        yield
        return
    self._feature_l_steps += 1
    if self._cd_pending is not None:
        raise ValueError('Unconsumed L feature graph')
    record = dict(first={}, second={}, x=None)
    self._cd_pending = record
    self._cd_phase = 'first'
    def capture_input(module, inputs):
        if record['x'] is not None or inputs[0].shape != (128, 2, 256):
            raise ValueError('Exactly one actual ordinary L128 input required')
        record['x'] = inputs[0].detach().clone()
    handles = [self.register_forward_pre_hook(capture_input)]
    for name in self._cd_levels:
        def capture(module, inputs, output, name=name):
            if self._cd_phase not in ('first', 'second'):
                raise ValueError('Feature hook escaped current L forwards')
            record[self._cd_phase][name] = output[:128]
        handles.append(self.get_submodule(PREFIX + name).register_forward_hook(capture))
    self._cd_handles = handles
    try:
        yield
    except BaseException:
        clear(self)
        raise
    finally:
        # Layer hooks continue through the actual second L forward only.
        handles[0].remove()
        self._cd_phase = None


def clear(self):
    for h in self._cd_handles:
        h.remove()
    self._cd_handles = []
    self._cd_phase = None
    self._cd_pending = None


def training_hook(self, *, model, labeled_output, labels, extra, domains, optimizer_step, identity_loss, labeled_count):
    if self is not model or not self.training or labeled_count != 128 or optimizer_step != self._feature_l_steps:
        raise ValueError('Ordered labeled source call required')
    pending = self._cd_pending
    if pending is None or pending['x'] is None:
        raise ValueError('Actual augmented source input not captured')
    c = self._cd_config
    ablation = c.get('ablation')
    step = optimizer_step
    epoch = (step - 1) // 222 + 1
    smoothing = 0. if ablation == 'F15' else .01
    first = labeled_output['tx_logits']
    try:
        sat_logits = None
        sat_info = None
        second_input = pending['x']
        second_domains = domains
        if c['satellite_concat']:
            sat, sat_info = self._cd_augmentor(pending['x'], (step - 1) / 11100., step)
            second_input = torch.cat((pending['x'], sat), 0)
            second_domains = torch.cat((domains, domains), 0)
        if ablation not in ('F05', 'F15'):
            self._cd_phase = 'second'
            second_out = self.forward_identity_only(second_input, domain_labels=second_domains)
            second = second_out['tx_logits'][:128]
            if c['satellite_concat']:
                sat_logits = second_out['tx_logits'][128:]
            p, q = first.log_softmax(1), second.log_softmax(1)
            kl = .5 * ((p.exp() - q.exp()) * (p - q)).sum(1).mean()
            ce = F.cross_entropy(second, labels, label_smoothing=smoothing)
            correction = .5 * (ce - identity_loss) + (0. if ablation == 'F04' else 1.) * kl
            self._cd_extra_L += len(labels)
        else:
            second_out = None
            kl = first.sum() * 0.
            correction = first.sum() * 0.
        self._cd_phase = None
        # Remove all layer hooks before satellite/U/EMA forwards.
        for handle in self._cd_handles:
            handle.remove()
        self._cd_handles = []
        weight = .1 * min(max((epoch - 5) / 10., 0.), 1.) if ablation != 'F15' else 0.
        detail = []
        if weight:
            rx = extra[1]['rx_i'].to(labels.device).long()
            if set(rx.tolist()) - set(c['source_rxs']):
                raise ValueError('L RX outside source contract')
            phases = ['first'] if second_out is None else ['first', 'second']
            values = []
            for phase in phases:
                maps = pending[phase]
                if ablation == 'F03':
                    value_out = labeled_output if phase == 'first' else second_out
                    maps = {'joint160': value_out['z_id']}
                if set(maps) != (set(self._cd_levels) if ablation != 'F03' else {'joint160'}):
                    raise ValueError('Actual requested L maps missing')
                value, stats = contrast(maps, labels, rx)
                values.append(value)
                detail.extend(stats)
            correction = correction + weight * torch.stack(values).mean()
        if c['satellite_concat']:
            if sat_logits is None or len(sat_logits) != 128:
                raise ValueError('Matched satellite slots must use actual clean+sat concatenation')
            correction = correction + .68 * F.cross_entropy(sat_logits, labels, label_smoothing=smoothing)
            self._cd_sat_seen += len(labels)
            self._cd_sat_applied += len(labels) * int(sat_info['applied'])
        if c['regime'] == 'matched_sat_plain_pl':
            try:
                batch = next(self._cd_U_iter)
            except StopIteration:
                self._cd_U_iter = iter(self._cd_U_loader)
                batch = next(self._cd_U_iter)
            u = batch[0].to(first.device)
            if not bool((batch[1] == -1).all()):
                raise ValueError('U truth exposed')
            was = self.training
            self.eval()
            with torch.no_grad():
                confidence, pseudo = self(u).softmax(1).max(1)
            self.train(was)
            accepted = confidence >= .95
            logits = self.forward_identity_only(u, domain_labels=batch[2].to(first.device))['tx_logits']
            correction = correction + (F.cross_entropy(logits, pseudo, reduction='none') * accepted).mean()
            self._cd_U_seen += len(u)
            self._cd_U_accepted += int(accepted.sum())
        metrics = dict(CRIC_RDrop_KL=kl.detach(), CRIC_supcon_weight=weight,
            CRIC_valid_anchors=sum(d['valid_anchors'] for d in detail),
            CRIC_total_anchors=sum(d['total_anchors'] for d in detail),
            CRIC_extra_L=self._cd_extra_L, CRIC_sat_seen=self._cd_sat_seen,
            CRIC_sat_applied=self._cd_sat_applied, CRIC_PL_seen=self._cd_U_seen,
            CRIC_PL_accepted=self._cd_U_accepted)
        self._cd_calls += 1
        return correction, metrics
    finally:
        clear(self)


training_hook._accepts_labeled_inputs = False


def install(model, c):
    from experiments.cvs_feature_sequence50.features import install as original_features
    from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool
    from experiments.cvs_rdrop_sinc_units.coordinate import CoordinateSinc
    from experiments.cvs_source_identity_readout.input_policy import apply
    model._identity_input_policy = apply(model)
    original_features(model, c['model_seed'], 'SIG_TIME')
    ref = model.id_backbone.encoder
    core = ref.core.id_backbone
    ref.sequence_pool = CompositePool(ref.sequence_pool.channels,
        'SIG_TIME' if c.get('ablation') == 'F06' else 'SIG_TIME_X3', c['model_seed'])
    core.sinc = CoordinateSinc(core.sinc, 'CENTER_25K')
    # The native wrapper is six-class; dimension the same original cosine head
    # at scratch construction, before any forward, optimizer, or checkpoint.
    if core.cls_head.weight.shape[0] != c['num_classes']:
        core.cls_head.weight = nn.Parameter(torch.empty(c['num_classes'], 160))
        nn.init.xavier_uniform_(core.cls_head.weight)
    ablation = c.get('ablation')
    if ablation == 'F07':
        for view in ref.sequence_pool.views:
            view.mode = 'SIG_NO_TIME'
            from .structural import no_time_descriptor
            view.forward = MethodType(no_time_descriptor, view)
    if ablation == 'F08':
        from .structural import NoSequence
        ref.sequence_pool = NoSequence()
    if ablation in ('F09', 'F10'):
        from .structural import remove_branch
        remove_branch(ref, ablation)
    model._cd_levels = ['time_down'] if ablation == 'F01' else ['t2.pool'] if ablation == 'F02' else [] if ablation in ('F03', 'F15') else ['time_down', 't2.pool']
    model._cd_config = deepcopy(c)
    model._cd_calls = model._cd_extra_L = model._cd_sat_seen = model._cd_sat_applied = model._cd_U_seen = model._cd_U_accepted = 0
    model._cd_pending = None
    model._cd_phase = None
    model._cd_handles = []
    model._phase1_labeled_feature_context = MethodType(labeled_context, model)
    model._source_labeled_feature_hook = MethodType(training_hook, model)
    if c['satellite_concat']:
        from experiments.phase1_external_dg_aug.augmentation import DefaultConcat
        model._cd_augmentor = DefaultConcat(2027)
    model._feature_learning_report.update(CRIC_contract_adapter=True, class_count=c['num_classes'],
        ablation=ablation, posthoc_head_fit=False, sinc_coordinates='CENTER_25K')
    return model
