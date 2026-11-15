"""Preserve REPL, exposing accepted-U histograms and one-factor changes."""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import GuardedReceiverExtrapolation


class ObservedREPL(GuardedReceiverExtrapolation):
    def __init__(self, c):
        super().__init__(c)
        self.pseudo_hist = Counter()
        self.conf_bins = Counter()
        self.parts = {}

    def accept_pseudo(self, student, x, confidence, pseudo):
        accepted = super().accept_pseudo(student, x, confidence, pseudo)
        self.pseudo_hist.update(pseudo[accepted].detach().cpu().tolist())
        self.conf_bins.update({'below90': int((confidence < .90).sum()),
                               '90to95': int(((confidence >= .90) & (confidence < .95)).sum()),
                               '95to99': int(((confidence >= .95) & (confidence < .99)).sum()),
                               '99plus': int((confidence >= .99).sum())})
        return accepted

    def loss(self, student, device):
        self.parts = {}
        value = super().loss(student, device)
        for i in range(self.c['num_classes']):
            self.metrics[f'HPL_pseudo_class_{i}'] = self.pseudo_hist[i]
        for k, v in self.conf_bins.items():
            self.metrics['HPL_conf_bin_' + k] = v
        return value

    def guide(self, labeled, student):
        import math
        import torch
        from experiments.pamf_ssl_research.u_gradprotect_objective import parameter_group
        for k in list(self.metrics):
            if k.startswith('HPL_component_'):
                del self.metrics[k]
        sample = self.calls % 100 == 0
        self.metrics['HPL_component_sample'] = int(sample)
        if sample:
            named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
            params = [p for _, p in named]
            lg = torch.autograd.grad(labeled, params, retain_graph=True, allow_unused=True)
            for part, loss in self.parts.items():
                if not loss.requires_grad:
                    continue
                ug = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                for group in ('t2', 'cls_head', 'other'):
                    dot = ll = uu = 0.
                    for (name, _), a, b in zip(named, lg, ug):
                        if parameter_group(name) != group:
                            continue
                        if a is not None:
                            a = a.detach().double(); ll += float(a.square().sum())
                        if b is not None:
                            b = b.detach().double(); uu += float(b.square().sum())
                        if a is not None and b is not None:
                            dot += float((a*b).sum())
                    prefix = f'HPL_component_{part}_{group}_'
                    self.metrics[prefix+'L_norm'] = math.sqrt(ll)
                    self.metrics[prefix+'U_norm'] = math.sqrt(uu)
                    self.metrics[prefix+'dot'] = dot
                    self.metrics[prefix+'cosine'] = dot / math.sqrt(ll*uu) if ll > 0 and uu > 0 else 0.
        return super().guide(labeled, student)

    def state_dict(self):
        state = super().state_dict()
        state['U_pseudo_observation'] = dict(accepted_by_pseudo_class=dict(self.pseudo_hist),
            confidence_bins=dict(self.conf_bins), U_truth_read=False)
        return state


def source_validation_proxy(c, model):
    """Evaluate full original V, only after a training segment; no U truth."""
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.io import read, write
    dataset = SourceDataset(c['source_cache'], 'V', read(c['source_contract']))
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    n = ok = accepted = accepted_ok = 0
    groups = {}
    was = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for x, y, _, meta in loader:
                logits = model(x.to(next(model.parameters()).device))
                conf, pred = logits.softmax(1).max(1)
                pred, conf = pred.cpu(), conf.cpu()
                selected = conf >= .95
                right = pred == y
                n += len(y); ok += int(right.sum()); accepted += int(selected.sum())
                accepted_ok += int((selected & right).sum())
                for cls in y.unique().tolist():
                    mask = y == cls
                    g = groups.setdefault(str(cls), dict(count=0, correct=0, accepted=0, accepted_correct=0))
                    g['count'] += int(mask.sum()); g['correct'] += int((mask & right).sum())
                    g['accepted'] += int((mask & selected).sum())
                    g['accepted_correct'] += int((mask & selected & right).sum())
    finally:
        model.train(was)
    write(Path(c['output_root']) / 'source_v_pseudo_proxy.json', dict(
        role='V', threshold=.95, count=n, correct=ok, accepted=accepted, accepted_correct=accepted_ok,
        per_true_class=groups, U_truth_read=False, used_for_training=False,
        interpretation='Source validation high-confidence precision proxy, not U pseudo-label accuracy'))


@contextmanager
def installed(c, holder):
    from experiments.pamf_paper_completion.model import installed as paper_scope
    from experiments.pamf_ssl_research import hard_pl_objective, hard_pl_joint_objective, hard_pl_receiver_extrapolation_objective
    from .plan import validate
    old = hard_pl_objective.mechanism_loss
    base_loss = hard_pl_objective.HardPLObjective.loss
    joint_loss = hard_pl_joint_objective.JointNoAsMixStyle.loss
    receiver_loss = hard_pl_receiver_extrapolation_objective.ReceiverExtrapolation.loss
    factor = c['tuning_recipe'].get('shallow_factor', 1.)
    def mechanism(o0, o1, m0, m1, r0, r1, y, accepted, physical, aux):
        holder['objective'].parts['shallow'] = aux * factor * (.1 / 1.25) * c['hard_pl']['weight']
        return old(o0, o1, m0, m1, r0, r1, y, accepted, physical, aux * factor)
    def capture_base(self, student, device):
        value = base_loss(self, student, device)
        self._observed_base = value
        return value
    def capture_joint(self, student, device):
        value = joint_loss(self, student, device)
        self.parts['joint'] = value - self._observed_base
        self._observed_joint = value
        return value
    def capture_receiver(self, student, device):
        value = receiver_loss(self, student, device)
        self.metrics['HPL_RXE_reallocated_CE_coefficient'] = c['hard_pl_receiver_extrapolation']['extrapolated_CE_coefficient']
        self.parts['extrap_delta'] = value - self._observed_joint
        return value
    hard_pl_objective.mechanism_loss = mechanism
    hard_pl_objective.HardPLObjective.loss = capture_base
    hard_pl_joint_objective.JointNoAsMixStyle.loss = capture_joint
    hard_pl_receiver_extrapolation_objective.ReceiverExtrapolation.loss = capture_receiver
    try:
        with paper_scope(c, holder, ssl_enabled=True, objective_factory=ObservedREPL, validate=validate):
            yield
            # The scope is also used by the no-query smoke, which must not create artifacts.
            if not c.get('hard_pl_smoke'):
                source_validation_proxy(c, holder['model'])
    finally:
        hard_pl_objective.mechanism_loss = old
        hard_pl_objective.HardPLObjective.loss = base_loss
        hard_pl_joint_objective.JointNoAsMixStyle.loss = joint_loss
        hard_pl_receiver_extrapolation_objective.ReceiverExtrapolation.loss = receiver_loss
