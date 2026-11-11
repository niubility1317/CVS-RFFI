"""Separate deployed PAMF branch gradients within the existing U/L guard."""
from contextlib import contextmanager

from experiments.pamf_ssl_research.u_gradprotect_objective import protected_loss
from .practical_gate import PracticalGate, installed as practical_scope


GROUPS = ('t2', 'head', 'time', 'frequency', 'pa', 'signature', 'reference', 'other')


def parameter_group(name):
    if '.cls_head.' in name:
        return 'head'
    if '.t2.' in name:
        return 't2'
    if not name.startswith('id_backbone.encoder.'):
        return 'other'
    if '.sequence_pool.' in name:
        return 'signature'
    if '.response_projection.' in name or name.endswith('.response_gain'):
        return 'reference'
    core_prefix = 'id_backbone.encoder.core.id_backbone.'
    core_part = name.removeprefix(core_prefix).split('.', 1)[0] if name.startswith(core_prefix) else ''
    if core_part in ('f1', 'f2', 'f3', 'f_proj', 'freq_gate', 'freq_stats_proj'):
        return 'frequency'
    if core_part in ('pa_b1', 'pa_b2', 'pa_b3', 'pa_gate', 'pa_proj', 'pa_stats_proj'):
        return 'pa'
    if core_part == 't_proj':
        return 'time'
    return 'other'


class BranchGuard(PracticalGate):
    def __init__(self, c):
        super().__init__(c)
        self.branch_conflicts = {name: 0 for name in GROUPS}
        self.branch_caps = {name: 0 for name in GROUPS}
        self.branch_last = {}

    def guide(self, labeled, student):
        if self.pending_gradient_loss is None:
            raise ValueError('Current hidden-label U loss required')
        unlabeled, self.pending_gradient_loss = self.pending_gradient_loss, None
        named = [(name, p) for name, p in student.named_parameters() if p.requires_grad]
        ratio = self.c['hard_pl_gradient_guard']['U_to_L_norm_cap']
        value, details, gradients = protected_loss(labeled, unlabeled, named, max_u_ratio=ratio,
                                                    group_fn=parameter_group, groups=GROUPS)
        self.guard_calls += 1
        self.branch_last = details
        for group, items in details.items():
            self.branch_conflicts[group] += int(items['conflict'])
            self.branch_caps[group] += int(items['capped'])
            for key, number in items.items():
                self.metrics[f'HPL_BGP_{group}_{key}'] = float(number)
        for original, names in {'t2': ('t2',), 'cls_head': ('head',),
                                'other': ('time', 'frequency', 'pa', 'signature', 'reference', 'other')}.items():
            self.guard_conflicts[original] += int(any(details[name]['conflict'] for name in names))
            self.guard_caps[original] += int(any(details[name]['capped'] for name in names))
        self.metrics['HPL_GP_calls'] = self.guard_calls
        if self.c.get('hard_pl_smoke'):
            self.smoke_guard_expected = {name: grad for (name, _), grad in zip(named, gradients)}
            self.smoke_guard_details = details
        return value

    def state_dict(self):
        state = super().state_dict()
        state['U_branch_gradient_guard'] = dict(groups=list(GROUPS), calls=self.guard_calls,
            conflicts=dict(self.branch_conflicts), caps=dict(self.branch_caps), last=dict(self.branch_last))
        return state


@contextmanager
def paper_base_scope(c, holder, enabled, objective_factory):
    from experiments.pamf_paper_completion.model import installed as paper_scope
    from .branchguard_plan import validate

    with paper_scope(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    scope = paper_base_scope if c['protocol_id'] != 'F0' else None
    with practical_scope(c, holder, base_scope=scope, objective_factory=BranchGuard):
        yield
