import torch

from experiments.pamf_physics_tri.branch_guard import GROUPS, parameter_group
from experiments.pamf_ssl_research.u_gradprotect_objective import projected_gradients


def test_deployed_paths_get_distinct_groups_without_capturing_domain_paths():
    prefix = 'id_backbone.encoder.'
    cases = {'core.id_backbone.t_proj.weight': 'time', 'core.id_backbone.f1.weight': 'frequency',
             'core.id_backbone.f2.weight': 'frequency', 'core.id_backbone.f3.weight': 'frequency',
             'core.id_backbone.freq_gate.weight': 'frequency', 'core.id_backbone.f_proj.weight': 'frequency',
             'core.id_backbone.freq_stats_proj.0.weight': 'frequency', 'core.id_backbone.pa_proj.0.weight': 'pa',
             'core.id_backbone.pa_b1.weight': 'pa', 'core.id_backbone.pa_b2.weight': 'pa',
             'core.id_backbone.pa_b3.weight': 'pa', 'core.id_backbone.pa_gate.weight': 'pa',
             'core.id_backbone.pa_stats_proj.0.weight': 'pa', 'sequence_pool.gain_raw': 'signature',
             'response_projection.0.weight': 'reference', 'response_gain': 'reference',
             'core.id_backbone.cls_head.weight': 'head', 'core.id_backbone.t2.pw.weight': 't2'}
    assert {name: parameter_group(prefix + name) for name in cases} == cases
    assert parameter_group('dom_backbone.f_proj.weight') == 'other'
    assert parameter_group(prefix + 'core.id_backbone.t1.weight') == 'other'


def test_each_branch_has_its_own_nonconflicting_capped_u_gradient():
    named = [('id_backbone.encoder.core.id_backbone.t_proj.weight', torch.nn.Parameter(torch.ones(2))),
             ('id_backbone.encoder.core.id_backbone.f_proj.weight', torch.nn.Parameter(torch.ones(2)))]
    labeled = [torch.tensor([1., 0.]), torch.tensor([1., 0.])]
    unlabeled = [torch.tensor([-2., 3.]), torch.tensor([2., 0.])]
    combined, details = projected_gradients(named, labeled, unlabeled, group_fn=parameter_group, groups=GROUPS)
    torch.testing.assert_close(combined[0], torch.tensor([1., 1.]))
    torch.testing.assert_close(combined[1], torch.tensor([2., 0.]))
    assert details['time']['conflict'] and not details['frequency']['conflict']
    assert details['time']['capped'] and details['frequency']['capped']
