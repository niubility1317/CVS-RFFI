"""Replication-specific protocol negatives; no IQ or checkpoint reads."""
from copy import deepcopy
import json
from . import design as d
from .protocol import final_metadata, checkpoint


def run():
    negative = 0
    def bad(fn):
        nonlocal negative
        try:
            fn()
        except (ValueError, TypeError, KeyError, StopIteration):
            negative += 1
        else:
            raise AssertionError('Invalid replication scope accepted')
    assert d.SEEDS == (2026092702, 2026092703) and d.VIEWS == ('clean',)
    assert len({d.config(r)['parent_source'] for r in d.rows()}) == 2
    for row in d.rows():
        c = d.config(row)
        d.validate(c)
        assert c['parent_source'] == d.parent_config(c)['parent_source']
        assert c['ce_objective'] == dict(temperature=1., weight=.5, smoothing=.01)
        assert c['epochs'] == 10 and c['updates'] == 2220 and c['step_offset'] == 4440
        assert c['As'] and c['Ag_mode'] == 'NONE' and c['L_NN_per_step'] == 768
        assert c['seed_binding']['F_factory_model_seed'] == row['model_seed']
        for key, value in [('target_access', True), ('As', False), ('epochs', 20), ('updates', 4440),
                           ('model_seed', 2026092701), ('parent_source', 'unknown'),
                           ('label_epochs', 13), ('test_views', ['clean', 'practical_low_urban']),
                           ('checkpoint_sources', []), ('arm', 'CE_T2')]:
            z = deepcopy(c)
            z[key] = value
            bad(lambda z=z: d.validate(z))
        for key, value in [('temperature', 2.), ('weight', .68), ('smoothing', 0.)]:
            z = deepcopy(c)
            z['ce_objective'][key] = value
            bad(lambda z=z: d.validate(z))
        bad(lambda: d.config(row, 20))
    c = d.config(d.rows()[0])
    roles = dict(L=['L'], U=['U'], V=['V'])
    done = dict(config=c, epoch=10, optimizer_steps=2220, target_access=False, head_fit_updates=0)
    frozen = dict(config=c, selection='final_only', target_access=False)
    initial = dict(scratch_only=False, target_contact=False, checkpoint_sources=c['checkpoint_sources'], source_roles='EXACT_MATCH')
    final_metadata(c, done, frozen, initial, dict(role_ids=roles), roles)
    for key, value in [('target_access', True), ('epoch', 20), ('optimizer_steps', 4440), ('head_fit_updates', 1)]:
        z = deepcopy(done)
        z[key] = value
        bad(lambda z=z: final_metadata(c, z, frozen, initial, dict(role_ids=roles), roles))
    cp = dict(epoch=10, a1_ema_successful_updates=2220, run_id=d.RUN, candidate_id=c['row_id'],
              checkpoint_selection='final_only', ema_model={},
              args=dict(baseline_ckpt=c['parent_source'] + '/final_ssdg.pth', teacher_ckpt=''))
    checkpoint(c, cp)
    for key, value in [('run_id', d.REFERENCE_RUN), ('candidate_id', 'CE_HALF-s2026092703'), ('ema_model', None)]:
        z = deepcopy(cp)
        z[key] = value
        bad(lambda z=z: checkpoint(c, z))
    from . import objective
    from experiments.cvs_satellite_ce64 import objective as original
    import torch
    g = torch.Generator().manual_seed(6419)
    x = torch.randn(64, 6, generator=g, requires_grad=True)
    y = torch.randn(64, 6, generator=g, requires_grad=True)
    labels = torch.arange(64) % 6
    a, ra = objective.replacement(x, y, labels, 'CE_HALF', c['ce_objective'])
    b, rb = original.replacement(x, y, labels, 'CE_HALF', c['ce_objective'])
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert ra == rb
    ga = torch.autograd.grad(a, (x, y), retain_graph=True)
    gb = torch.autograd.grad(b, (x, y))
    for one, two in zip(ga, gb):
        torch.testing.assert_close(one, two, rtol=0, atol=0)
    return dict(status='PASS', protocol_negatives=negative, unchanged_CE_HALF_value_and_gradient=True,
                IQ=0, CP=0, CVS_NN=0, optimizer_updates=0)


if __name__ == '__main__':
    print(json.dumps(run()))
