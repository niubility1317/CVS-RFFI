import copy
import importlib
import json
import torch
from . import design as d
from experiments.cvs_source_variance_angle import audit as shared
from experiments.cvs_source_variance_angle.checks import checks as formula_checks


def checks():
    formula = formula_checks()
    first = d.first.config(d.first.rows()[0])
    identity_keys = {'schema', 'run_id', 'row_id', 'model_seed', 'source_config',
                     'checkpoint_parent_run', 'checkpoint_parent_execution', 'source_qualification_run',
                     'source_decision_commit', 'first_seed_remeasured', 'combined_model_seeds',
                     'output_root', 'launch_owner'}
    changes = []
    negatives = 0
    for row in d.rows():
        c = d.config(row); d.validate(c)
        differing = {k for k in first.keys() | c.keys() if first.get(k) != c.get(k)}
        assert differing <= identity_keys
        assert c['source_config']['model_seed'] == row['model_seed']
        assert c['source_config']['arm'] == 'SIG_TIME'
        assert c['checkpoint_epoch'] == 50 and c['qualification'] == d.first.QUALIFICATION
        changes.append(dict(model_seed=row['model_seed'], scientific_configuration_unchanged=True))
        for key, value in [('role', 'V'), ('source_packets', 6300), ('head_fit_updates', 1),
                           ('query_read', True), ('variance_norm_epsilon', 1e-8),
                           ('model_seed', 2026092701), ('checkpoint_parent_execution', 'unverified')]:
            bad = copy.deepcopy(c); bad[key] = value
            try:
                d.validate(bad)
            except ValueError:
                negatives += 1
            else:
                raise AssertionError('Unregistered remaining source configuration accepted')
    raw = [dict(vectors={key: [torch.tensor([1., 2.], dtype=torch.float64),
                               torch.tensor([2., 3.], dtype=torch.float64)]
                         for key in ('CE', 'RAW', 'ANGLE')}) for _ in range(5)]
    summary = shared.summarize(raw, design=d)
    assert summary == shared.summarize(raw)
    assert shared.qualification(summary, design=d) == shared.qualification(summary)
    assert shared.qualification(summary, design=d)['status'] == 'PASS'
    for key, value in [('cosine', -1e-8), ('auxiliary_norm', 0.), ('CE_norm', 0.)]:
        bad = copy.deepcopy(summary); bad[0]['ANGLE']['global_gradient'][key] = value
        assert shared.qualification(bad, design=d)['status'] == 'FAIL'
    bad = copy.deepcopy(summary); bad[1]['ANGLE']['nonzero_cohorts'] = 4
    assert shared.qualification(bad, design=d)['status'] == 'FAIL'
    for name in ('audit', 'preflight', 'publish'):
        importlib.import_module(f'experiments.cvs_source_variance_angle_seeds.{name}')
    from .publish import configure, transport
    configure()
    assert transport.d is d
    for suffix in ('preflight', 'audit'):
        assert f'experiments.cvs_source_variance_angle_seeds.{suffix}' in transport.REMOTE
    assert 'experiments/cvs_backbone_fishr_seeds' in transport.PATHS
    compile(transport.REMOTE, '<source-seeds-remote>', 'exec')
    compile(transport.INSPECT, '<source-seeds-inspect>', 'exec')
    return dict(status='PASS', original_formula_checks=formula, remaining_config_negatives=negatives,
                qualification_negatives=4, shared_statistics_equivalent=True,
                formula_and_threshold_changes=0, source_models=changes,
                real_IQ_or_checkpoint_read=False, query_read=False, optimizer_steps=0)


if __name__ == '__main__':
    print(json.dumps(checks()))
