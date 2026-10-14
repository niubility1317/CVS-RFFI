import copy
import importlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from . import design as d
from .model import build
from experiments.cvs_receiver_dro50.checks import checks as first_checks
from experiments.cvs_receiver_dro50.observer import Observer


def checks():
    core = first_checks()
    negatives = 0
    contexts = []
    mechanism_keys = ('feature_combination', 'receiver_risk', 'hypothesis_basis', 'source_falsification', 'goal')
    equal = lambda a, b: a == b or (isinstance(a, float) and isinstance(b, float)
                                    and math.isnan(a) and math.isnan(b))
    assert equal(float('nan'), float('nan')) and not equal(float('nan'), 0.)
    for seed in d.SEEDS:
        initial = None
        for row in d.training_rows():
            if row['model_seed'] != seed:
                continue
            c = d.config(row); d.validate(c)
            first = d.first.config(next(r for r in d.first.rows() if r['arm'] == row['arm']))
            if any(c[k] != first[k] for k in mechanism_keys):
                raise AssertionError('Remaining seed changed original scientific settings')
            a, original = d.make_args(c, 'cpu'), d.first.make_args(first, 'cpu')
            changed = {k for k in vars(a).keys() | vars(original).keys()
                       if not equal(vars(a).get(k), vars(original).get(k))}
            if changed - {'seed', 'run_id', 'candidate_id', 'output_dir'}:
                raise AssertionError(f'Unexpected native-context difference: {sorted(changed)}')
            m = build(c, 'cpu').eval()
            values = {k: v.clone() for k, v in m.state_dict().items()}
            if initial is None:
                initial = values
            elif set(initial) != set(values) or any(not torch.equal(v, values[k]) for k, v in initial.items()):
                raise AssertionError('Same-seed remaining risk arms have different initial states')
            with tempfile.TemporaryDirectory(prefix='cvs-dro-context-') as temporary:
                observer = Observer(design=d)
                observer.initialize(m, c, Path(temporary)); observer.steps = 15
                logits = m.forward_identity_only(torch.randn(11, 2, 256))['tx_logits']
                labels = torch.arange(11) % 6
                receivers = torch.tensor([1] * 3 + [3] * 2 + [4] * 2 + [6] * 2 + [8] * 2)
                original_loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=.01)
                native = SimpleNamespace(_labeled_identity_loss_hook=None, _trusted_pseudo_loss_hook=None)
                with observer.native_scope(native):
                    actual = native._labeled_identity_loss_hook(model=m, args=a, epoch=1, batch_idx=16,
                        logits=logits, labels=labels, receivers=receivers, clean_count=11,
                        original_loss=original_loss, role='L_s')
                    U = logits.sum() * 0
                    assert native._trusted_pseudo_loss_hook(model=m, args=a, epoch=1, batch_idx=16,
                        loss_l=actual, loss_u=U, loss_ent=U, active=False, logits=None, pseudo=None, mask=None) is U
                    assert observer.steps == 16 and len(observer.grad) == 1
                    assert all(torch.isfinite(g).all() and g.norm() > 1e-12
                               for g in observer.grad[0]['vectors']['selected'])
                    if c['receiver_risk']['mode'] == 'ERM':
                        assert actual is original_loss
                assert native._labeled_identity_loss_hook is None and native._trusted_pseudo_loss_hook is None
            contexts.append(dict(row_id=row['row_id'], native_difference_keys=sorted(changed),
                                 scientific_dictionary_identical=True, actual_same_seed_initial_state=True,
                                 actual_observer_context_joint_gradients_and_U_path=True))
            bad = copy.deepcopy(c); bad['receiver_risk']['step_size'] = .02
            try:
                d.validate(bad)
            except ValueError:
                negatives += 1
            else:
                raise AssertionError('Modified remaining risk formula accepted')
    qualification = dict(status='PASS', source_only=True, target_metrics_read_for_decision=False,
        fixed_rules=d.first.config(d.first.training_rows()[0])['source_falsification'],
        predefined_remaining_seeds=list(d.SEEDS), expand_all_three_arms_unchanged=True)
    d.validate_qualification(qualification)
    for key, value in [('status', 'FAIL'), ('source_only', False),
                       ('target_metrics_read_for_decision', True), ('fixed_rules', {}),
                       ('predefined_remaining_seeds', [2026092702]), ('expand_all_three_arms_unchanged', False)]:
        bad = copy.deepcopy(qualification); bad[key] = value
        try:
            d.validate_qualification(bad)
        except ValueError:
            negatives += 1
        else:
            raise AssertionError('Invalid first-source expansion authority accepted')
    for name in ('source', 'predict', 'score', 'feature_check', 'audit', 'dispatch', 'preflight', 'publish'):
        importlib.import_module(f'experiments.cvs_receiver_dro_seeds.{name}')
    from .publish import configure, transport
    configure()
    compile(transport.REMOTE, '<dro-seeds-remote>', 'exec')
    compile(transport.INSPECT, '<dro-seeds-inspect>', 'exec')
    if 'experiments.cvs_receiver_dro_seeds.preflight' not in transport.REMOTE:
        raise AssertionError('Wrong remaining preflight entry')
    if 'experiments/cvs_backbone_fishr_seeds' not in transport.PATHS:
        raise AssertionError('Remaining configuration dependency missing')
    return dict(status='PASS', unchanged_core_checks=core, negatives=negatives,
        native_contexts=contexts, real_IQ_checkpoint_or_first_qualification_read=False,
        first_seed_retrained=False, new_scientific_hypothesis=False,
        new_models=6, historical_baselines_reused=2, head_fit_updates=0)


if __name__ == '__main__':
    print(json.dumps(checks()))
