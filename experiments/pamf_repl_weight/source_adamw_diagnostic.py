"""Source-only, disposable AdamW counterfactuals for the original Tri E20.

This records fresh-moment first-step behavior, not the later AdamW trajectory.
No model, optimizer state, U labels, validation examples, or target examples
are persisted or used for training.
"""
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import GuardedReceiverExtrapolation
from experiments.pamf_ssl_research.source_branch_gradients import _relation, source_loaders
from experiments.pamf_ssl_research.source_optimizer_counterfactual import _first_step_options
from experiments.pamf_ssl_research.source_optimizer_shadow import shadow_step
from experiments.pamf_ssl_research.u_gradprotect_objective import projected_gradients


GROUPS = ('t2', 'cls_head', 'other')
SETTINGS = ('ten_class', 'source_rx_6')
SEED = 2026092701
BATCH_OFFSETS = (0, 1, 2)


@contextmanager
def capture_guard_derivatives():
    original = torch.autograd.grad
    captured = []

    def observe(*args, **kwargs):
        values = original(*args, **kwargs)
        captured.append(tuple(None if v is None else v.detach().clone() for v in values))
        return values

    torch.autograd.grad = observe
    try:
        yield captured
    finally:
        torch.autograd.grad = original


def difference(left, right, named):
    return tuple(None if a is None and b is None else
                 (torch.zeros_like(p) if a is None else a) -
                 (torch.zeros_like(p) if b is None else b)
                 for (_, p), a, b in zip(named, left, right))


class DiagnosticObjective(GuardedReceiverExtrapolation):
    def guide(self, labeled, student):
        named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
        with capture_guard_derivatives() as gradients:
            value = super().guide(labeled, student)
        if len(gradients) != 2:
            raise ValueError('Expected exactly the native full L and U derivatives')
        labeled_gradient, unlabeled_gradient = gradients
        guarded, details = projected_gradients(named, labeled_gradient, unlabeled_gradient,
                                              max_u_ratio=self.c['hard_pl_gradient_guard']['U_to_L_norm_cap'])
        options = _first_step_options(self.c)
        baseline, base_meta = shadow_step(student, named, labeled_gradient, **options)
        combined, combined_meta = shadow_step(student, named, guarded, **options)
        delta_gradient = difference(guarded, labeled_gradient, named)
        delta_step = difference(combined, baseline, named)
        self.diagnostic = dict(
            groups={g: dict(guard=details[g],
                            protected_gradient=_relation(named, labeled_gradient, delta_gradient, g),
                            incremental_descent=_relation(named, labeled_gradient, delta_step, g),
                            total_descent=_relation(named, labeled_gradient, combined, g))
                    for g in GROUPS},
            baseline_optimizer=base_meta, combined_optimizer=combined_meta,
            expected_guarded_gradient=guarded)
        return value


def collect(runtime):
    from experiments.cric_dura_comparison.bootstrap import activate, numerical_policy
    activate(runtime)
    numerical_policy()
    from experiments.cric_dura_comparison import f_train, t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    from experiments.pamf_branch_training.source import checked_e20_parent
    from experiments.pamf_paper_completion import plan as paper
    from experiments.pamf_paper_completion.model import installed as paper_scope

    rows = []
    for setting in SETTINGS:
        row_id = f'{setting}-factorial-ssl-full-permuted-s{SEED}'
        source = Path(runtime['project']) / 'runs' / paper.RUN / 'F' / row_id / 'source'
        complete = read(source / 'complete.json')
        c = complete['config']
        paper.validate_config(c)
        if (complete['status'] != 'SOURCE_FROZEN' or complete['target_access'] or complete['U_truth_read']
                or complete['epoch'] != 50 or c['row_id'] != row_id):
            raise ValueError('Original source-only paper model identity differs')
        prefix = source / 'segment20'
        evidence = checked_e20_parent(dict(c, shared_prefix=str(prefix)))
        cp = torch.load(prefix / 'final_ssdg.pth', map_location='cpu', weights_only=False)
        contract = read(c['source_contract'])
        for offset in BATCH_OFFSETS:
            cfg = dict(c, segment_epochs=10, epoch_offset=20, step_offset=4440,
                       parent_source=str(prefix))
            torch.manual_seed(SEED + offset)
            holder = {}
            with paper_scope(cfg, holder, ssl_enabled=True, objective_factory=DiagnosticObjective):
                loaders = source_loaders(cfg, contract)
                with f_train.class_dimension(cfg), native_scope(f_train.original_config(SEED)) as native:
                    args = f_train.arguments(cfg, 'cpu')
                    model = t1_model.install(native.build_baseline_model(args, 'cpu'), cfg)
                    if not args.use_aug or args.use_concat_sat_channel_aug:
                        raise ValueError('Expected the original ordinary L augmentation without concatenated satellite input')
                    aug_base = native.build_aug_base_cfg(args)
                    augmentor = native.make_augmentor(aug_base)
                    prior_args = SimpleNamespace(**cp['args'])
                    prior_args.epochs, prior_args.label_epochs, prior_args.pseudo_epochs = 20, 13, 7
                    # f_train.train pins the continuation's augmentation schedule
                    # to the original E20 parent's labeled epoch 13.
                    native.configure_augmentor_for_epoch(augmentor, aug_base, 13, prior_args)
                model.load_state_dict(cp['model'], strict=True)
                model.eval()
                with torch.no_grad():
                    logits = model(loaders['train_loader'].dataset.x[:2])
                if logits.shape != (2, c['num_classes']) or not torch.isfinite(logits).all():
                    raise ValueError('Real source checkpoint smoke failed')
                model.train()
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                model._cd_U_loader = loaders['unlabeled_loader']
                model._cd_U_iter = islice(iter(model._cd_U_loader), offset, None)
                x, y, domain, metadata = next(islice(iter(loaders['train_loader']), offset, None))
                if x.shape != (128, 2, 256) or y.shape != (128,) or not bool(((y >= 0) & (y < c['num_classes'])).all()):
                    raise ValueError('Expected one valid labeled source batch')
                ordinary = torch.nan_to_num(augmentor(x, labels=y, no_pa=not bool(args.aug_enable_pa_normal)),
                                            nan=0., posinf=0., neginf=0.)
                with model._phase1_labeled_feature_context():
                    output = model(ordinary, return_aux=True, domain_labels=domain)
                ce = F.cross_entropy(output['tx_logits'], y, label_smoothing=.01)
                correction, _ = model._source_labeled_feature_hook(
                    model=model, labeled_output=output, labels=y, extra=(None, metadata), domains=domain,
                    optimizer_step=1, identity_loss=ce, labeled_count=128, labeled_inputs=x)
                loss = ce + correction
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite source loss')
                loss.backward()
                named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                fact = holder['objective'].diagnostic
                expected = fact.pop('expected_guarded_gradient')
                for (name, p), gradient in zip(named, expected):
                    if (p.grad is None) != (gradient is None):
                        raise ValueError('Native guard backward connectivity differs: ' + name)
                    if gradient is not None:
                        torch.testing.assert_close(p.grad, gradient, rtol=1e-5, atol=1e-7)
                state = holder['objective'].state_dict()
                if (state['calls'] != 1 or state['seen'] != 256 or state['student_views'] != 1536
                        or state['U_truth_read'] or state['EMA_updates']
                        or any(not torch.equal(p.detach(), before[n]) for n, p in model.named_parameters())):
                    raise ValueError('Original REPL source execution or disposable model changed')
                rows.append(dict(setting=setting, row_id=row_id, batch_offset=offset, seed=SEED,
                                 classes=c['num_classes'], accepted=state['accepted'],
                                 ordinary_L_augmentation=True, augmentation_schedule_epoch=13,
                                 source_epoch=20, source_ancestor=evidence['folder'],
                                 ce=float(ce.detach()), groups=fact['groups'],
                                 baseline_optimizer=fact['baseline_optimizer'],
                                 combined_optimizer=fact['combined_optimizer']))
    return dict(status='SOURCE_ADAMW_DIAGNOSTIC_COMPLETE', rows=rows,
                source_models=2, independent_source_batch_snapshots=len(rows),
                U_truth_reads=0, V_reads=0, target_reads=0, original_parameter_updates=0,
                disposable_optimizer_steps=2*len(rows), new_training=False,
                interpretation='Three source batches per setting, from fixed E20 parameters, '
                'fresh optimizer moments only. Negative incremental descent dot indicates a '
                'worse local first-order L change than the matched L-only step; it is not '
                'a measured loss increase, later-trajectory result, or target outcome.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m experiments.pamf_repl_weight.source_adamw_diagnostic RUNTIME_JSON')
    print(json.dumps(collect(read(sys.argv[1])), allow_nan=False), flush=True)
