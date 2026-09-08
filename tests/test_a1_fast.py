from copy import deepcopy
import math

import pytest
import torch

from cvsrffi.orbit_teacher import EMALossScaleNormalizer
from cvsrffi.daot_training import compute_daot_batch_objective
from model_dual_cvsincnet import build_dual_model
from SSDG.train_ssdg import _forward_daot_teacher_views, build_arg_parser
from SSDG.train_ssdg import _validate_a1_scratch_only, _validate_daot_config


@pytest.mark.parametrize('values', [[0., 1e-7, -2., 3.], [float('nan'), float('inf'), 0., 1.]])
def test_scale_batching_keeps_python_state_loss_gradient_and_order(values):
    old = EMALossScaleNormalizer()
    fast = EMALossScaleNormalizer(batched_readback=True)
    for step in range(12):
        active = {'l': step % 3 != 0, 'u': True, 'off': False}
        a = {k: torch.tensor(v, requires_grad=True) for k, v in zip(active, [values[step % 4], 2.+step, -5.])}
        b = {k: v.detach().clone().requires_grad_() for k, v in a.items()}
        x, sx = old.normalize(a, active=active)
        y, sy = fast.normalize(b, active=active)
        assert sx == sy
        assert old.state_dict() == fast.state_dict()
        for key in a:
            torch.testing.assert_close(x[key], y[key], rtol=0, atol=0, equal_nan=True)
            x[key].backward(); y[key].backward()
            torch.testing.assert_close(a[key].grad, b[key].grad, rtol=0, atol=0, equal_nan=True)
    assert 'off' not in old.state_dict()['scales']


def test_inactive_scale_and_resume_schema_are_preserved():
    old = EMALossScaleNormalizer()
    old.normalize({'l': torch.tensor(4.)})
    fast = EMALossScaleNormalizer(batched_readback=True)
    fast.load_state_dict(old.state_dict())
    for n in [old, fast]:
        result, scales = n.normalize({'l': torch.tensor(20., requires_grad=True)}, active={'l': False})
        assert result['l'].item() == 5.
        assert scales['l'] == 4.
    assert old.state_dict() == fast.state_dict()


def _objective(clean, channel, views, norm):
    b = clean['z_id'].shape[0]
    return compute_daot_batch_objective(
        student_clean=clean, student_channel=channel, teacher_views=views,
        reliability=torch.ones(b, 2), importance=torch.ones(b, 2), recoverability=torch.ones(b),
        orbit_scale=1., tangent_scale=0., weights={'orbit_z': .5, 'orbit_logit': .2, 'orbit_proto': .1},
        coverage_floor=.15, huber_beta_min=.3, temperature=2., loss_normalizer=norm)


def test_real_identity_teacher_outputs_state_rng_and_downstream_update():
    torch.set_num_threads(2)
    torch.manual_seed(392005)
    model = build_dual_model(3, 2, model_size='S', input_len=256, use_daot_nuisance_head=True).eval()
    state = deepcopy(model.state_dict())
    inputs = [torch.randn(3, 2, 256), torch.randn(3, 2, 256)]
    domains = torch.tensor([0, 1, 0])
    rng = torch.get_rng_state().clone()
    all_views = {}
    with torch.no_grad():
        for mode in ['legacy', 'identity_sequential', 'identity_batched']:
            all_views[mode], _ = _forward_daot_teacher_views(model, inputs, domain_labels=domains, efficiency_mode=mode)
            assert torch.equal(torch.get_rng_state(), rng)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        for mode in ['identity_sequential', 'identity_batched']:
            for a, b in zip(all_views['legacy'], all_views[mode]):
                for key in ['z_id', 'tx_logits']:
                    torch.testing.assert_close(a[key], b[key], rtol=2e-5, atol=2e-6)
    template = torch.nn.Linear(4, 3)
    outputs = []
    for mode in ['legacy', 'identity_sequential', 'identity_batched']:
        head = deepcopy(template)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        z = head(torch.ones(3, 4))
        feature_dim = all_views[mode][0]['z_id'].shape[-1]
        # Independent differentiable projection produces the actual feature width.
        projection = torch.linspace(-1, 1, 3*feature_dim).reshape(3, feature_dim)
        student = {'z_id': z @ projection, 'tx_logits': z}
        out = _objective(student, student, all_views[mode], EMALossScaleNormalizer(batched_readback=mode!='legacy'))
        out['loss'].backward()
        grads = [p.grad.clone() for p in head.parameters()]
        opt.step()
        outputs.append((out, grads, deepcopy(head.state_dict())))
    for out, grads, state in outputs[1:]:
        torch.testing.assert_close(out['loss'], outputs[0][0]['loss'], rtol=2e-5, atol=2e-6)
        assert torch.equal(out['diagnostics']['consensus_mask'], outputs[0][0]['diagnostics']['consensus_mask'])
        for x,y in zip(grads, outputs[0][1]): torch.testing.assert_close(x,y,rtol=3e-5,atol=3e-6)
        for k,v in state.items(): torch.testing.assert_close(v,outputs[0][2][k],rtol=3e-5,atol=3e-6)


def test_identity_path_rejects_train_mode_and_batch_statistics():
    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.bn = torch.nn.BatchNorm1d(2, track_running_stats=False)
        def forward_identity_only(self,x,**kw): return {'z_id': self.bn(x), 'tx_logits': self.bn(x)}
    model = Teacher()
    views = [torch.ones(3,2),torch.ones(3,2)]
    with pytest.raises(ValueError, match='eval'):
        _forward_daot_teacher_views(model,views,domain_labels=None,efficiency_mode='identity_sequential')
    model.eval()
    with pytest.raises(ValueError, match='BatchNorm'):
        _forward_daot_teacher_views(model,views,domain_labels=None,efficiency_mode='identity_batched')


def test_execution_flags_default_to_legacy():
    args = build_arg_parser().parse_args(['--output_dir', 'unused'])
    assert args.daot_efficiency_mode == 'legacy'
    assert not args.daot_batched_scale_readback
    assert not args.daot_skip_mean_metadata


def test_batched_logs_preserve_float_order_missing_and_nonfinite():
    from post_stage_common import mean_logs
    logs=[{'a':torch.tensor(1e-8,dtype=torch.float64),'b':torch.tensor(3.,dtype=torch.float16),'missing':'no',
           'bad':torch.tensor(float('nan')),'vector':torch.ones(3)},
          {'a':torch.tensor(1e8,dtype=torch.float64),'b':torch.tensor(7.),'bad':float('inf')},
          {'a':torch.tensor(-1e8,dtype=torch.float64),'b':1.}]
    assert mean_logs(logs)==mean_logs(logs,batched_readback=True)


def test_core90_technical_completion_is_not_all_mechanism_promotion():
    from SSDG.train_ssdg import _resolve_phase1_terminal_status
    import inspect
    sig=inspect.signature(_resolve_phase1_terminal_status)
    # Exercise the existing terminal resolver with the exact fresh dependency facts.
    result=_resolve_phase1_terminal_status(tail_stopped=False,export_failed=False,final_blocked=False,
        selected_checkpoint_exists=True,heldout_eval_status='COMPLETE',external_final_eval=False,
        p0_mechanisms_ready=False,p1_mechanisms_ready=False,endpoint_export_ready=False,
        mechanism_gates_required=False,endpoint_export_required=False)
    assert result=='COMPLETE'


def test_matched_core90_pipeline_has_no_historical_checkpoint_and_equal_data():
    import importlib.util
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('a1_runner',root/'code/scripts/run_a1_fast_matched_core90.py')
    runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    matrix=json.loads((root/'configs/a1_fast_matched_core90_20260908.json').read_text(encoding='utf-8'))
    project=Path('/new_project'); run=project/'runs/fresh_run'
    parser=build_arg_parser()
    core=parser.parse_args(runner.build_train_command(matrix,project_root=project,run_root=run,row_id='CORE90_MATCHED_FRESH',core90=True)[3:])
    _validate_daot_config(core); _validate_a1_scratch_only(core)
    assert core.from_scratch and not core.baseline_ckpt and not core.teacher_ckpt
    assert not core.use_adv3b02_daot_stn and not core.fasttrust_rc4
    for row in matrix['rows']:
        args=parser.parse_args(runner.build_train_command(matrix,project_root=project,run_root=run,row_id=row['id'],row=row)[3:])
        _validate_daot_config(args)
        assert args.baseline_ckpt==str(run/'CORE90_MATCHED_FRESH/final_ssdg.pth')==args.teacher_ckpt
        assert not args.from_scratch and args.fasttrust_rc4 and args.rc4_use_anchor
        for key in ['seed','epochs','labeled_ratio','unlabeled_ratio','source_val_ratio','wisig_train_days','wisig_test_days','wisig_train_rxs','wisig_test_rxs','model_size','model_variant','num_classes','batch_size','checkpoint_selection']:
            assert getattr(args,key)==getattr(core,key),key
    core.baseline_ckpt='old.pth'
    with pytest.raises(ValueError,match='forbids'):
        _validate_a1_scratch_only(core)
