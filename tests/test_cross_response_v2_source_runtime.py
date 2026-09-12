"""Actual runtime collection, source queries, observation consumption and resume."""
import copy
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader

from test_cross_response_v2_integration import Model, Head, mechanism_controls
from test_cross_response_runtime_contract import SourceDataset
from test_cross_response_v2_geometry_gate import make_gate, evidence
from test_cross_response_v2_scheduler import evidence_config
from cvsrffi.cross_response.config import validate_runtime_config
from cvsrffi.cross_response.integration import CrossResponseRuntime
from cvsrffi.cross_response.source_mechanism import joint_objective_contract
from cvsrffi.cross_response.replay_audit import first_divergence, capture_rng_state, restore_rng_state
from scripts.register_cross_response_v2 import configuration
from scripts.core90_cross_response_matrix import resolve_variant


class SixSourceDataset(SourceDataset):
    transform = None
    def __init__(self, validation=False):
        super().__init__(validation)
        self.index = [SimpleNamespace(tx_i=t, rx_i=r, day_i=0, eq_i=0,
            sig_i=s+(20 if validation else 0), event_id=None)
            for t in range(6) for r in range(5) for s in range(8)]


class SixModel(Model):
    def __init__(self):
        super().__init__()
        self.id_backbone.cls_head.head = Head(3, 6)
        self.dom_backbone.cls_head.head = Head(3, 5)
    def forward(self, x, **kwargs):
        return super().forward(x)


def make_runtime(tmp_path, variant='U2', *, reliable=False):
    torch.manual_seed(31)
    model = SixModel()
    artifact = tmp_path / 'source.bin'
    artifact.write_bytes(b'synthetic_source_only')
    args = SimpleNamespace(cross_response_variant=variant, wisig_train_rxs='0,1,2,3,4', wisig_test_rxs='5',
        wisig_train_days='0', wisig_test_days='1', output_dir=str(tmp_path), batch_size=32,
        eval_batch_size=32, seed=392005, wisig_pkl=str(artifact), lr=.01, weight_decay=.01,
        amp=False, max_grad_norm=.1, sat_fs_hz=25e6, sat_fc_hz=2.462e9)
    gate = make_gate(1).config
    gate['terminal_identity_module'] = 'time_fuse'
    # Test-only impossible thresholds prove the live path stays closed on failure.
    gate['thresholds'] = {key: 1e6 for key in gate['thresholds']}
    c = resolve_variant(configuration(), variant)
    c.update(mechanism_controls())
    c.update(mechanism_gate=gate, target_family='iq', target_dim=5,
             source_fit_max_records=32, source_eval_max_blocks=1, gate_min_blocks=1)
    ctx = dict(train_loader=DataLoader(SixSourceDataset(), batch_size=32),
        val_loader=DataLoader(SixSourceDataset(True), batch_size=32), input_len=32,
        domain_label_map={i:i for i in range(5)})
    validate_runtime_config(c)
    rt = CrossResponseRuntime(model, ctx, c, args, torch.device('cpu'))
    if reliable:
        cfg = evidence_config()
        cfg['source_evidence'].update(source_contract=rt.source_contract,
            joint_objective=joint_objective_contract(c, 'cls_head'))
        cfg['source_evidence']['gate_result']['source_freeze_id'] = gate['source_freeze_id']
        c.update(gain_strategy='reliable_evidence', evidence_config=cfg, scheduler_mode='feedback')
        # Same established input, no mutation of the old runtime's sampler.
        ctx['train_loader'] = DataLoader(SixSourceDataset(), batch_size=32)
        validate_runtime_config(c)
        rt = CrossResponseRuntime(model, ctx, c, args, torch.device('cpu'))
    return model, rt, ctx


def test_actual_source_update_enters_gate_without_manual_observe(tmp_path):
    model, rt, ctx = make_runtime(tmp_path)
    opt = torch.optim.AdamW(list(model.parameters()) + rt.main_optimizer_parameters(), lr=.01)
    sc = torch.amp.GradScaler('cuda', enabled=False)
    iterator = iter(ctx['train_loader'])
    for _ in range(2):
        x,y,d,meta = next(iterator)
        plan = rt.begin_batch((d, meta), epoch=1, phase='label')
        out = model(x)
        terms = rt.losses(model, out, y, plan)
        ce = torch.nn.functional.cross_entropy(out['tx_logits'], y)
        base = ce + .7*out['z_dom'].square().mean()
        opt.zero_grad(set_to_none=True)
        before = copy.deepcopy((model.state_dict(), rt.auxiliary.state_dict(), opt.state_dict(), sc.state_dict()))
        rng = capture_rng_state()
        rt.audit_source_update(model, opt, sc, base, ce, terms, training_context={'test': True})
        assert rt.pending_source_update['status'] == 'MEASURED', rt.pending_source_update.get('reason')
        assert first_divergence(before, (model.state_dict(), rt.auxiliary.state_dict(), opt.state_dict(), sc.state_dict())) is None
        assert first_divergence(rng, capture_rng_state()) is None
        rt.backward(model, base, ce, terms, sc)
        sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+rt.main_optimizer_parameters(), .1)
        sc.step(opt)
        sc.update()
        rt.commit(True)
    audit = rt.mechanism_audit_state
    assert audit['paired_updates'] == 2 and len(audit['observations']) == 1
    observation = audit['observations'][0]
    assert observation['evidence']['uncertainty']['pairs'] == 2
    assert len(observation['evidence']['update_value']['leo_groups']) == 15
    assert not observation['result']['authorized'] and not rt.joint_open
    assert rt.mechanism_gate.last_result is not None
    assert (tmp_path / 'cross_response_diagnostics.jsonl').exists()


def test_live_feedback_starts_uniform_until_current_gate_qualifies(tmp_path):
    _, rt, ctx = make_runtime(tmp_path, variant='U5', reliable=True)
    next(iter(ctx['train_loader']))
    assert not rt.joint_open
    assert all(row['qualification'] == 'CLOSED_UNIFORM' for row in rt.sampler.scheduler.last_probability_audit)


def test_phase_change_closes_gate_before_first_loader_ticket(tmp_path):
    model, rt, ctx = make_runtime(tmp_path, variant='U5', reliable=True)
    rt.prepare_training_phase(epoch=1, phase='label')
    row = evidence(0, 'cls_head')
    row['update_value'].update(training_physical_ids=list(rt.source_contract['train'])[:2],
        query_physical_ids=list(rt.source_contract['validation'])[:4], base_risk=2e6)
    row['capability']['baseline_error'] = 2e6
    row['necessity'].update(shuffled_tx_error=2e6, rx_only_error=2e6, head_only_error=2e6)
    assert rt.observe_source_mechanism(model, row)['authorized']
    rt.prepare_training_phase(epoch=2, phase='pseudo')
    next(iter(ctx['train_loader']))
    assert not rt.joint_open
    assert all(row['qualification'] == 'CLOSED_UNIFORM' for row in rt.sampler.scheduler.last_probability_audit)


def test_gate_invalidation_clears_extended_roles_and_live_feedback(tmp_path):
    model, rt, _ = make_runtime(tmp_path)
    row = evidence(0, 'time_fuse')
    row['update_value'].update(training_physical_ids=list(rt.source_contract['train'])[:2],
                               query_physical_ids=list(rt.source_contract['validation'])[:4])
    # Tiny fixture has explicit huge thresholds; make measured gaps huge too.
    row['capability']['baseline_error'] = 2e6
    row['necessity'].update(shuffled_tx_error=2e6, rx_only_error=2e6, head_only_error=2e6)
    row['update_value']['base_risk'] = 2e6
    assert rt.observe_source_mechanism(model, row, extension='time_fuse')['authorized']
    rt._close_source_mechanism('missing_measurement', scope='time_fuse')
    assert not rt.joint_open and rt.extended_identity_module is None
    assert not any(name.startswith('id_backbone.time_fuse.') for name, _ in rt.roles['identity_tail'])


def test_resume_preserves_pending_pair_and_next_gate_observation(tmp_path):
    model, rt, ctx = make_runtime(tmp_path)
    opt = torch.optim.AdamW(list(model.parameters()) + rt.main_optimizer_parameters(), lr=.01)
    sc = torch.amp.GradScaler('cuda', enabled=False)

    def step(model, rt, ctx, opt, sc):
        rt.prepare_training_phase(epoch=1, phase='label')
        x, y, d, meta = next(iter(ctx['train_loader']))
        plan = rt.begin_batch((d, meta), epoch=1, phase='label')
        out = model(x)
        terms = rt.losses(model, out, y, plan)
        ce = torch.nn.functional.cross_entropy(out['tx_logits'], y)
        base = ce + .7 * out['z_dom'].square().mean()
        opt.zero_grad(set_to_none=True)
        rt.audit_source_update(model, opt, sc, base, ce, terms, training_context={'test': True})
        assert rt.pending_source_update['status'] == 'MEASURED'
        rt.backward(model, base, ce, terms, sc)
        sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + rt.main_optimizer_parameters(), .1)
        sc.step(opt)
        sc.update()
        rt.commit(True)

    step(model, rt, ctx, opt, sc)
    assert len(rt.mechanism_audit_state['window']) == 1
    checkpoint = copy.deepcopy((model.state_dict(), rt.state_dict(), opt.state_dict(), sc.state_dict(), capture_rng_state()))
    step(model, rt, ctx, opt, sc)
    expected = copy.deepcopy((model.state_dict(), rt.auxiliary.state_dict(), opt.state_dict(),
        sc.state_dict(), rt.mechanism_audit_state, rt.mechanism_gate.state_dict(),
        rt.sampler.state_dict(), capture_rng_state()))
    resumed_model, resumed, resumed_ctx = make_runtime(tmp_path)
    resumed_opt = torch.optim.AdamW(list(resumed_model.parameters()) + resumed.main_optimizer_parameters(), lr=.01)
    resumed_sc = torch.amp.GradScaler('cuda', enabled=False)
    resumed_model.load_state_dict(checkpoint[0])
    resumed.load_state_dict(checkpoint[1])
    resumed_opt.load_state_dict(checkpoint[2])
    resumed_sc.load_state_dict(checkpoint[3])
    restore_rng_state(checkpoint[4])
    step(resumed_model, resumed, resumed_ctx, resumed_opt, resumed_sc)
    actual = (resumed_model.state_dict(), resumed.auxiliary.state_dict(), resumed_opt.state_dict(),
        resumed_sc.state_dict(), resumed.mechanism_audit_state, resumed.mechanism_gate.state_dict(),
        resumed.sampler.state_dict(), capture_rng_state())
    assert len(resumed.mechanism_audit_state['observations']) == 1
    assert first_divergence(expected, actual) is None
