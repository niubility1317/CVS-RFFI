import json
from pathlib import Path
from types import SimpleNamespace
import torch
from cvsrffi.xuc_fusion.full_dr_objective import FULL_OPTIONS
from cvsrffi.xuc_fusion.activation import ActivationLedger
from cvsrffi.xuc_fusion.runtime import resolve_args
from cvsrffi.xuc_fusion.tickets import TicketStream
from cvsrffi.game_tracking.data import SourceData
from cvsrffi.daot_training import compute_daot_batch_objective

ROOT=Path(__file__).resolve().parents[1]

def test_full_matrix_parent_scope_and_budget():
    m=json.loads((ROOT/'configs/matrix_dr_full.json').read_text())
    parents={r['id']:r for r in json.loads((ROOT/'configs/matrix15.json').read_text())['runs']}
    assert len(m['runs'])==9 and m['common']['final_prediction_count']==9*672000
    assert m['common']['total_main_steps']==44400
    assert m['common']['full_dr_options']==FULL_OPTIONS
    assert sum(r.get('dr_full_extensions',False) for r in m['runs'])==7
    for r in m['runs']:
        assert r['steps_per_epoch']==222 and r['unlabeled_batch']==256
        if r.get('dr_full_extensions'):
            for k in ['x_enabled','u_enabled','control','curriculum','ticket_curriculum_enabled','cstar_action_enabled','carrier']:
                assert r[k]==parents[r['parent_row']][k]
    assert all(FULL_OPTIONS['daot_lambda_'+k]>0 for k in ['orbit_z','orbit_logit','orbit_proto','orbit_relation','tangent','nuisance','fingerprint'])
    assert FULL_OPTIONS['rc4_enable_negative'] and FULL_OPTIONS['rc4_total_identity_effective_budget']==0
    assert FULL_OPTIONS['rc4_identity_tail_partial_conditional_final']>0

def test_full_budget_consumes_all_U_and_full_L_batches():
    m=json.loads((ROOT/'configs/matrix_dr_full.json').read_text())
    row=next(r for r in m['runs'] if r['id']=='F-M11')
    args=resolve_args(json.loads((ROOT/'configs/core90_recipe_reference.json').read_text()),row,dataset='unused',output='unused',device='cpu')
    # Index-only datasets exercise real ticket and real U batch scheduling without IQ allocation.
    source=SimpleNamespace(train=range(6300),unlabeled=range(56700))
    source.unlabeled_epoch_loader=lambda *a,**kw: SourceData.unlabeled_epoch_loader(source,*a,**kw)
    stream=TicketStream(source,args,row);assert stream.steps==222
    stream.fill();epoch=[t for t in stream.pending if t.epoch==1]
    assert len(epoch)==222 and all(len(t.labeled)==128 for t in epoch)
    ids=[i for t in epoch for i in t.unlabeled]
    assert len(ids)==len(set(ids))==56700 and set(ids)==set(range(56700))

def test_fingerprint_native_hinge_gradient_and_satisfied_zero():
    # Controlled feature-level diagnostic, never used to alter formal IQ/routes/thresholds.
    for offset,positive in [(0.01,True),(0.2,False)]:
        z=torch.tensor([[1.,0.],[1.,0.]],requires_grad=True)
        pert=torch.tensor([[1.,offset],[1.,offset]],requires_grad=True)
        logits=torch.zeros(2,6,requires_grad=True)
        student=dict(z_id=z,tx_logits=logits)
        teacher=dict(z_id=z.detach(),tx_logits=logits.detach())
        result=compute_daot_batch_objective(student_clean=student,student_channel=student,
            teacher_views=[teacher]*3,reliability=torch.ones(2,3),importance=torch.ones(2,3),
            recoverability=torch.ones(2),orbit_scale=1.,tangent_scale=1.,weights={'fingerprint':.1},
            coverage_floor=.1,huber_beta_min=.1,temperature=1.,fingerprint_perturbed=dict(z_id=pert),
            fingerprint_minimum=.5,tangent_delta=.05)
        grad=torch.autograd.grad(result['loss'],pert)[0]
        assert torch.isfinite(grad).all()
        assert (float(result['loss'])>0)==positive
        assert (float(grad.norm())>0)==positive

def test_activation_never_confuses_configuration_with_gradient_or_C2():
    ledger=ActivationLedger(dict(id='test',x_enabled=False,u_enabled=False,cstar_action_enabled=False))
    ledger.observe(dict(step=0,action='CORRECT',daot_rc4=dict(weighted_components={'daot_fingerprint':0.,'daot_tangent':0.})))
    cold=ledger.report(1)
    assert cold['components']['daot_fingerprint']['status']=='WARMUP_NOT_DUE'
    assert cold['cstar_actual_control_actions']==0
    assert ledger.report(61)['components']['daot_tangent']['status']=='ENABLED_NO_NONZERO_EVIDENCE'
