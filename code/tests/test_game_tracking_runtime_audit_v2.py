"""Training-mode v2 audit scope and complete caller-state isolation."""
from copy import deepcopy
import torch
import pytest

from scripts.accept_core90_v2_source import acceptance_args, first_difference
from cvsrffi.game_tracking import runtime
from cvsrffi.game_tracking.runtime_audit_v2 import (game_audit_v2, reference_coverage,
                                                  _gradient_reference, _freeze_actual_u)
from cvsrffi.game_tracking.data import build_source, audit_indices_v2
from cvsrffi.game_tracking.source_audit import SourceAuditor, AuditConfig, isolated_rng
from cvsrffi.game_tracking.step_context import prepare_context
from cvsrffi.game_tracking.state import RNGState
from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
from cvsrffi.game_tracking.legacy.options import _loss_weights
from cvsrffi.schedule import build_stage_state


def fixture(epoch=131):
    args=acceptance_args(synthetic=True)
    source=build_source(args)
    model=runtime.build_model(args,len(source.domains),torch.device('cpu')).train()
    proto=PrototypeMemoryBank(args.num_classes,len(source.domains))
    proto._lazy_init(160,torch.device('cpu'),torch.float32)
    batch=next(iter(source.loader('train',18,seed=9,shuffle=True)))
    ubatch=next(iter(source.loader('unlabeled',18))) if epoch>130 else None
    weights=_loss_weights(args,build_stage_state(epoch,args))
    ctx=prepare_context(batch,ubatch,model,deepcopy(model).eval(),args,epoch,1,weights,
                         torch.Generator().manual_seed(3))
    return args,source,model,proto,ctx


def test_missing_capture_cannot_claim_representative_coverage():
    args,source,_,_,_=fixture(1)
    indexes=audit_indices_v2(source,2)
    good=reference_coverage(source,args,indexes['lag_reference'])
    assert good['valid'] and good['expected_groups']==54
    missing=indexes['lag_reference'][2:]
    bad=reference_coverage(source,args,missing)
    assert not bad['valid'] and bad['missing_groups']


def test_v2_train_head_same_target_and_no_caller_mutation():
    old=torch.get_num_threads();torch.set_num_threads(2)
    try:
        args,source,model,proto,ctx=fixture()
        model.id_backbone.cls_head.eval()  # Deliberately mixed flags must persist.
        indexes=audit_indices_v2(source,2)
        auditor=SourceAuditor(AuditConfig(steps=2))
        before=deepcopy(dict(model=model.state_dict(),proto=vars(proto),ctx=vars(ctx),
                            flags={n:m.training for n,m in model.named_modules()},rng=vars(RNGState.capture())))
        metrics,sets=game_audit_v2(model,source,indexes,auditor,args,ctx,0,0,131,proto)
        after=dict(model=model.state_dict(),proto=vars(proto),ctx=vars(ctx),
                   flags={n:m.training for n,m in model.named_modules()},rng=vars(RNGState.capture()))
        assert first_difference(before,after) is None
        assert metrics['schema']=='game_audit_v2' and metrics['coverage_valid']
        assert metrics['lag']['objective_scope']=='current_training_head_objective'
        assert metrics['lag']['head_forward_parity']['max_logit_error'] <= 1e-6
        assert metrics['lag']['sampling']['satellite_direct_loss_weight']==0.
        assert metrics['lag']['status']=='BUDGET_INCONCLUSIVE'
        assert metrics['gradient']['status']=='UNAVAILABLE'
        assert ctx.strong_mask is None
        assert 'domain_fit' in sets and auditor.calls==1
    finally:torch.set_num_threads(old)


def test_full_objective_gradient_replay_and_head_scale_sign_equivalence():
    old=torch.get_num_threads();torch.set_num_threads(2)
    try:
        args,source,model,proto,ctx=fixture()
        # Actual pseudo labels stay unchanged; fixture explicitly exercises selected U.
        ctx.strong_mask=torch.ones_like(ctx.base_mask)
        indexes=audit_indices_v2(source,2)
        before=deepcopy(dict(model=model.state_dict(),proto=vars(proto),ctx=vars(ctx),rng=vars(RNGState.capture())))
        modes=[]
        for head_scale in ('legacy_weighted','separate_head_scale'):
            args.game_head_scale=head_scale
            with isolated_rng(345):
                m=_gradient_reference(model,deepcopy(model.adv_head),source,indexes['gradient_reference'],
                    args,ctx,131,76,None,ctx,proto)
            assert m['scope']=='full_current_training_objective'
            assert m['valid'] and m['direction_imbalance']==pytest.approx(0.,abs=1e-12)
            assert m['nonadversarial_replay_max_abs_error']==0.
            assert m['pseudo_selected']==18 and m['unlabeled_samples']==18
            assert 'labeled_nonadversarial' in m and m['backward_evaluations']==8
            modes.append(m)
        assert modes[0]['id_online']['right_norm']==pytest.approx(modes[1]['id_online']['right_norm'],rel=1e-5)
        after=dict(model=model.state_dict(),proto=vars(proto),ctx=vars(ctx),rng=vars(RNGState.capture()))
        assert first_difference(before,after) is None
    finally:torch.set_num_threads(old)


def test_reference_head_ce_deterioration_cannot_authorize_direction():
    old=torch.get_num_threads();torch.set_num_threads(2)
    try:
        args,source,model,proto,ctx=fixture(80)
        bad=deepcopy(model.adv_head)
        final=[m for m in bad.modules() if isinstance(m,torch.nn.Linear)][-1]
        with torch.no_grad():final.bias[0]+=100.
        with isolated_rng(123):
            metrics=_gradient_reference(model,bad,source,audit_indices_v2(source,2)['gradient_reference'],
                args,ctx,80,4,None,ctx,proto)
        assert metrics['reference_recovered_adv_ce']>metrics['reference_online_adv_ce']+1e-6
        assert metrics['status']=='TRANSFER_FAILURE'
        assert not metrics['valid'] and metrics['reason']=='reference_head_objective_worsened'
        assert metrics['online_recovered']['cosine'] is not None
    finally:torch.set_num_threads(old)
