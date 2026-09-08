from copy import deepcopy
import pytest
import torch
from test_a1_scratch import parsed_rows, train
from cvsrffi.a1_r3_objective import r3_pair_objective
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint


@pytest.fixture
def setup_model():
    torch.set_num_threads(2)
    torch.manual_seed(392005)
    args=parsed_rows()[1]
    train._validate_daot_config(args)
    merged=train._apply_model_cli_args(train.merge_checkpoint_args({},args,input_len=256,num_domains=15),args)
    model=train.build_baseline_model(merged,torch.device('cpu'))
    return model,args


def objective(model,args,epoch,step=2,role='U_s'):
    return r3_pair_objective(model=model,clean_iq=torch.randn(4,2,256),
        domains=torch.arange(4),physical_ids=tuple('abcd'),role=role,args=args,
        epoch=epoch,batch_idx=2,optimizer_step=step,apply_sat_fn=train.apply_sat_channel_for_scenario)


@pytest.mark.parametrize('epoch,step,expected',[(1,2,(1,0,0)),(41,2,(1,0,0)),
    (90,2,(1,1,1)),(91,3,(0,0,0)),(91,4,(1,1,1)),(161,2,(1,1,1))])
def test_real_r3_activation_and_finite_gradients(setup_model,epoch,step,expected):
    model,args=setup_model
    loss,logs=objective(model,args,epoch,step)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for name,active in zip(('self','swap','shared'),expected):
        assert logs[f'train/r3_u_s_{name}_active']==active
        if active: assert logs[f'train/r3_u_s_{name}_weighted'].abs()>0
    assert logs['train/r3_u_s_eta_valid']==0 and logs['train/r3_u_s_eta_weighted']==0
    if expected[0]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.a1_r3.content.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.a1_r3.fingerprint_operator.parameters())


def test_control_matches_rng_and_buffers_with_zero_aux_gradient(setup_model):
    model,args=setup_model
    a,b=deepcopy(model),deepcopy(model)
    torch.manual_seed(77); args.a1_r3_aux_scale=0
    zero,logs=objective(a,args,90); rng=torch.get_rng_state().clone(); zero.backward()
    torch.manual_seed(77); args.a1_r3_aux_scale=1
    active,_=objective(b,args,90)
    assert torch.equal(rng,torch.get_rng_state())
    assert zero.item()==0 and active.abs()>0
    for k,v in a.named_buffers(): torch.testing.assert_close(v,dict(b.named_buffers())[k],rtol=0,atol=0)
    assert all(p.grad is None or p.grad.eq(0).all() for p in a.parameters())


def test_exact_rebuild_and_all_identity_paths(setup_model):
    model,args=setup_model
    model.eval()
    rebuilt,audit=build_exact_ssdg_model_from_checkpoint({'args':vars(args),'model':model.state_dict()},
        input_len=256,device=torch.device('cpu'))
    rebuilt.eval(); x=torch.randn(4,2,256)
    with torch.no_grad():
        full=model(x,return_aux=True)
        identity=model.forward_identity_only(x)
        factors=model.forward_r3_factors(x)
        torch.testing.assert_close(full['z_id'],identity['z_id'],rtol=0,atol=0)
        torch.testing.assert_close(full['z_id'],factors.factors.z_f_id,rtol=0,atol=0)
        torch.testing.assert_close(full['tx_logits'],model(x,return_aux=False),rtol=0,atol=0)
        torch.testing.assert_close(full['tx_logits'],rebuilt(x,return_aux=True)['tx_logits'],rtol=0,atol=0)
    assert audit['missing_keys']==audit['unexpected_keys']==0
    assert factors.decode.decoder_mode=='control'


def test_ema_formal_head_refreshes_after_cached_calibration(setup_model):
    model,args=setup_model
    ema=deepcopy(model).eval()
    head=ema.a1_r3_identity_head
    x=torch.randn(4,160)
    with torch.no_grad():
        before=head(x).clone()
        model.a1_r3_identity_head.weight.add_(torch.randn_like(head.weight))
        train._update_ema_model(ema,model,0.5)
        actual=head(x).clone()
        head._norm_weight_cache_key=None
        expected=head(x).clone()
    assert not torch.equal(actual,before)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize('role',['query','V'])
def test_no_query_or_validation_optimizer_path(setup_model,role):
    model,args=setup_model
    with pytest.raises(ValueError): objective(model,args,1,role=role)
