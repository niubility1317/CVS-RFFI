import copy
from types import SimpleNamespace
import pytest
import torch
from SSDG.train_ssdg import build_arg_parser, _validate_daot_config, _compute_pair_unlabeled_step
from cvsrffi.pair_reform_runtime import compute_pair_batch,StepStateTransaction
from model_dual_cvsincnet import DualCVSincNetDisentangle


def args_for(mode):
    a=build_arg_parser().parse_args(['--output_dir','unused','--pair_reform',mode,
        '--use_muse_ssdg','true','--muse_level','M1','--sat_training_mode','concat_masked','--sat_cons_start_epoch','1'])
    a.pair_start_epoch=1;a.pair_pseudo_start_epoch=1
    _validate_daot_config(a)
    return a


def test_safe_actual_dual_backbone_forward_backward():
    torch.set_num_threads(2)
    m=DualCVSincNetDisentangle(num_classes=3,num_domains=2,input_len=256)
    t=copy.deepcopy(m).eval()
    x=torch.randn(2,2,256)
    out=m(torch.cat([x,x+.02]),return_aux=True,domain_labels=torch.zeros(4,dtype=torch.long))
    c={k:v[:2] for k,v in out.items() if torch.is_tensor(v) and v.ndim and len(v)==4}
    l={k:v[2:] for k,v in out.items() if torch.is_tensor(v) and v.ndim and len(v)==4}
    a=args_for('safe')
    r=compute_pair_batch(model=m,teacher=t,clean=x,channel=x+.02,student_clean=c,student_channel=l,
        domains=torch.zeros(2,dtype=torch.long),labels=torch.zeros(2,dtype=torch.long),
        metadata={'snr_db':torch.ones(2)*20},args=a,epoch=1,physical_ids=[1,2])
    loss=r['loss']+out['tx_logits'].square().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert all(p.grad is None for p in t.parameters())


def test_unified_u_has_one_student_pair_and_no_true_labels():
    from test_pair_reform_runtime import Toy
    m=Toy(); t=Toy().eval(); a=args_for('point')
    x=torch.randn(4,4)
    def augment(x,scenario,args,**kwargs):
        return x+.1,{'snr_db':torch.ones(len(x))*20}
    r,losses,out,mask=_compute_pair_unlabeled_step(model=m,ema_model=t,x_u=x,d_u=None,
        args=a,epoch=1,physical_ids=[1,2,3,4],apply_sat_fn=augment)
    assert m.calls==[8] and t.calls==[8]
    assert losses['identity'].item()==0 and 'orbit_logit' in r['weighted_components']
    r['loss'].backward()


def test_incompatible_safe_stacking_rejected():
    a=build_arg_parser().parse_args(['--output_dir','unused','--pair_reform','safe','--pair_route_weight','.1',
        '--use_muse_ssdg','true','--muse_level','M1'])
    with pytest.raises(ValueError,match='replacement'):
        _validate_daot_config(a)


@pytest.mark.parametrize('overrides,match', [
    ({'use_muse_ssdg': False}, 'MUSE'),
    ({'muse_level': 'M0'}, 'MUSE'),
    ({'sat_anchor_ssl': True}, 'SAT anchor'),
    ({'pair_weight': 0., 'pair_tangent_weight': .1}, 'direction'),
    ({'pair_direction_ratio': 0., 'pair_route_weight': .1}, 'direction'),
    ({'pair_direction_delta': 0., 'pair_tangent_weight': .1}, 'direction'),
    ({'pair_direction_scale': float('nan'), 'pair_tangent_weight': .1}, 'direction'),
    ({'sat_training_mode':'disabled'}, 'concat'),
    ({'sat_cons_start_epoch':20}, 'LEO supervision'),
    ({'fasttrust_rc4':True}, 'RC4'),
])
def test_pair_configuration_rejects_silent_or_invalid_execution(overrides, match):
    a=args_for('point')
    for key,value in overrides.items():
        setattr(a,key,value)
    with pytest.raises(ValueError,match=match):
        _validate_daot_config(a)


@pytest.mark.parametrize('overrides', [
    {'id_feature_key': 'feat_cls'}, {'id_feature_key': 'base'},
    {'arch_family': 'resnet18_1d'}, {'sat_anchor_adapter': True},
])
def test_safe_configuration_rejects_incompatible_classifier_space(overrides):
    a=args_for('safe')
    for key,value in overrides.items():
        setattr(a,key,value)
    with pytest.raises(ValueError,match='safe region'):
        _validate_daot_config(a)


@pytest.mark.parametrize('mode', ['point','asymmetric','safe'])
def test_supported_pair_configuration_preserves_core_and_enables_teacher(mode):
    a=args_for(mode)
    assert a.use_muse_ssdg and a.muse_level == 'M1'
    assert a.use_ema_teacher and not a.use_adv3b02_daot_stn
    assert not a.use_daot_nuisance_head
