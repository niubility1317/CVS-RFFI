import math
import torch
from torch import nn
import pytest
from experiments.pamf_iq_physics_compact import design as d
from experiments.pamf_iq_physics_compact.models import objective,make_probes,action_from_arrays,compact_from_teacher
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_unrolled_restore.models import UnrolledRestorer

torch.set_num_threads(2)


def identity(count=2,gain=1.):
    kernel=torch.zeros(count,2,64);kernel[:,0,0]=1
    coefficients=torch.zeros(count,4,256);coefficients[:,0]=gain
    return action_from_arrays(kernel,coefficients)


def arguments():
    clean=torch.ones(2,2,256,requires_grad=True)
    outputs=[torch.zeros(2,2,256,requires_grad=True) for _ in range(3)]
    teacher=[torch.zeros(2,2,256,requires_grad=True) for _ in range(3)]
    base=torch.full((2,2,256),2.,requires_grad=True);native=torch.full((2,),2.,requires_grad=True)
    context=(base,identity(),native);target_variance=torch.ones(2,requires_grad=True)
    probes=[torch.ones(2,256,dtype=torch.complex64),torch.full((2,256),1j,dtype=torch.complex64)]
    return outputs,teacher,clean,20,context,identity(gain=2.),target_variance,probes


def test_physics_block_uses_absolute_variance_and_fixed_source_recipe_weights():
    args=arguments();loss,terms=objective(*args)
    assert torch.allclose(loss,torch.full((2,),1+.1+.1*.25+.01*math.log(2)**2))
    assert torch.allclose(terms['action_probe_nmse'],torch.full((2,),.25))
    loss.sum().backward()
    assert args[2].grad is None and args[6].grad is None and all(v.grad is None for v in args[1])
    assert args[4][0].grad.abs().sum()>0 and args[4][2].grad.abs().sum()>0
    assert d.WEIGHTS==dict(coarse_clean_nmse=.1,action_probe_nmse=.1,noise_log_variance_mse=.01)


@pytest.mark.parametrize('value',[0.,-1.,float('nan')])
def test_bad_native_target_rejected_before_loss(value):
    args=list(arguments());args[6]=torch.tensor([value,1.])
    with pytest.raises(ValueError):objective(*args)


def test_wrong_probe_shape_and_count_are_rejected():
    args=list(arguments());args[-1]=args[-1][:1]
    with pytest.raises(ValueError):objective(*args)


def test_simulation_parameters_are_detached_not_fitted_and_gain_factorization_is_invariant():
    kernel=torch.zeros(2,2,64);kernel[:,0,0]=2;kernel.requires_grad_(True)
    coefficients=torch.zeros(2,4,256);coefficients[:,0]=.5;coefficients.requires_grad_(True)
    target=action_from_arrays(kernel,coefficients)
    probe=torch.randn(2,256,dtype=torch.complex64)
    assert torch.allclose(target(probe),probe,rtol=1e-5,atol=1e-6)
    assert not target.kernel.requires_grad and not target.main.requires_grad


@pytest.mark.parametrize('arm',d.ARMS)
def test_all_actual_architecture_components_receive_gradients_with_no_teacher_or_F_updates(arm):
    torch.manual_seed(2701)
    teacher=UnrolledRestorer('unroll_unshared',ActionRestorer(ComplexTFUNet())).eval().requires_grad_(False)
    with torch.no_grad():
        for name,module in teacher.named_modules():
            if isinstance(module,nn.Conv1d) and name.endswith('.out'):module.weight.normal_(0,.002)
        teacher.frozen.estimator.global_head.weight.normal_(0,.0002)
    before={k:v.clone() for k,v in teacher.state_dict().items()}
    model=compact_from_teacher(teacher,arm);x=torch.randn(2,2,256)*.1;r=x+torch.randn_like(x)*.01
    from experiments.pamf_r17_compact.models import stages
    context=model.estimate_context(r);outputs=stages(model.prior,r,context)[0]
    with torch.no_grad():desired=teacher.stage_outputs(r)[0]
    probes=make_probes(2,'cpu',torch.Generator().manual_seed(d.PROBE_SEED))
    loss,_=objective(outputs,desired,x,1,context,identity(),torch.full((2,),1e-4),probes);loss.mean().backward()
    for part in (model.frozen.coarse,model.frozen.estimator,*model.prior):
        grads=[p.grad for p in part.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(v).all() for v in grads) and sum(float(v.abs().sum()) for v in grads)>0
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(value,before[key]) for key,value in teacher.state_dict().items())
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]


def test_independent_probe_generator_replays_equal_probes_for_both_widths():
    one=make_probes(3,'cpu',torch.Generator().manual_seed(d.PROBE_SEED))
    torch.randn(97)
    two=make_probes(3,'cpu',torch.Generator().manual_seed(d.PROBE_SEED))
    assert len(one)==2 and all(torch.equal(a,b) for a,b in zip(one,two))


def test_fixed_training_permissions_and_both_version_formal_test_matrix():
    config=d.configuration()
    assert config['TX_labels_read']==config['F_weights_read']==config['F_outputs_read']==0
    assert not config['target_training'] and not config['V_training'] and not config['source_performance_gate']
    assert len(config['views'])==7 and len(config['arms'])==2 and config['physics_target_examples']==1260000
    assert config['training_probes_per_IQ']==2 and config['source_physics_records_per_arm']==63
