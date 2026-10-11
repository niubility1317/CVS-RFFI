import argparse
import copy
import json
from pathlib import Path
import torch
from .physical import signal_state,nonlinear_memory_features,repetition_quality,aligned_cycles,known_linear
from .reference import install_reference
from .objectives import soft_identity_kl,identity_weights,physical_weights


def check():
    torch.manual_seed(31);x=torch.randn(8,2,256)
    assert torch.equal(known_linear(x,torch.zeros(8,4)),x)
    assert signal_state(x).shape==(8,40) and nonlinear_memory_features(x).shape==(8,17)
    z=torch.complex(x[:,0],x[:,1]);n=torch.arange(256)
    shifted=z*torch.exp(1j*(.6+.03*n));v=torch.stack((shifted.real,shifted.imag),1)
    assert torch.allclose(nonlinear_memory_features(x),nonlinear_memory_features(v),atol=1e-5,rtol=1e-5)
    a=aligned_cycles(x);assert torch.allclose(a.abs(),z[:,80:160].reshape(8,4,20).abs(),atol=1e-6)
    for value in [x.clone(),torch.zeros_like(x)]:
        value.requires_grad_();loss=signal_state(value).sum()+nonlinear_memory_features(value).sum();loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(value.grad).all()
    # Identity rejection does not delete physically usable observations.
    uncertain=torch.full((8,6),1/6);assert identity_weights(uncertain).sum()==0
    good=torch.tensor([[1.,0.,1.]]).expand(8,-1)
    assert physical_weights(good,'repetition').sum()==8
    student=torch.randn(8,6,requires_grad=True);q=torch.softmax(torch.randn(8,6),1).requires_grad_();w=torch.ones(8,requires_grad=True)
    loss=soft_identity_kl(student,q,w);loss.backward();assert q.grad is None and w.grad is None
    assert student.grad is not None and torch.isfinite(student.grad).all()
    logits=torch.randn(8,6,requires_grad=True);q=torch.softmax(logits.detach(),1)
    same=soft_identity_kl(logits,q,torch.ones(8));same.backward();assert logits.grad.abs().max()<1e-7
    l=soft_identity_kl(student.detach(),q,torch.ones(8));half=soft_identity_kl(student.detach(),q,torch.tensor([1.]*4+[0.]*4))
    per=torch.nn.functional.kl_div(torch.log_softmax(student.detach(),1),q,reduction='none').sum(1)
    assert torch.allclose(half,per[:4].sum()/8)
    from experiments.cvs_stage2_ssl import design as d
    from experiments.cvs_stage2_decouple.design import make_args
    from experiments.cvs_phase1_stack.runtime import installed
    pc=d.parent_config(d.SEEDS[0]);args=make_args(pc,'cpu');args.num_domains,args.input_len=5,256
    with installed(pc) as native:model=native.build_baseline_model(args,torch.device('cpu'))
    before=copy.deepcopy(model.state_dict());contract=install_reference(model)
    assert all(torch.equal(before[n],p) for n,p in model.state_dict().items())
    frontend=next(m.response for n,m in model.named_modules() if hasattr(m,'response_projection'))
    for value in [x.clone(),torch.zeros_like(x)]:
        value.requires_grad_();features=frontend(value)
        assert features.shape==(8,29) and torch.equal(features[:,27],torch.zeros(8))
        assert 'cfo' not in frontend.components(value)
        features.sum().backward();assert torch.isfinite(value.grad).all()
    assert contract['old_reference_state_action_caches_compatible'] is False
    from .sampling import DomainStream
    rx=torch.arange(5).repeat_interleave(30);day=torch.arange(3).repeat_interleave(10).repeat(5)
    stream=DomainStream(rx,day,37);i,p=stream.batch(300)
    assert set(p.tolist())=={1,2}
    assert sorted(i[:150].tolist())==sorted(i[150:].tolist())==list(range(150))
    domains=rx[i]*3+day[i];counts=torch.bincount(domains[:128],minlength=15)
    assert int(counts.max()-counts.min())<=1
    resumed=DomainStream(rx,day,99);resumed.load_state_dict(stream.state_dict())
    a,ap=stream.batch();b,bp=resumed.batch();assert torch.equal(a,b) and torch.equal(ap,bp)
    return dict(status='PASS',zero_action_exact=True,cycle_amplitudes_preserved=True,
        nonlinear_descriptor_constant_linear_phase_invariance=True,zero_IQ_gradient_finite=True,
        reference_zero_slot27=True,no_explicit_frequency_estimator_or_output=True,
        parameter_values_unchanged_by_frontend_install=True,old_feature_caches_marked_incompatible=True,
        soft_KL_fixed_batch_denominator=True,teacher_and_weights_detached=True,
        same_view_KL_gradient_zero=True,identity_and_physical_weights_separate=True,
        source_U_domain_balance=True,complete_UID_traversal_before_reuse=True,source_U_sampler_resume_exact=True,
        source_input='synthetic native architecture, not original E200',target_access=False,
        full_training_integration_verified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);result=check();Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
