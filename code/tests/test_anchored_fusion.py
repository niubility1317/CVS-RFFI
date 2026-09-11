import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.anchored_fusion import mix_log_probs,realize_actions,action_utilities,utility_features,UtilityGate


def test_actual_soft_mixture_can_select_neither_expert():
    p0=torch.tensor([[.40,.35,.25]],dtype=torch.double)
    pg=torch.tensor([[.25,.35,.40]],dtype=torch.double)
    mixed=mix_log_probs(p0.log(),pg.log(),.5)
    assert mixed.argmax(-1).item()==1
    out=realize_actions(p0.log(),pg.log(),protection_threshold=1.)
    u=action_utilities(out['log_probabilities'],torch.tensor([1]),p0.argmax(-1),lambda_h=2)
    assert u[0,2]==1 and u[0,0]==0 and u[0,4]==0
    torch.testing.assert_close(mix_log_probs(p0.log(),pg.log(),0.),p0.log())
    torch.testing.assert_close(mix_log_probs(p0.log(),pg.log(),1.),pg.log())


def test_protection_preserves_unique_h0_and_labels_realized_actions():
    p0=torch.tensor([[.6,.3,.1],[.4,.35,.25],[.5,.5,0.]],dtype=torch.double).clamp_min(1e-100)
    pg=torch.tensor([[.1,.8,.1],[.25,.35,.4],[.1,.8,.1]],dtype=torch.double)
    out=realize_actions(p0.log(),pg.log(),protection_threshold=.2)
    assert out['alpha'][0,-1] < .3/(.3+.7)
    assert (out['log_probabilities'][0].argmax(-1)==0).all()
    assert out['log_probabilities'][1,2].argmax()==1
    assert (out['alpha'][2]==0).all()
    u=action_utilities(out['log_probabilities'],torch.tensor([1,1,1]),p0.argmax(-1),lambda_h=2)
    assert (u[0]==0).all()  # Protection retained H0's wrong answer; no invented rescue.


def test_feature_and_action_class_permutation_invariance():
    torch.manual_seed(9);s0=torch.randn(13,6);sg=torch.randn(13,6);q=torch.rand(13,3);coverage=torch.ones(13)
    p=torch.tensor([5,1,0,4,2,3])
    a=utility_features(s0,sg,q,coverage);b=utility_features(s0[:,p],sg[:,p],q,coverage)
    torch.testing.assert_close(a,b)
    left=realize_actions(s0.log_softmax(-1),sg.log_softmax(-1),protection_threshold=.2)
    right=realize_actions(s0[:,p].log_softmax(-1),sg[:,p].log_softmax(-1),protection_threshold=.2)
    torch.testing.assert_close(left['alpha'],right['alpha'])
    torch.testing.assert_close(left['log_probabilities'][:,:,p],right['log_probabilities'])


def test_gate_no_positive_utility_falls_back_and_freezes():
    torch.manual_seed(8);x=torch.randn(20,11,requires_grad=True)
    y=torch.zeros(20,5);y[:,1:]=-1
    gate=UtilityGate().fit(x,y,role='L_s')
    assert (gate.choose(x)['action']==0).all()
    assert x.grad is None
    with pytest.raises(RuntimeError,match='frozen'):gate.fit(x,y,role='L_s')
    with pytest.raises(ValueError,match='L_s'):UtilityGate().fit(x,y,role='V')
    loaded=UtilityGate.from_state_dict(gate.state_dict())
    torch.testing.assert_close(loaded.predict_utility(x),gate.predict_utility(x))


def test_extreme_logits_remain_finite_and_normalized():
    p0=torch.tensor([[1000.,-1000.,0.]]).log_softmax(-1)
    pg=torch.tensor([[-1000.,1000.,0.]]).log_softmax(-1)
    for alpha in (0.,.25,.5,.75,1.):
        out=mix_log_probs(p0,pg,alpha)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out.exp().sum(-1),torch.ones(1,dtype=out.dtype))
def test_nonfinite_expert_row_falls_back_without_affecting_others():
    s0=torch.tensor([[2.,1.,0.],[1.,0.,2.]])
    sg=torch.tensor([[float('nan'),1.,0.],[3.,0.,1.]])
    result=realize_actions(s0,sg,protection_threshold=1.)
    assert (result['alpha'][0]==0).all() and (result['reason'][0]==3).all()
    assert torch.isfinite(result['log_probabilities']).all()
    torch.testing.assert_close(result['log_probabilities'][0],s0[0].double().log_softmax(-1).expand(5,-1))
