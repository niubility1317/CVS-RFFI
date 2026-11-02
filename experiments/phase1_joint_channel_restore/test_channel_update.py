import torch
from channel_update import design_matrix,update_channel
from experiments.phase1_blind_bayes_validation.blind_action_matrix import action_matrix

def test_signal_design_agrees_with_fixed_channel_operator():
 torch.manual_seed(19);s=torch.randn(2,2,256);h=torch.randn(2,2,64);lhs=torch.bmm(design_matrix(s),h.reshape(2,128,1));rhs=torch.bmm(action_matrix(h),s.reshape(2,512,1));assert torch.allclose(lhs,rhs,atol=2e-5,rtol=1e-5)

def test_regularized_channel_update_improves_known_forward_residual():
 torch.manual_seed(20);s=torch.randn(2,2,256);truth=torch.randn(2,2,64);initial=truth+.1*torch.randn_like(truth);X=design_matrix(s);x=torch.bmm(X,truth.reshape(2,128,1)).reshape_as(s);before=(torch.bmm(X,initial.reshape(2,128,1)).reshape_as(s)-x).square().sum();fixed=update_channel(s,x,initial,.01);after=(torch.bmm(X,fixed.reshape(2,128,1)).reshape_as(s)-x).square().sum();assert torch.isfinite(fixed).all() and after<before
 same=update_channel(s,x,truth,1.);assert torch.allclose(same,truth,atol=1e-5,rtol=1e-5)
