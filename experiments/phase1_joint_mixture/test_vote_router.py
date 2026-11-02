import torch
from vote_router import VoteRouter,choose_waveforms

def test_choices_are_candidate_predictions_and_equivariant_to_class_relabeling():
 torch.manual_seed(41);m=VoteRouter(9);p=torch.tensor([[0,1,1,2,2,3,4,5,1],[4,4,4,0,1,2,3,5,0],[5,5,5,5,5,5,5,5,5]]);choice,pred=m.choose(p);assert torch.equal(pred,p.gather(1,choice[:,None]).squeeze(1));perm=torch.tensor([3,5,0,4,1,2]);choice2,pred2=m.choose(perm[p]);assert torch.equal(choice,choice2) and torch.equal(pred2,perm[pred])

def test_ties_use_fixed_candidate_priority_and_selected_wave_is_exact():
 m=VoteRouter(9)
 for p in m.parameters():torch.nn.init.zeros_(p)
 predictions=torch.tensor([[0,2,3,4,5,0,1,2,3],[5,4,3,2,1,0,5,4,3]]);choice,pred=m.choose(predictions);assert torch.equal(choice,torch.tensor([1,1])) and torch.equal(pred,predictions[:,1]);waves=torch.arange(2*9*2*8).reshape(2,9,2,8);out=choose_waveforms(waves,choice);assert torch.equal(out,waves[:,1])

def test_trainable_disagreement_has_finite_gradient():
 m=VoteRouter(9);pred=torch.tensor([[0,1,1,2,1,2,3,1,1],[3,2,2,3,2,3,1,2,2],[1,4,4,1,4,1,3,4,4]]);truth=torch.tensor([1,3,4]);loss=torch.nn.functional.cross_entropy(m(pred),truth);loss.backward();assert torch.isfinite(loss) and any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters())
