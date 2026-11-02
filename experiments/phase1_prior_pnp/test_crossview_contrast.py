import math
import pytest
import torch
from crossview_contrast import paired_class_contrast

def test_same_record_and_same_class_positives_match_manual():
 a=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]],dtype=torch.float64);b=torch.tensor([[.8,.6],[0.,1.],[-.8,.6]],dtype=torch.float64);labels=torch.tensor([0,0,-1]);actual=paired_class_contrast(a,b,labels,temperature=.5);features=torch.nn.functional.normalize(torch.cat((a,b)),dim=1);total=0.
 for i in range(6):
  sims=[float(features[i]@features[j])/.5 for j in range(6) if j!=i];den=math.log(sum(math.exp(x) for x in sims));pos=[j for j in range(6) if j!=i and (i%3==j%3 or (int(labels[i%3])>=0 and labels[i%3]==labels[j%3]))];total+=sum(den-float(features[i]@features[j])/.5 for j in pos)/len(pos)
 assert actual.item()==pytest.approx(total/6,rel=1e-12,abs=1e-12)

def test_unknowns_do_not_become_false_class_positives_and_gradients_flow():
 a=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True);b=torch.tensor([[1.,.1],[.1,1.]],requires_grad=True);unknown=torch.tensor([-1,-1]);known=torch.tensor([0,0]);loss=paired_class_contrast(a,b,unknown);same=paired_class_contrast(a,b,known);assert torch.isfinite(loss) and loss<same;loss.backward();assert torch.isfinite(a.grad).all() and a.grad.abs().sum()>0 and b.grad.abs().sum()>0

def test_one_pair_and_invalid_temperature():
 x=torch.tensor([[.4,.6]],requires_grad=True);assert paired_class_contrast(x,x,torch.tensor([-1])).item()==pytest.approx(0.,abs=1e-7)
 with pytest.raises(ValueError):paired_class_contrast(x,x,torch.tensor([-1]),temperature=0.)
