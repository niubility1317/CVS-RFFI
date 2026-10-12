import json
import torch
from .statistics import source_scatter,operators,fit,classify
def run():
 torch.set_num_threads(2)
 torch.manual_seed(62);y=torch.arange(6).repeat_interleave(5*20);rx=torch.tensor([1,3,4,6,8]).repeat_interleave(20).repeat(6);ID=torch.randn(6,160)*.1;N=torch.randn(6,5,160)*.02;z=torch.stack([ID[t]+N[t,[1,3,4,6,8].index(r)] for t,r in zip(y,rx)])+torch.randn(600,160)*.02;z=torch.nn.functional.normalize(z,dim=1);native=torch.randn(6,160);states,report=fit(z,y,rx,native);s=source_scatter(z,y,rx);A,_=operators(s)
 for name in ('ORTHOGONAL','OBLIQUE'):
  if not torch.allclose(A[name]@s['D'],torch.zeros_like(s['D']),atol=1e-9) or not torch.allclose(A[name]@A[name],A[name],atol=1e-9):raise AssertionError('ActualPD/P2 projection formula')
 if report['identity_rank']>5 or report['nuisance_rank']>24 or report['GEIV']['GEIV_full_width']!=160 or report['effective_Fisher_min_eigenvalue']< -1e-8:raise AssertionError('Rank/Fisherworkingmodel PSD')
 for name in states:
  a=classify(z[:4],states[name]);b=torch.cat([classify(v[None],states[name]) for v in z[:4]])
  if a.shape!=(4,6) or not torch.isfinite(a).all() or not torch.allclose(a,b,atol=1e-5,rtol=1e-5):raise AssertionError('Ordinarycos query independence')
 states2,_=fit(z,(y+1)%6,rx,native.roll(1,0))
 for name in states:
  if not torch.allclose(classify(z[:4],states[name]).roll(1,1),classify(z[:4],states2[name]),atol=1e-4,rtol=1e-4):raise AssertionError('Class permutation equity')
 # Given nonorthogonal ID and nuisance directions, the oblique example.
 P=torch.tensor([[0.,1.],[0.,1.]]);u=torch.tensor([1.,1.]);n=torch.tensor([1.,0.])
 if not torch.equal(P@n,torch.zeros(2)) or not torch.equal(P@u,u):raise AssertionError('Reportoblique 2D example')
 try:source_scatter(z,y,torch.zeros_like(rx))
 except ValueError:pass
 else:raise AssertionError('Missing sourcecrosscell accepted')
 return dict(status='PASS',methods=list(states),identity_rank=report['identity_rank'],nuisance_rank=report['nuisance_rank'],PD_P2=True,class_permutation=True,perpacket=True,Fisher_Gaussian_PSD=True,ordinarycos=True)
if __name__=='__main__':print(json.dumps(run()))
