import copy,json
import torch
from torch.nn import functional as F
from . import design as d
from .frontends import *
def close(a,b,tol=3e-4):
 if not torch.allclose(a,b,atol=tol,rtol=tol):raise AssertionError('Mathematical action failed '+str(float((a-b).abs().max())))
def run(device='cpu'):
 torch.set_num_threads(2);torch.manual_seed(77);x=torch.randn(12,2,256,device=device);z=complex_iq(x);n=torch.arange(256,device=device);rot=2.3*torch.exp(1j*(.7+.09*n));zz=z*rot;xx=torch.stack((zz.real,zz.imag),1);close(polynomial(x),polynomial(xx),.001)
 a=torch.tensor([[1.3,.2],[-.15,.8]],device=device);xx=torch.einsum('ij,bjn->bin',a,x)+torch.tensor([.5,-1.2],device=device)[None,:,None];close(affine(x),affine(xx));close(affine(torch.ones_like(x)),torch.zeros(12,48,device=device))
 spec=torch.fft.fft(z,dim=1);g=(torch.arange(8,device=device)+1).repeat_interleave(32).float()*torch.exp(1j*torch.arange(8,device=device).repeat_interleave(32)*.2);zz=torch.fft.ifft(spec*g,dim=1);close(subband(x),subband(torch.stack((zz.real,zz.imag),1)),.001)
 s=torch.randn(12,256,device=device);b=log_basis(device,s.dtype);close(remove_log_basis(s),remove_log_basis(s+torch.randn(12,5,device=device)@b.T))
 f=fragments(x);g=(torch.rand(32,device=device)+.5)*torch.exp(1j*torch.randn(32,device=device));close(scalar_loops(f),scalar_loops(f*g));close(scalar_loops(torch.ones_like(f)),torch.tensor([1.,0.,1.,0.,1.,0.],device=device)[None].expand(12,-1))
 v=torch.randn(12,32,8,2,device=device,dtype=torch.complex128);aa=torch.eye(2,device=device,dtype=v.dtype)[None,None].expand(12,32,-1,-1)+.1*torch.randn(12,32,2,2,device=device,dtype=v.dtype);trans=torch.einsum('bsij,bsnj->bsni',aa,v);lo,ok=matrix_loops(v);ll,kk=matrix_loops(trans)
 if not ok.all() or not kk.all():raise AssertionError('Toy covariance rank')
 close(lo,ll,1e-9);deg=torch.ones_like(v);lo,ok=matrix_loops(deg)
 if ok.any() or lo.abs().sum():raise AssertionError('Rank1 guard')
 for arm,dim in DIMS.items():
  ff=features(arm,x);zero=features(arm,torch.zeros_like(x))
  if ff.shape!=(12,dim) or not torch.isfinite(zero).all():raise AssertionError('Frontend finite shapes')
  close(ff[:2],torch.cat([features(arm,x[i:i+1]) for i in range(2)]),.001)
 # Counterexamples: ideal individualactions never promise joint/general invariance.
 filt=torch.linspace(.5,2.,256,device=device);zz=torch.fft.ifft(torch.fft.fft(z,dim=1)*filt,dim=1);filtered=torch.stack((zz.real,zz.imag),1)
 if torch.allclose(affine(x),affine(filtered),atol=1e-6):raise AssertionError('Affine accidentallytreated generalfilter invariant')
 if torch.allclose(polynomial(x),polynomial(filtered),atol=1e-6):raise AssertionError('Polynomial accidentallygeneralconvolution invariant')
 for k in range(4):close(phase_rotate(phase_rotate(x,k),4-k if k else 0),x)
 for kind in ('smooth_gain','delay'):close(digital_action(digital_action(x,kind,1.),kind,-1.),x)
 residual,coeff=bounded_tangent_distance(torch.tensor([2.,-.5],device=device),torch.eye(2,device=device));close(coeff,torch.tensor([1.,-.5],device=device));close(residual,torch.tensor(1.,device=device))
 from .model import build,groups
 from experiments.cvs_finite_margin.objective import deterministic
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));m=build(c,device);pg,_=groups(m,c);m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():
   out=m.forward_identity_only(x[:2]);h=out['id_feat_joint'];close(out['tx_logits'],30*F.linear(F.normalize(h,dim=1,eps=1e-4),F.normalize(m.head.weight,dim=1,eps=1e-4)))
   if F.linear(F.normalize(h,dim=1),F.normalize(torch.randn(9,160,device=device),dim=1)).shape!=(2,9):raise AssertionError('Phase2 refit interface')
   if arm=='C4_ORBIT':close(m(x[:2]),m(phase_rotate(x[:2],1)),1e-4)
  if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise AssertionError('Inference state mutated')
  m.train();loss=F.cross_entropy(m(x[:2]),torch.tensor([0,1],device=device));loss.backward()
  if not all(p.grad is None or torch.isfinite(p.grad).all() for g in pg for p in g['params']):raise AssertionError('Finite gradients')
  if arm=='TANGENT':
   m.zero_grad(set_to_none=True)
   from .source import tangent
   aux,_=tangent(m,x[:2],1);aux.backward()
   if m.head.weight.grad is not None:raise AssertionError('Tangent changedcos classweight')
  bad=copy.deepcopy(c);bad['epochs']=31
  try:d.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Epoch budget guard')
  del m
 from experiments.cvs_cross_rx_feedback.sampling import RotatingStreams
 # Actual five RX contract and 25/26 feedback size; U sampler never reads y.
 data=dict(rx=torch.repeat_interleave(torch.tensor([1,3,4,6,8]),108),day=torch.tensor([1,2,3]*180),y=torch.tensor(list(range(6))*90));stream=RotatingStreams(data,77,True)
 for r,n in zip((1,3,4,6,8),(26,26,26,25,25)):
  ix,_=stream.batch(r,n)
  if len(ix)!=n or set(data['y'][ix].tolist())!=set(range(6)):raise AssertionError('FiveRX TXcoverage')
 from .feedback import objective,parameter_selection,bounded_weights
 from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled
 class AcceptedBank:
  def update(self,ids,prediction,confidence,passes):return torch.ones_like(prediction,dtype=torch.bool),dict(test_only=True)
 for arm in ('UNIFORM_SSL','META_SSL'):
  c=d.config(next(r for r in d.rows() if r['arm']==arm));m=build(c,device);pg,_=groups(m,c);names,fp=parameter_selection(m,pg);opt=torch.optim.AdamW(pg,weight_decay=.0001);xx=torch.randn(128,2,256,device=device);yy=torch.arange(128,device=device)%6
  for n in (25,26):
   opt.zero_grad(set_to_none=True);before={k:v.clone() for k,v in m.state_dict().items()};loss,_,detail=objective(m,xx,yy,xx[-n:],yy[-n:],xx,['u'+str(i) for i in range(128)],torch.full((128,),2),AcceptedBank(),opt,fp,c,1111)
   if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()) or detail['U_gradient_groups']!=4 or detail['accepted_U']!=128 or id(m.head.weight) in {id(p) for p in fp}:raise AssertionError('Fullsource feedback/state/head scope')
   loss.backward()
   if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in pg for p in g['params']):raise AssertionError('Actualfullsource feedback gradients')
   if arm=='UNIFORM_SSL' and detail['group_weights']!=[1.]*4:raise AssertionError('Uniform control differs')
   opt.step()
  del m
 if bounded_weights([-1.,0.,-2.,-.1],device).sum()!=0:raise AssertionError('Conflict onlyU mustoff')
 try:validate_unlabeled({'y':torch.zeros(1)})
 except ValueError:pass
 else:raise AssertionError('Utruth schema allowed')
 return dict(status='PASS',arms=12,frontends=6,ordinarycos=True,refit_classes=9,mathematical_actions='gain/phase/linearCFO polynomial;GL2/DC fullrank;piecewiseflat subband;fixedlogB;scalar/matrixpernodegauge;C4featuremean',degeneracy='zeroIQ/rank1/constantloop guards',sourceVperformance_gate=False,query_access=False)
if __name__=='__main__':print(json.dumps(run()))
