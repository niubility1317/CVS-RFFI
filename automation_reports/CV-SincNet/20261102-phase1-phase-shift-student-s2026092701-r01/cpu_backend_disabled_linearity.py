import torch,sys,json,pathlib
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');sys.path.insert(0,str(root/'releases/20261102-phase1-phase-shift-student-s2026092701-r01-release/experiments/phase1_phase_shift_restorer'))
from phase_shift_model import PhaseShiftRestorer
from phase_factor_model import AdaptivePhaseFactor
torch.set_num_threads(2);torch.backends.mkldnn.enabled=False;torch.manual_seed(123);m=PhaseShiftRestorer(AdaptivePhaseFactor()).train();params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.0001);bad=[]
for i in range(12):
 x=torch.randn(16,2,256);out=m(x);loss=out.square().mean();g0=torch.autograd.grad(loss*0,params,retain_graph=True,allow_unused=True);g1=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);g2=torch.autograd.grad(loss*2,params,retain_graph=True,allow_unused=True)
 zero=max(float(g.abs().max()) for g in g0 if g is not None);diff=max(float((b-2*a).abs().max()) for a,b in zip(g1,g2) if a is not None and b is not None);mag=max(float(g.abs().max()) for g in g1 if g is not None)
 if not (zero==0 and diff<=1e-5*max(1.,mag)):
  bad.append({'step':i,'zero_max':zero,'scale_diff':diff,'grad_max':mag});print('VIOLATION',json.dumps(bad[-1]),flush=True);break
 opt.zero_grad(set_to_none=True)
 for p,g in zip(params,g1):p.grad=g
 torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
print('SYNTHETIC_LINEARITY',json.dumps({'steps':i+1,'violations':bad},allow_nan=False),flush=True)
