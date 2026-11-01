import torch,sys,json,pathlib
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');sys.path.insert(0,str(root/'releases/20261102-phase1-phase-shift-student-s2026092701-r01-release/experiments/phase1_phase_shift_restorer'))
from phase1_axial_spectral_mask import AxialSpectralMask
torch.set_num_threads(int(sys.argv[1]));torch.backends.mkldnn.enabled=bool(int(sys.argv[2]));torch.manual_seed(123);m=AxialSpectralMask().train();named=list(m.named_parameters());params=[p for n,p in named];out=[]
for i in range(4):
 x=torch.randn(16,2,256);v=m(x);loss=v.square().mean();g0=torch.autograd.grad(loss*0,params,retain_graph=True,allow_unused=True);g1=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);rows=[]
 for (n,p),z in zip(named,g0):
  if z is not None:rows.append((float(z.detach().double().abs().max()),n,int((~torch.isfinite(z)).sum())))
 rows.sort(reverse=True);out.append({'iteration':i,'zero_max':rows[0][0],'top':rows[:3],'zero_nonfinite':sum(r[2] for r in rows)})
print(json.dumps({'torch':torch.__version__,'threads':torch.get_num_threads(),'mkldnn':torch.backends.mkldnn.enabled,'cases':out},allow_nan=False))
