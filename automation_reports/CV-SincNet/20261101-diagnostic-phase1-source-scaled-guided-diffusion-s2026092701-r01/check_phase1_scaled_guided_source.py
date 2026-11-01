import ast,subprocess
from pathlib import Path
p=Path(__file__).parent;source=(p/'phase1_source_scaled_guided_diffusion_probe.py').read_text(encoding='utf8');tree=ast.parse(source)
remote=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='remote' for x in n.targets)))
remote_tree=ast.parse(remote);call=next(n for n in ast.walk(remote_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run')
inner=ast.literal_eval(call.args[0].elts[3]);compile(inner,'<scaled-source-inner>','exec');itree=ast.parse(inner)
pieces=[ast.get_source_segment(inner,n) for n in itree.body if isinstance(n,ast.ClassDef) and n.name in ('Block','NoiseNet')];assert len(pieces)==2
code='''from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core,physical_rich_router_experiment as g
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
models=g.load_models(root,'cpu');teacher,contract,dura,*_=models;cache,clean,roles=core.data(root,contract)
'''+ '\n\n'.join(pieces)+'''
x=core.tensor(np.asarray(clean[:2],dtype=np.float32),'cpu');a=dura(x);net=NoiseNet();sigma=torch.full((2,),.3);eps=torch.randn_like(x);scale=(1-sigma[:,None,None]**2).sqrt();z=scale*a+sigma[:,None,None]*eps
target=(z-scale*x)/sigma[:,None,None];pred=net(z,a,x,sigma);estimate=(z-sigma[:,None,None]*pred)/scale
with torch.no_grad():base=teacher.classify_features(teacher.features(x))
logits=teacher.classify_features(teacher.features(estimate));loss=F.mse_loss(pred,target)+.05*F.kl_div(F.log_softmax(logits,dim=1),F.softmax(base,dim=1),reduction='batchmean')+.2*F.mse_loss(estimate,x)
assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters());assert all(v.grad is None and not v.requires_grad for v in teacher.parameters())
print('VERIFIED_SCALED_F_GUIDED_DIFFUSION_SMOKE',float(loss),'source_rows',len(x),'test_reads',0,'teacher_frozen',True)
'''
remote='''from pathlib import Path
import os,subprocess
r=Path('/home/szu2070436088/2510044040/CV-SincNet');release=r/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=r/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-c',CODE],cwd=release,env=env,check=True,timeout=120)
'''.replace('CODE',repr(code))
proc=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=140)
print(proc.stdout.decode('utf8',errors='replace'));print(proc.stderr.decode('utf8',errors='replace'));raise SystemExit(proc.returncode)
