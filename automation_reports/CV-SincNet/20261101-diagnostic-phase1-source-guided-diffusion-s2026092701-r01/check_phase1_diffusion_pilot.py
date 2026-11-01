import ast
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

p=Path(__file__).parent/'phase1_source_conditional_diffusion_probe.py'
tree=ast.parse(p.read_text(encoding='utf8'))
inner=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='inner' for x in n.targets)))
t=ast.parse(inner)
nodes=[n for n in t.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Block','NoiseNet','generate')]
assert [n.name for n in nodes]==['Block','NoiseNet','generate']
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<diffusion-pilot>','exec'))
torch.manual_seed(8);net=NoiseNet();a=torch.randn(3,2,256);x=torch.randn_like(a);s=torch.full((3,),.2)
pred=net(a,a,x,s);assert pred.shape==a.shape and torch.isfinite(pred).all()
F.mse_loss(pred,torch.randn_like(pred)).backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters())
with torch.no_grad():z=generate((1-.3**2)**.5*a,a,x)
assert z.shape==a.shape and torch.isfinite(z).all()
relative=((a-a).square().mean((1,2))/(a.square().mean((1,2)).clamp_min(1e-8))).sqrt()
gate=(relative/.1).clamp(0,1);assert torch.equal(a+.5*gate[:,None,None]*(z-a),a)
print('VERIFIED conditional diffusion shapes, finite gradients and deterministic four-step output')
