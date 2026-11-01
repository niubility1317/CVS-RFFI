import ast
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

p=Path(__file__).parent/'phase1_source_adaptive_diffusion_probe.py';tree=ast.parse(p.read_text(encoding='utf8'))
inner=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='inner' for x in n.targets)))
t=ast.parse(inner);nodes=[n for n in t.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Block','NoiseNet','generate_from')]
assert [n.name for n in nodes]==['Block','NoiseNet','generate_from']
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<adaptive-diffusion>','exec'))
torch.manual_seed(2);net=NoiseNet();a=torch.randn(3,2,256);x=torch.randn_like(a)
with torch.no_grad():
 for sigma in (.1,.2,.3):
  z=generate_from(a,x,sigma);assert z.shape==a.shape and torch.isfinite(z).all()
print('VERIFIED adaptive diffusion starts and finite waveform shapes')
