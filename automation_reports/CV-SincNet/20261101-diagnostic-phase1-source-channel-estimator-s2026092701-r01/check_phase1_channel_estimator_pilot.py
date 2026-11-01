import ast
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
p=Path(__file__).parent/'phase1_source_channel_estimator_probe.py';tree=ast.parse(p.read_text(encoding='utf8'))
inner=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='inner' for x in n.targets)))
t=ast.parse(inner);node=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='ChannelNet')
exec(compile(ast.Module(body=[node],type_ignores=[]),'<channel-estimator>','exec'))
torch.manual_seed(11);net=ChannelNet();x=torch.randn(3,4,256);amp=torch.randn(3,1);h=F.normalize(torch.randn(3,124),dim=1);aux=torch.randn(3,2)
pred=net(x,amp);assert pred.shape==(3,126);loss=(pred[:,:124]-h).square().sum(1).mean()+.05*F.mse_loss(pred[:,124],aux[:,0])+.01*F.mse_loss(pred[:,125],aux[:,1]);loss.backward()
assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters())
print('VERIFIED source channel estimator shape and finite gradients')
