import torch
from torch import nn
from torch.nn import functional as F
from . import design as d
def canonical(x):
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not torch.isfinite(x).all():raise ValueError('Finite perpacket IQ256 required')
 z=torch.complex(x[:,0].double(),x[:,1].double());z=z/z.abs().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)
 w=z[:,80:160];cross=(w[:,20:]*w[:,:-20].conj()).sum(-1);slope=torch.where(cross.abs()>1e-12,torch.atan2(cross.imag,cross.real)/20,torch.zeros_like(cross.real));t=torch.arange(256,device=x.device,dtype=torch.float64)
 z=z*torch.polar(torch.ones_like(slope[:,None]*t[None,:]),-slope[:,None]*t[None,:]);return torch.stack((z.real,z.imag),1).float()
class WaveCNN(nn.Module):
 def __init__(self):
  super().__init__();layers=[];channels=2
  for width,k in zip((32,64,128,128),(9,5,5,3)):
   layers.extend([nn.Conv1d(channels,width,k,padding=k//2),nn.GroupNorm(8,width),nn.GELU(),nn.AvgPool1d(2)]);channels=width
  self.encoder=nn.Sequential(*layers);self.readout=nn.Sequential(nn.Flatten(),nn.Linear(2048,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,160),nn.LayerNorm(160));self.class_weight=nn.Parameter(torch.randn(6,160)*.02);self._current_q=None
  self._identity_input_policy=dict(explicit_CFO_identity_input=False,function_policy='no_explicit_CFO_training_and_prediction',source_training_policy='no_explicit_CFO',physical_compensation='fixedpacketlag20in80:160only;no knownpreambleclaim')
  self.layer_info=[dict(level=f'CNNstage{i+1}',channels=width,time=256//(2**(i+1))) for i,width in enumerate((32,64,128,128))]
 def forward(self,x):
  h=self.readout(self.encoder(canonical(x)));return 16*F.normalize(h,dim=1)@F.normalize(self.class_weight,dim=1).T
def build(c,fit_x,device):d.validate(c);torch.manual_seed(c['model_seed']);return WaveCNN().to(device)
def groups(model,c):
 for p in model.parameters():p.requires_grad_(True)
 return [dict(params=list(model.parameters()),name='wholeCNN',lr=.0002)],dict(wholeCNN=[n for n,p in model.named_parameters()])
