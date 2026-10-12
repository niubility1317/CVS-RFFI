import math,torch
from torch import nn
from torch.nn import functional as F
from . import design as d
from experiments.cvs_complement_cnn.model import WaveCNN,canonical
def rotate(x,phi):
 if phi.shape!=(len(x),):raise ValueError('Perpacket global phase required')
 co=phi.cos()[:,None];si=phi.sin()[:,None];return torch.stack((co*x[:,0]-si*x[:,1],si*x[:,0]+co*x[:,1]),1)
class Moments(nn.Module):
 def forward(self,x):return torch.cat((x.mean(-1),x.std(-1,unbiased=False)),1)
class PhasePoolCNN(WaveCNN):
 def __init__(self,c):
  super().__init__();self.config=c;self.augmented_packets=0
  if c['pool']:self.readout=nn.Sequential(Moments(),nn.Linear(256,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,160),nn.LayerNorm(160))
  self.layer_info.append(dict(level='readout',pool=c['pool'],phase_train=c['phase_train'],phase_label_policy='globalcomplexunitrotation leavesTXhardware unchanged;not arbitraryRXinvariance'))
 def forward(self,x):
  if self.training and self.config['phase_train']:
   x=rotate(x,(torch.rand(len(x),device=x.device,dtype=x.dtype)*2-1)*math.pi);self.augmented_packets+=len(x)
  return super().forward(x)
def build(c,fit_x,device):d.validate(c);torch.manual_seed(c['model_seed']);return PhasePoolCNN(c).to(device)
def groups(model,c):
 for p in model.parameters():p.requires_grad_(True)
 return [dict(params=list(model.parameters()),name='wholeCNN',lr=.0002)],dict(wholeCNN=[n for n,p in model.named_parameters()])
