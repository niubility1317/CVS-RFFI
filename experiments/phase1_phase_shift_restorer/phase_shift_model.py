"""Independent single-pass phase/timing anchored IQ restoration network."""
import torch
from torch import nn
from torch.nn import functional as F
from phase_factor_model import AdaptivePhaseFactor
from phase1_axial_spectral_mask import AxialSpectralMask

def configure_training_backend(device):
 if torch.device(device).type=='cpu':torch.backends.mkldnn.enabled=False

class PhaseShiftRestorer(nn.Module):
 def __init__(self,phase):
  super().__init__();self.phase=phase.eval().requires_grad_(False);self.body=AxialSpectralMask()
 def train(self,mode=True):
  super().train(mode);self.phase.eval();return self
 def anchor(self,x):
  if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Expected [batch,2,256] IQ')
  with torch.no_grad():factor=self.phase(x)
  shifted=F.pad(x,(64,64),mode='reflect')[:,:,88:344];z=torch.complex(shifted[:,0],shifted[:,1])*factor[:,None]
  return torch.stack((z.real,z.imag),1)
 def forward(self,x):
  return self.body(self.anchor(x))
