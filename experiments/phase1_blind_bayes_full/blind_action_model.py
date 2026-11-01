"""Blind short-window complex action and noise estimator."""
import torch
from torch import nn

def effective_kernel(kernel,coefficients,rms):
 ar=coefficients[:,0,128:129];ai=coefficients[:,1,128:129];hr=kernel[:,0];hi=kernel[:,1]
 return torch.stack((ar*hr-ai*hi,ar*hi+ai*hr),1)/rms[:,None,None]

class BlindActionNet(nn.Module):
 def __init__(self):
  super().__init__()
  self.encoder=nn.Sequential(nn.Conv1d(2,32,9,stride=2,padding=4),nn.SiLU(),nn.Conv1d(32,64,7,stride=2,padding=3),nn.SiLU(),nn.Conv1d(64,96,5,stride=2,padding=2),nn.SiLU(),nn.Conv1d(96,128,5,stride=2,padding=2),nn.SiLU(),nn.Flatten())
  self.head=nn.Sequential(nn.Linear(2048,256),nn.SiLU(),nn.Linear(256,129))
  with torch.no_grad():self.head[-1].weight.mul_(.01);self.head[-1].bias.zero_();self.head[-1].bias[-1]=-6.
 def forward(self,x):
  rms=x.square().sum(1).mean(1).sqrt().clamp_min(1e-8);v=self.head(self.encoder(x/rms[:,None,None]));return v[:,:128].reshape(-1,2,64)*rms[:,None,None],v[:,128]+2*rms.log()

def loss(model,x,kernel,coefficients,variance):
 rms=x.square().sum(1).mean(1).sqrt().clamp_min(1e-8);pred,logvar=model(x);target=effective_kernel(kernel,coefficients,rms);pred=pred/rms[:,None,None];den=target.square().sum((1,2)).clamp_min(1e-6);channel=(pred-target).square().sum((1,2))/den;noise=(logvar-variance.clamp_min(1e-10).log()).square();return (channel+.05*noise).mean()
