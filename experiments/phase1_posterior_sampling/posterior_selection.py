"""Select an actual posterior draw by a weighted frozen-classifier consensus."""
import torch

def choose_consensus(probabilities,logweights):
 if probabilities.ndim!=3 or logweights.shape!=probabilities.shape[:2] or probabilities.shape[-1]<2:raise ValueError('Candidate class probabilities and mixture logweights required')
 if not torch.isfinite(probabilities).all() or not torch.isfinite(logweights).all() or (probabilities<0).any() or (probabilities>1).any() or not torch.allclose(probabilities.sum(-1),torch.ones_like(probabilities[...,0]),atol=1e-5,rtol=1e-5):raise ValueError('Valid finite class probabilities required')
 weights=torch.softmax(logweights,dim=1);mass=(weights[:,:,None]*probabilities).sum(1);preferred=mass.argmax(1);support=probabilities.gather(2,preferred[:,None,None].expand(-1,probabilities.shape[1],1)).squeeze(-1);choice=(weights*support).argmax(1);return choice,preferred

def select_waveforms(waves,choice):
 if waves.ndim<3 or choice.shape!=(len(waves),) or choice.dtype!=torch.long or (choice<0).any() or (choice>=waves.shape[1]).any():raise ValueError('Candidate waveforms and valid choices required')
 return waves[torch.arange(len(waves),device=waves.device),choice]
