"""Input-only confidence selector and source-only oracle reachability."""
import torch

def select_confidence(confidence,margin):
    if confidence.ndim!=2 or confidence.shape[1]<2 or margin<0:raise ValueError('Parent and at least one candidate, nonnegative margin required')
    value,index=confidence[:,1:].max(1);return torch.where(value>confidence[:,0]+margin,index+1,torch.zeros_like(index))

def oracle_reachable(predictions,clean_pseudo):
    if predictions.ndim!=2 or clean_pseudo.shape!=(len(predictions),):raise ValueError('Matched per-record pseudo targets required')
    return (predictions==clean_pseudo[:,None]).any(1)
