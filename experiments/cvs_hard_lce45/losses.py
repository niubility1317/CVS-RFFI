"""Replace CE only within the L hook; preserve original KL and restore on exit."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

@contextmanager
def hard_L_terms(module,records):
 original=module.terms
 def changed(first,second,y,alpha,smoothing=.01):
  # Original function supplies all existing shape/finite/label/alpha checks.
  original_total,old_ce,kl,_,_=original(first,second,y,alpha,smoothing)
  ce1=F.cross_entropy(first,y,label_smoothing=0.)
  ce2=F.cross_entropy(second,y,label_smoothing=0.)
  ce=.5*(ce1+ce2)
  records.append(dict(CE0=float(ce.detach()),CE01=float(old_ce.detach()),KL=float(kl.detach()),L_smoothing=0.,alpha=float(alpha),NN=0))
  return ce+alpha*kl,ce,kl,ce1,ce2
 module.terms=changed
 try:yield
 finally:module.terms=original
