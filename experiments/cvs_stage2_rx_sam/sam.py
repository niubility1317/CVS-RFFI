"""One SAM second backward; caller supplies first backward and base optimizer step."""
import torch

def rng_state():
 return (torch.get_rng_state().clone(),torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)

def set_rng(state):
 torch.set_rng_state(state[0])
 if state[1] is not None:torch.cuda.set_rng_state_all(state[1])

def buffers(model):return {n:b.detach().clone() for n,b in model.named_buffers()}

@torch.no_grad()
def restore_buffers(model,state):
 current=dict(model.named_buffers())
 if current.keys()!=state.keys():raise ValueError('Model buffer names changed during SAM')
 for n,b in current.items():b.copy_(state[n])

def sam_backward(model,optimizer,rho,before_buffers,closure):
 """Auxiliary gradients stay intact; optimizer contains identity parameters only."""
 eligible=[(n,p) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None and not n.endswith(('low_hz_','band_hz_'))]
 if not eligible:raise ValueError('No SAM identity gradient')
 norm=torch.stack([p.grad.detach().norm() for _,p in eligible]).norm()
 if not torch.isfinite(norm):raise FloatingPointError('Nonfinite SAM first gradient')
 original=[(p,p.detach().clone()) for _,p in eligible]
 end_buffers=buffers(model);end_rng=rng_state()
 try:
  with torch.no_grad():
   for _,p in eligible:p.add_(p.grad,alpha=float(rho/norm.clamp_min(1e-12)))
   actual=torch.stack([(p-v).norm() for p,v in original]).norm()
  optimizer.zero_grad(set_to_none=True)
  restore_buffers(model,before_buffers)
  loss=closure()
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite SAM perturbed loss')
  loss.backward()
  result=dict(rho=rho,first_dense_gradient_norm=float(norm),actual_perturbation_l2=float(actual),perturbed_identity_loss=float(loss.detach()),identity_parameters_perturbed=sum(p.numel() for _,p in eligible),auxiliary_recomputed=False,targets_recomputed=False)
 finally:
  with torch.no_grad():
   for p,value in original:p.copy_(value)
  restore_buffers(model,end_buffers);set_rng(end_rng)
 return result
