"""Exact SAM temporary perturbation and one permanent AdamW update."""
import random,time
import numpy as np
import torch

def rng_state():
    return dict(torch=torch.get_rng_state().clone(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],python=random.getstate(),numpy=np.random.get_state())

def set_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python']);np.random.set_state(state['numpy'])

def take_update(c,model,optimizer,closure,audit=None):
    start=rng_state();before_buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    optimizer.zero_grad(set_to_none=True)
    loss,parts,weighted,detail=closure()
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite first engineering loss')
    first_audit=audit(weighted) if audit else None
    loss.backward()
    params=[p for group in optimizer.param_groups for p in group['params']]
    first_norm=torch.nn.utils.clip_grad_norm_(params,float('inf'),error_if_nonfinite=True)
    record=dict(sam_applied=False,temporary_parameter_restoration=True,identity_optimizer_steps=1,
        first_gradient_norm=float(first_norm),second_gradient_norm=None,second_forward_calls=0,second_backward_calls=0,seconds=0.)
    if c['engineering']['sam']:
        tick=time.perf_counter();eligible=[p for p in params if p.grad is not None]
        if not eligible:raise ValueError('SAM has no identity gradients')
        saved=[p.detach().clone() for p in eligible];end=rng_state();end_buffers={n:b.detach().clone() for n,b in model.named_buffers()}
        rho=c['engineering']['sam_rho'];norm=first_norm.clamp_min(1e-12)
        try:
            with torch.no_grad():
                for p in eligible:p.add_(p.grad,alpha=float(rho/norm))
                perturb=torch.stack([(p-v).norm() for p,v in zip(eligible,saved)]).norm()
                for name,b in model.named_buffers():
                    if not torch.equal(b,before_buffers[name]):b.copy_(before_buffers[name])
            optimizer.zero_grad(set_to_none=True);set_rng(start)
            second,_,_,second_detail=closure()
            if not bool(torch.isfinite(second)):raise FloatingPointError('Nonfinite perturbed engineering loss')
            second.backward();second_norm=torch.nn.utils.clip_grad_norm_(params,float('inf'),error_if_nonfinite=True)
            after=rng_state()
            if not torch.equal(after['torch'],end['torch']) or len(after['cuda'])!=len(end['cuda']) or any(not torch.equal(a,b) for a,b in zip(after['cuda'],end['cuda'])):
                raise ValueError('SAM same-shape closure consumed different model RNG')
            record.update(sam_applied=True,rho=rho,actual_perturbation_l2=float(perturb),second_loss=float(second.detach()),
                perturbed_parameter_tensors=len(eligible),perturbed_parameters=sum(p.numel() for p in eligible),
                missing_gradient_parameter_tensors=len(params)-len(eligible),second_gradient_norm=float(second_norm),
                second_forward_calls=1+2*int(second_detail['U_packets']>0),second_backward_calls=1,
                same_samples_and_noise=True,same_dropout_rng=True,second_targets_recomputed=False)
        finally:
            with torch.no_grad():
                for p,value in zip(eligible,saved):p.copy_(value)
                for name,b in model.named_buffers():
                    if not torch.equal(b,end_buffers[name]):b.copy_(end_buffers[name])
            set_rng(end)
        if any(not torch.equal(p.detach(),v) for p,v in zip(eligible,saved)):raise AssertionError('Temporary SAM state was not restored')
        record['seconds']=time.perf_counter()-tick
    optimizer.step()
    final_norm=record['second_gradient_norm'] if record['sam_applied'] else record['first_gradient_norm']
    return loss,parts,weighted,detail,first_audit,final_norm,record
