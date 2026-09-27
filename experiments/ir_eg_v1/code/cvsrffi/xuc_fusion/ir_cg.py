"""Bounded CG in whitened active coordinates, using FP64 scalar products."""
import math
import torch
from .ir_types import CGResult

def cg_quadratic_energy(e,metric,ggn_operator,solution):
    """Return the local whitened model energy, not a classification-risk bound.

    solution is active-coordinate u. The supplied GGN operator consumes and
    returns full phi coordinates, exactly as in solve_response.
    """
    idx=metric.active_indices;root=metric.sqrt_diagonal
    if e.ndim!=1 or solution.shape!=root.shape or idx.shape!=root.shape:
        raise ValueError('quadratic energy active layout mismatch')
    if not torch.isfinite(e).all() or not torch.isfinite(solution).all():
        raise ValueError('nonfinite quadratic energy input')
    if idx.numel()==0:return 0.
    if idx.dtype!=torch.long or idx.unique().numel()!=idx.numel() or idx.min()<0 or idx.max()>=e.numel():
        raise ValueError('quadratic energy active indices invalid')
    if idx.device!=e.device or root.device!=e.device or solution.device!=e.device:
        raise ValueError('quadratic energy device mismatch')
    if not torch.isfinite(root).all() or (root<=0).any():raise ValueError('invalid quadratic energy metric')
    u=solution.detach();full=torch.zeros_like(e);full[idx]=root*u
    gv=ggn_operator(full)
    if gv.shape!=e.shape:raise ValueError('GGN changed full parameter layout')
    au=u+root*gv.detach()[idx];b=-root*e.detach()[idx]
    energy=.5*torch.dot(u.double(),au.double())-torch.dot(b.double(),u.double())
    if not torch.isfinite(energy):raise ValueError('nonfinite quadratic energy')
    return float(energy)

def solve_response(e,metric,ggn_operator,*,max_iterations=2,rtol=1e-3):
    if type(max_iterations) is not int or not 1<=max_iterations<=2:
        raise ValueError('IR v1 requires one or two CG iterations')
    if isinstance(rtol,bool) or not math.isfinite(rtol) or rtol<=0:raise ValueError('invalid CG tolerance')
    idx=metric.active_indices;root=metric.sqrt_diagonal
    if e.ndim!=1 or idx.ndim!=1 or root.shape!=idx.shape or metric.diagonal.shape!=root.shape:
        raise ValueError('CG active layout mismatch')
    if idx.dtype!=torch.long or idx.unique().numel()!=idx.numel() or (idx.numel() and (idx.min()<0 or idx.max()>=e.numel())):
        raise ValueError('CG active indices invalid')
    if idx.numel()==0:
        return CGResult(e.new_empty(0),'zero_rhs',0,0.,None)
    if idx.device!=e.device or root.device!=e.device:raise ValueError('CG active device mismatch')
    zero=torch.zeros_like(root)
    def failed(reason,iterations=0,residual=float('inf')):
        return CGResult(zero.clone(),'numerical_failure',iterations,residual,reason)
    if not torch.isfinite(e).all() or not torch.isfinite(root).all() or not torch.isfinite(metric.diagonal).all():return failed('nonfinite input')
    if (root<=0).any() or (metric.diagonal<=0).any():return failed('nonpositive active metric')
    if not torch.allclose(root.square(),metric.diagonal,rtol=2e-6,atol=0.):raise ValueError('inconsistent square-root metric')
    b=-root*e.detach()[idx]
    dot=lambda a,b:torch.dot(a.double(),b.double())
    bb=dot(b,b)
    if float(bb)==0:return CGResult(zero,'zero_rhs',0,0.,None)
    def operator(v):
        full=torch.zeros_like(e);full[idx]=root*v
        gv=ggn_operator(full)
        if gv.shape!=e.shape:raise ValueError('GGN changed full parameter layout')
        return v+root*gv[idx]
    u=zero.clone();r=b.clone();p=r.clone();rr=bb
    for iteration in range(1,max_iterations+1):
        ap=operator(p)
        denominator=dot(p,ap)
        if not torch.isfinite(ap).all() or not torch.isfinite(denominator):return failed('nonfinite operator',iteration-1)
        if float(denominator)<=0:return failed('nonpositive search denominator',iteration-1)
        alpha=(rr/denominator).to(u.dtype);u=u+alpha*p;r=r-alpha*ap
        rr_next=dot(r,r);relative=float(torch.sqrt(rr_next/bb))
        if not torch.isfinite(u).all() or not math.isfinite(relative):return failed('nonfinite iterate',iteration)
        if relative<=rtol:return CGResult(u.detach(),'early_converged',iteration,relative,None)
        if iteration==max_iterations:return CGResult(u.detach(),'truncated_usable',iteration,relative,None)
        p=r+(rr_next/rr).to(p.dtype)*p;rr=rr_next
