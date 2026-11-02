"""Cross-time agreement guard for source-only blind action diagnosis."""
import torch
from channel_refinement import fir_design,fit_bounded_delta

def agreement_metrics(received,true_forward,parent_forward):
 if received.shape!=true_forward.shape or received.shape!=parent_forward.shape or received.ndim!=2 or received.shape[1]!=256:raise ValueError('Matching complex source signals required')
 gt=fir_design(true_forward);gp=fir_design(parent_forward);time=torch.arange(256,device=received.device);den=received.abs().square().sum(1).clamp_min(1e-8);rows=[]
 for hold in range(3):
  test=(time%3)==hold;train1=(time%3)==((hold+1)%3);train2=(time%3)==((hold+2)%3);res=received-parent_forward;g1=fit_bounded_delta(gp,res,train1);g2=fit_bounded_delta(gp,res,train2);norm1=torch.linalg.vector_norm(g1,dim=1);norm2=torch.linalg.vector_norm(g2,dim=1);cos=(g1.conj()*g2).sum(1).real/(norm1*norm2).clamp_min(1e-12);g=(g1+g2)*.5;delta=torch.bmm(gt,g.unsqueeze(-1)).squeeze(-1);r=received-true_forward;base=r[:,test].abs().square().sum(1)/den;full=(r-delta)[:,test].abs().square().sum(1)/den;columns=[base,full,cos/3]
  for threshold in (.3,.6,.9):
   active=cos>threshold;guarded=r-delta*active[:,None];err=guarded[:,test].abs().square().sum(1)/den;columns.extend((active.float()/3,err))
  rows.append(torch.stack(columns,1))
 out=rows[0]+rows[1]+rows[2]
 if not torch.isfinite(out).all():raise ValueError('Finite channel agreement diagnosis required')
 return out
