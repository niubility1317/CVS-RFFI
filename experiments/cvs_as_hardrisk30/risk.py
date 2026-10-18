"""Bounded within-TX CE risk;all IQ/labels/NN/KL remain original."""
import time

def packet_weights(scores,labels,mode,step):
 import torch
 if mode not in ('HARD','SHUFFLE') or step not in range(1,2221) or scores.shape!=(64,) or labels.shape!=(64,) or labels.dtype!=torch.long or not torch.isfinite(scores).all() or not bool(((labels>=0)&(labels<6)).all()):raise ValueError('Visible As64 and fixed risk mode/step required')
 scores=scores.detach();weights=torch.ones_like(scores);groups=[];g=torch.Generator(device='cpu').manual_seed(20261019091+step)
 for cls in range(6):
  idx=(labels==cls).nonzero(as_tuple=False).flatten();n=len(idx)
  if not n:continue
  k=(n+1)//2;order=torch.argsort(scores[idx],descending=True,stable=True);local=torch.full((n,),.75,dtype=scores.dtype,device=scores.device);local[order[:k]]+=.25*n/k
  if mode=='SHUFFLE':local=local[torch.randperm(n,generator=g).to(local.device)]
  weights[idx]=local;groups.append(dict(TX=cls,count=n,top_count=k,weight_sum=float(local.sum()),weight_min=float(local.min()),weight_max=float(local.max())))
 return weights.detach(),groups

def correction(first,second,labels,mask,mode,step,enabled=True,old_CE_means=None):
 start=time.perf_counter()
 import torch
 from torch.nn import functional as F
 if first.shape!=second.shape or tuple(first.shape)!=(128,6) or labels.shape!=(128,) or labels.dtype!=torch.long or mask.shape!=(128,) or mask.dtype!=torch.bool or int(mask.sum())!=64 or not bool(((labels>=0)&(labels<6)).all()) or not torch.isfinite(first).all() or not torch.isfinite(second).all() or not enabled:raise ValueError('Original finite L128/mixedAs64 required')
 if not isinstance(old_CE_means,(tuple,list)) or len(old_CE_means)!=2 or any(v.ndim!=0 or not bool(torch.isfinite(v)) for v in old_CE_means):raise ValueError('Reuse original two mixed CE means')
 losses=[F.cross_entropy(z[mask],labels[mask],reduction='none',label_smoothing=.01) for z in (first,second)]
 scores=.5*(losses[0].detach()+losses[1].detach());weights,groups=packet_weights(scores,labels[mask],mode,step);changes=[((weights-1)*ce).sum()/128 for ce in losses];delta=.25*(changes[0]+changes[1]);operator_seconds=time.perf_counter()-start;scalar_start=time.perf_counter()
 old_s=[float(v.mean().detach()) for v in losses];new_s=[float((weights*v).mean().detach()) for v in losses];ordinary=[2*float(v.detach())-o for v,o in zip(old_CE_means,old_s)];mixed_old=.5*sum(float(v.detach()) for v in old_CE_means);mixed_delta=.5*float((changes[0]+changes[1]).detach())
 rec=dict(mode=mode,selected_packets=64,native_label_smoothing=.01,old_epsilon=.01,selected_CE_old=.5*sum(old_s),selected_CE_new=.5*sum(new_s),selected_CE_old_pass0=old_s[0],selected_CE_old_pass1=old_s[1],selected_CE_new_pass0=new_s[0],selected_CE_new_pass1=new_s[1],ordinary_CE_pass0=ordinary[0],ordinary_CE_pass1=ordinary[1],mixed_CE_old=mixed_old,mixed_CE_new=mixed_old+mixed_delta,mixed_CE_delta=mixed_delta,loss_delta=float(delta.detach()),extra_NN=0,extra_CE_operator_calls=2,extra_CE_logit_packets=128,baseline_mixed_CE_operator_widths=[128,128],extra_CE_operator_widths=[64,64],actual_mixed_CE_operator_widths=[128,128,64,64],CE_operator_cpu_wall_seconds=operator_seconds,scalarization_cpu_wall_seconds=time.perf_counter()-scalar_start,explicit_CUDA_synchronize=False,all_six_classes_same_formula=True,confidence_filter=False,weight_min=float(weights.min()),weight_max=float(weights.max()),weight_sum=float(weights.sum()),class_groups=groups,ranking_gradient_stopped=not weights.requires_grad,ordinary_CE_changed=False,KL_changed=False,global_RNG_changed=False)
 return delta,rec
