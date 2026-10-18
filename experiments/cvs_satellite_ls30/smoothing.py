"""CE target-entropy correction on existing mixed logits;no model call or RNG."""
import time

def correction(first,second,labels,mask,epsilon=.05,enabled=True,old_CE_means=None):
 start=time.perf_counter()
 import torch
 from torch.nn import functional as F
 if first.shape!=second.shape or tuple(first.shape)!=(128,6) or labels.shape!=(128,) or labels.dtype!=torch.long or mask.shape!=(128,) or mask.dtype!=torch.bool or int(mask.sum())!=64 or not bool(((labels>=0)&(labels<6)).all()) or not torch.isfinite(first).all() or not torch.isfinite(second).all() or epsilon not in (.01,.05):raise ValueError('Existing128x6 logits/visiblelabels/original64mask required')
 if not enabled:
  return first.sum()*0,dict(selected_packets=0,epsilon=epsilon,selected_CE_old=0.,selected_CE_new=0.,mixed_CE_delta=0.,loss_delta=0.,extra_NN=0,extra_CE_operator_calls=0,extra_CE_logit_packets=0,CE_operator_cpu_wall_seconds=time.perf_counter()-start,scalarization_cpu_wall_seconds=0.,explicit_CUDA_synchronize=False)
 if not isinstance(old_CE_means,(tuple,list)) or len(old_CE_means)!=2 or any(v.ndim!=0 or not bool(torch.isfinite(v)) for v in old_CE_means):raise ValueError('Reuse both original mixed CE01 scalar means')
 old=[F.cross_entropy(z[mask],labels[mask],reduction='none',label_smoothing=.01) for z in (first,second)]
 new=[F.cross_entropy(z[mask],labels[mask],reduction='none',label_smoothing=epsilon) for z in (first,second)]
 selected_old=old;changes=[(n-o).sum()/128 for n,o in zip(new,selected_old)]
 delta=.25*(changes[0]+changes[1]);operator_seconds=time.perf_counter()-start;scalar_start=time.perf_counter()
 old_s=[float(v.mean().detach()) for v in selected_old];new_s=[float(v.mean().detach()) for v in new];ordinary=[2*float(v.detach())-o for v,o in zip(old_CE_means,old_s)]
 mixed_old=.5*(float(old_CE_means[0].detach())+float(old_CE_means[1].detach()));mixed_delta=.5*float((changes[0]+changes[1]).detach())
 record=dict(selected_packets=64,epsilon=epsilon,old_epsilon=.01,native_label_smoothing=.01,selected_CE_old=.5*sum(old_s),selected_CE_new=.5*sum(new_s),selected_CE_old_pass0=old_s[0],selected_CE_old_pass1=old_s[1],selected_CE_new_pass0=new_s[0],selected_CE_new_pass1=new_s[1],ordinary_CE_pass0=ordinary[0],ordinary_CE_pass1=ordinary[1],mixed_CE_old=mixed_old,mixed_CE_new=mixed_old+mixed_delta,mixed_CE_delta=mixed_delta,loss_delta=float(delta.detach()),extra_NN=0,extra_CE_operator_calls=4,extra_CE_logit_packets=256,baseline_mixed_CE_operator_widths=[128,128],extra_CE_operator_widths=[64,64,64,64],actual_mixed_CE_operator_widths=[128,128,64,64,64,64],CE_operator_cpu_wall_seconds=operator_seconds,scalarization_cpu_wall_seconds=time.perf_counter()-scalar_start,explicit_CUDA_synchronize=False,all_six_classes_same_formula=True,confidence_filter=False)
 return delta,record
