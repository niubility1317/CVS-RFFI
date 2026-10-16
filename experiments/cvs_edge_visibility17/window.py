"""One nonlearned real IQ edge window;no shift,noise,channel or label input."""
import math,statistics
import torch
from . import design as d

def apply(x,side):
 if side not in ('left','right'):raise ValueError('Fixed left/right edge only')
 if not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[1:]!=(2,256) or not len(x) or x.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(x).all()):raise ValueError('Finite nonempty real float32/64 IQ[B,2,256] required')
 power=x.square().sum((1,2))
 if bool((power<=0).any()):raise ValueError('Nonzero IQ required;no silent packet filtering')
 w=torch.ones(256,dtype=x.dtype,device=x.device);w[:8]=0;w[8:16]=.5*(1-torch.cos(torch.arange(1,9,dtype=x.dtype,device=x.device)*(math.pi/8)))
 if side=='right':w=w.flip(0)
 y=x*w[None,None,:]
 outside=(slice(16,256) if side=='left' else slice(0,240))
 info=dict(relative_IQ_change=torch.sqrt((y-x).square().sum((1,2))/power).detach().tolist(),power_increase=((y.square().sum((1,2))-power)/power).detach().tolist(),reference_unchanged=(y[:,:,80:160]==x[:,:,80:160]).flatten(1).all(1).tolist(),outside_support_unchanged=(y[:,:,outside]==x[:,:,outside]).flatten(1).all(1).tolist())
 return y,info

def qualify(raw,view,stats):
 c=d.config()['criteria'];keys=('relative_IQ_change','power_increase','reference_unchanged','outside_support_unchanged')
 if any(k not in stats for k in keys) or not stats[keys[0]] or len({len(stats[k]) for k in keys})!=1 or any(not math.isfinite(v) for k in keys[:2] for v in stats[k]):raise ValueError('Complete finite edge statistics required')
 conditions=dict(overall_accuracy=100*(view['accuracy']-raw['accuracy'])>=c['min_overall_accuracy_gain_pp']-1e-12,overall_macroF1=100*(view['macro_F1']-raw['macro_F1'])>=c['min_overall_macroF1_gain_pp']-1e-12,raw_correct_retention=view['raw_correct_retention']>=c['min_raw_correct_retention'],nonzero_IQ_change=statistics.median(stats['relative_IQ_change'])>=c['min_median_relative_IQ_change'],bounded_IQ_change=max(stats['relative_IQ_change'])<=c['max_relative_IQ_change'],no_power_increase=max(stats['power_increase'])<=c['max_power_increase'],reference_unchanged=all(stats['reference_unchanged']),outside_support_unchanged=all(stats['outside_support_unchanged']))
 for key,allowed in (('RX',d.FIT),('TX',range(6)),('day',(1,2,3))):conditions['every_'+key]=all(100*(view['by_'+key][str(v)]['accuracy']-raw['by_'+key][str(v)]['accuracy'])>=c['min_each_'+key+'_accuracy_gain_pp']-1e-12 for v in allowed)
 return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions,accuracy_gain_pp=100*(view['accuracy']-raw['accuracy']),macroF1_gain_pp=100*(view['macro_F1']-raw['macro_F1']),raw_correct_retention=view['raw_correct_retention'],relative_IQ_change_min=min(stats['relative_IQ_change']),relative_IQ_change_median=statistics.median(stats['relative_IQ_change']),relative_IQ_change_max=max(stats['relative_IQ_change']),max_power_increase=max(stats['power_increase']))
