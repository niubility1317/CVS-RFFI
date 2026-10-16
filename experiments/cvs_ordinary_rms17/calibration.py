"""Restore original packet RMS using one positive scalar,with no random draw."""
import math
import torch
from . import design as d

def restore(raw,ordinary):
 if not isinstance(raw,torch.Tensor) or not isinstance(ordinary,torch.Tensor) or raw.ndim!=3 or raw.shape[1:]!=(2,256) or not len(raw) or ordinary.shape!=raw.shape or raw.dtype not in (torch.float32,torch.float64) or ordinary.dtype!=raw.dtype or ordinary.device!=raw.device or not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(ordinary).all()):raise ValueError('Matching finite real float32/64 IQ[B,2,256] required')
 px=raw.double().square().sum((1,2));po=ordinary.double().square().sum((1,2))
 if bool((px<=0).any()) or bool((po<=0).any()):raise ValueError('Nonzero packetpower required,no silent filtering')
 gain=torch.sqrt(px/po);exact=ordinary.double()*gain[:,None,None];out=exact.to(ordinary.dtype);pn=out.double().square().sum((1,2))
 if not bool(torch.isfinite(out).all()) or not bool((gain>0).all()):raise ValueError('Finite positive scale required')
 stats=dict(gain=gain.detach().tolist(),raw_RMS=torch.sqrt(px/256).detach().tolist(),ordinary_RMS=torch.sqrt(po/256).detach().tolist(),restored_RMS=torch.sqrt(pn/256).detach().tolist(),relative_power_error=(pn/px-1).abs().detach().tolist(),positive_scalar_shape_error=torch.sqrt((out.double()-exact).square().sum((1,2))/px).detach().tolist(),relative_IQ_change_from_ordinary=torch.sqrt((out.double()-ordinary.double()).square().sum((1,2))/po).detach().tolist())
 return out,stats

def qualify(raw,ordinary,restored,draw_gains,stats):
 c=d.config()['criteria'];keys=('gain','relative_power_error','positive_scalar_shape_error')
 if len(draw_gains)!=4 or any(k not in stats or not stats[k] for k in keys) or len({len(stats[k]) for k in keys})!=1 or any(not math.isfinite(v) for k in keys for v in stats[k]):raise ValueError('Complete finite fourdraw calibration statistics required')
 relative=100*(restored['accuracy']-ordinary['accuracy']);f1=100*(restored['macro_F1']-ordinary['macro_F1']);rawgain=100*(restored['accuracy']-raw['accuracy']);rawf1=100*(restored['macro_F1']-raw['macro_F1']);fraction=sum(abs(v-1)>.01 for v in stats['gain'])/len(stats['gain'])
 conditions=dict(repair_ordinary_accuracy=relative>=c['min_RMS_vs_ordinary_accuracy_gain_pp']-1e-12,repair_ordinary_macroF1=f1>=c['min_RMS_vs_ordinary_macroF1_gain_pp']-1e-12,positive_draws=sum(v>0 for v in draw_gains)>=c['min_positive_draws'],every_draw_protected=min(draw_gains)>=c['min_each_draw_RMS_vs_ordinary_gain_pp']-1e-12,raw_accuracy=rawgain>=c['min_RMS_vs_raw_accuracy_gain_pp']-1e-12,raw_macroF1=rawf1>=c['min_RMS_vs_raw_macroF1_gain_pp']-1e-12,raw_correct_retention=restored['raw_correct_retention']>=c['min_raw_correct_retention'],nontrivial_scale_fraction=fraction>=c['min_fraction_gain_outside_1pct'],packetpower_restored=max(stats['relative_power_error'])<=c['max_relative_power_error'],positive_scalar_shape_preserved=max(stats['positive_scalar_shape_error'])<=c['max_positive_scalar_shape_error'])
 for key in ('RX','TX','day'):
  conditions['every_'+key+'_vs_raw']=all(100*(z['accuracy']-raw['by_'+key][value]['accuracy'])>=c['min_each_group_RMS_vs_raw_gain_pp']-1e-12 for value,z in restored['by_'+key].items());conditions['every_'+key+'_vs_ordinary']=all(100*(z['accuracy']-ordinary['by_'+key][value]['accuracy'])>=c['min_each_group_RMS_vs_ordinary_gain_pp']-1e-12 for value,z in restored['by_'+key].items())
 return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions,RMS_vs_ordinary_accuracy_gain_pp=relative,RMS_vs_ordinary_macroF1_gain_pp=f1,RMS_vs_raw_accuracy_gain_pp=rawgain,RMS_vs_raw_macroF1_gain_pp=rawf1,raw_correct_retention=restored['raw_correct_retention'],draw_accuracy_gains_pp=draw_gains,positive_draws=sum(v>0 for v in draw_gains),fraction_gain_outside_1pct=fraction,max_relative_power_error=max(stats['relative_power_error']),max_positive_scalar_shape_error=max(stats['positive_scalar_shape_error']))
