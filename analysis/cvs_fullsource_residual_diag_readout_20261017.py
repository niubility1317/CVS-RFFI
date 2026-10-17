"""Readonly source-diagnostic readout; no IQ/checkpoint/model/target access."""
import json,sys
from pathlib import Path
import numpy as np
O=Path(sys.argv[1])
p=dict(np.load(O/'observations/physics.npz',allow_pickle=False));summary=json.loads((O/'summary.json').read_text());arms=['B00','B01','RAW_PRAC'];scenes=['practical_low_suburban','practical_mid_urban','practical_low_urban']
def quant(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return dict(n=int(x.size),min=float(x.min()),q01=float(np.quantile(x,.01)),median=float(np.median(x)),q99=float(np.quantile(x,.99)),max=float(x.max())) if x.size else None
out=dict(source_only=True,NN=0,updates=0,target_access=False,physical={},branches={},gates={},metric_recount={})
for si,s in enumerate(scenes):
 z={k:quant(p[k][:,si,0]) for k in ['quality_snr_db','waveform_minus_white_power_db','signal_to_added_noise_db','output_frequency_hz','noise_coloration_dynamic_range_db','agc_gain_db','residual_kernel_energy']}
 z.update(agc_limited=int(p['agc_limited'][:,si,0].sum()),kernel_lengths=np.unique(p['kernel_length'][:,si,0]).tolist(),numerical_delay=np.unique(p['numerical_delay'][:,si,0]).tolist(),interior_count=np.unique(p['interior_count'][:,si,0]).tolist(),actual_power_lock_changes=int((p['lock'][:,si,9]!=p['lock'][:,si,0]).sum()),n=3600)
 out['physical'][s]=z
for a in arms:
 pred=dict(np.load(O/a/'predictions.npz',allow_pickle=False));n=pred['TX'].size;truth=np.broadcast_to(pred['TX'],(2,n));verified=0
 for si,s in enumerate(scenes):
  for ci,c in enumerate(['full','no_noise/fixed','no_noise/total','flat_H/fixed','flat_H/total','no_carrier/fixed','no_carrier/total','no_RX_IQ/fixed','no_RX_IQ/total','actual_power_quality','edge_boundary']):
   pp=pred['satellite'][:,si,ci].argmax(-1);cm=np.bincount((truth*6+pp).ravel(),minlength=36).reshape(6,6);r=summary['metrics'][a][s+'/'+c]
   assert cm.tolist()==r['confusion'] and int(np.trace(cm))==r['correct'];verified+=1
 out['metric_recount'][a]=dict(overall_cells=verified,source_prediction_decisions=2*3*11*n,unique_sourceV=n)
 b=np.load(O/a/'branch_metrics.npz',allow_pickle=False);values=b['values'];out['branches'][a]={s:{name:{metric:float(values[:,si,:,ni,mi].mean()) for mi,metric in enumerate(b['metrics'].tolist())} for ni,name in enumerate(b['names'].tolist())} for si,s in enumerate(scenes)}
 r=json.loads((O/a/'result.json').read_text());out['gates'][a]={k:quant(v) for k,v in r['gates'].items() if isinstance(v,list)}
print(json.dumps(out,indent=2))
