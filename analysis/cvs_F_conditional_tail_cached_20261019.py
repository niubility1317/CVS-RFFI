"""Cached source-only two-draw error stability; no fitting or model calls."""
import argparse,json
from pathlib import Path
import numpy as np

def run(source):
 with np.load(source,allow_pickle=False) as f:
  y=f['TX'];rx=f['RX'];day=f['day'];rows={}
  if y.shape!=(360,) or set(y.tolist())!=set(range(6)) or set(rx.tolist())!={1,3,4,6,8} or set(day.tolist())!={1,2,3}:raise ValueError('Existing fixed source V360 provenance/shape differs')
  for scene in ('mid','low'):
   z=np.stack([f[f'd{i}_{scene}_student'] for i in (0,1)])
   if z.shape!=(2,360,6) or not np.isfinite(z).all():raise ValueError('Complete source draw logits required')
   logp=z-z.max(-1,keepdims=True);logp-=np.log(np.exp(logp).sum(-1,keepdims=True));ce=-.99*logp[:,np.arange(360),y]-.01*logp.mean(-1);ok=z.argmax(-1)==y
   rows[scene]=dict(packets=360,draw_accuracy=ok.mean(1).tolist(),both_correct=int(ok.all(0).sum()),one_correct=int((ok.sum(0)==1).sum()),both_wrong=int((~ok).all(0).sum()),argmax_disagree=int((z[0].argmax(-1)!=z[1].argmax(-1)).sum()),mean_absolute_pair_CE_gap=float(abs(ce[0]-ce[1]).mean()),uniform_mean_CE=float(ce.mean()),half_mean_half_max_CE=float(.5*ce.mean()+.5*ce.max(0).mean()))
 return dict(status='SOURCE_CACHED_ANALYZED',origin_run='20261019-diagnostic-F-CSIL-inputJacobian-s2026092701-m1-r01',student='original pure F B1 FINAL20',role='previously declared source V360 diagnostic readout only',rows=rows,model_seed_count=1,NN=0,new_IQ=0,parameter_or_threshold_fits=0,target_access=False,claim='Observed conditional draw variability, not proof conditional-tail training will help; coefficients fixed before this recount')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();x=run(a.source);Path(a.output).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(x,ensure_ascii=False))
