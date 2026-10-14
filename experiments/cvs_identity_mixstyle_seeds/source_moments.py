"""Frozen full-L observer for four new models; no head fitting or target read."""
import json
from . import design as d
from experiments.cvs_identity_mixstyle50.source_moments import run as shared_run,LAYERS
def complete():
 out=d.BASE/'source_style_mechanism.json'
 if out.exists():raise FileExistsError('Existing immutable result; reuse it')
 rows=[r for r in d.rows() if r['arm']!=d.BASELINE_ARM]
 results=[shared_run(d.config(r),design=d) for r in rows]
 comparisons=[]
 for seed in d.SEEDS:
  control=next(v for v in results if v['model_seed']==seed and v['arm']=='MATCHED_CONTROL')
  candidate=next(v for v in results if v['model_seed']==seed and v['arm']==d.PRIMARY)
  comparisons.append(dict(seed=seed,layers={n:dict(RX_conditional_fraction_difference=candidate['moments'][n]['RX_within_TX_day_fraction']-control['moments'][n]['RX_within_TX_day_fraction'],TX_conditional_fraction_difference=candidate['moments'][n]['TX_within_RX_day_fraction']-control['moments'][n]['TX_within_RX_day_fraction'],RX_conditional_absolute_SS_difference=candidate['moments'][n]['RX_within_TX_day_SS']-control['moments'][n]['RX_within_TX_day_SS']) for n in LAYERS}))
 result=dict(status='VERIFIED',run_id=d.RUN,models=len(results),source_packets_per_model=6300,source_batches_per_model=50,packet_forwards=sum(v['packet_forwards'] for v in results),rows=results,comparisons_to_matched_control=comparisons,head_fit=False,head_updates=0,optimizer_steps=0,query_read=False,U_truth_read=False,U_IQ_read=False,V_cache_materialized=True,V_forward=False,scope='Frozen post-training source observer only; no parameter change, selection or target feedback; descriptive moments are not causal invariance or transfer proof.')
 d.write(out,result);return result
if __name__=='__main__':print(json.dumps(complete()))
