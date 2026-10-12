"""Synthetic truth-last close of all declared views and three primary rows."""
import argparse,json
from pathlib import Path
import numpy as np
from . import design as d
from . import predict,score
from experiments.cvs_scratch30_baseline import design as b

def run(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=False);saved=(d.BASE,d.TRUTH,b.BASE,predict.source_proof,predict.observation_metadata)
 try:
  d.BASE=root;b.BASE=root/'baseline';d.TRUTH=str(root/'fixture_truth.json')
  ids=np.asarray([f'synthetic-final-ensemble-{i}' for i in range(168000)])
  from experiments.cvs_report_ssl.evaluate import HELDOUT_RX,TARGET_DAYS,metric_rows
  y=np.repeat(np.arange(6,dtype=np.int64),28000);rx=np.tile(np.repeat(np.asarray(HELDOUT_RX),4000),6);days=np.tile(np.repeat(np.asarray(TARGET_DAYS),1000),42)
  predict.source_proof=lambda c:None;predict.observation_metadata=lambda:(ids,{})
  for row in d.rows():
   c=d.config(row);p=root/row['row_id']/'prediction';p.mkdir(parents=True)
   d.write(p/'provenance.json',dict(status='VERIFIED',config=c,checkpoint=str(Path(c['source_output'])/'identity.pth'),own_final30=True,ancestors=[],truth_read=False,query_fit=False,source_roles='EXACT_MATCH'))
   d.write(p/'complete.json',dict(status='PREDICTIONS_COMPLETE',config=c,count=168000,views=list(d.VIEWS),truth_read=False,query_fit=False));np.savez_compressed(p/'predictions.npz',ids=ids,**{v:y for v in d.VIEWS})
   bc=d.baseline_config(row['model_seed']);bp=b.BASE/bc['row_id'];d.write(bp/'completion.json',dict(config=bc,status='TRAINED_PREDICTED_SCORED',views=list(d.VIEWS),count_per_view=168000,truth_last=True));records=[]
   for view in d.VIEWS:
    rr,dd=metric_rows(bc,y,ids,y,rx,days)
    for r in rr+dd:r['view']=view
    records.extend(rr+dd)
   d.write(bp/'scores/metrics.json',dict(records=records,truth_last=True,independent_recount='VERIFIED'))
  row=d.rows()[0]['row_id'];p=root/row/'prediction/predictions.npz';np.savez_compressed(p,ids=ids,**{v:y for v in d.VIEWS[:-1]})
  try:score.score(row)
  except ValueError:pass
  else:raise AssertionError('Missing view opened truth')
  if (root/row/'scores').exists():raise AssertionError('Scoring began before all views closed')
  np.savez_compressed(p,ids=ids,**{v:y for v in d.VIEWS})
  truth={str(uid):dict(label=int(label),receiver=str(receiver),day=str(day)) for uid,label,receiver,day in zip(ids,y,rx,days)};d.write(Path(d.TRUTH),truth)
  for row in d.rows():score.score(row['row_id'])
  result=score.finish()
  if len(result['records'])!=378 or result['prediction_count']!=3528000 or not result['goal_84']:raise AssertionError('Synthetic final matrix close differs')
  return dict(status='PASS',synthetic_only=True,rows=3,views=7,records=378,predictions=3528000,truth_closed_on_missing_view=True)
 finally:d.BASE,d.TRUTH,b.BASE,predict.source_proof,predict.observation_metadata=saved
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.root);d.write(a.output,v);print(json.dumps(v))
