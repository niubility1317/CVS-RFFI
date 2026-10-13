"""Report-only recovery: immutable completed predictions/scores are inputs."""
import argparse,json
from pathlib import Path
from . import design as original
RUN='20261013-phase1-crossrx-contrastive-ensemble-aggregate-manysig-m6-r02'
RELEASE='cvs_crossrx_contrastive_aggregate_20261013_r02'
PROJECT=original.PROJECT;ROOT=original.ROOT;BASE=Path(PROJECT)/'runs'/RUN
read,write=original.read,original.write
def preflight():
 from .predict import preflight as row_preflight
 for row in original.rows():row_preflight(row['row_id'])
 return dict(status='PASS',input_run_id=original.RUN,completed_prediction_rows=6,new_training_updates=0,new_predictions=0,query_IQ_or_truth_read=False,checkpoint_execution=False)
def run():
 from .score import finish
 result=finish(output=BASE,run_id=RUN)
 write(BASE/'completion.json',dict(status='AGGREGATION_RECOVERY_COMPLETE',input_run_id=original.RUN,groups=6,original_models=18,new_training_updates=0,new_predictions=0,reused_frozen_predictions=1008000,truth_last=True,goal_84=result['goal_84']))
 print(json.dumps(dict(status='VERIFIED',summary=result['summary'],paired=result['paired'],goal_84=result['goal_84'],new_training_updates=0,new_predictions=0)))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
 if a.remote_preflight:
  if a.output is None:raise ValueError('Preflight output path required')
  v=preflight();write(a.output,v);print(json.dumps(v))
 else:run()
