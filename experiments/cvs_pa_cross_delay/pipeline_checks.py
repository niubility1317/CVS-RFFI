import ast,copy,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import design as d
from .predict import frozen_metadata,validate_default_freeze
from .score import validate_records,row_completion,finish
from experiments.cvs_report_ssl.evaluate import HELDOUT_RX,TARGET_DAYS

def records_for(c,error_fraction=.1):
 result=[];strata=dict(overall=['ALL'],receiver=list(HELDOUT_RX),transmitter=list(d.CLASSES),day=list(TARGET_DAYS))
 for dimension,values in strata.items():
  n=d.QUERY_COUNT//len(values)
  for stratum in values:
   cm=np.zeros((6,6),dtype=np.int64)
   for i in range(6):
    count=n//6+int(i<n%6);bad=int(count*error_fraction);cm[i,i]=count-bad;cm[i,(i+1)%6]=bad
   denom=cm.sum(0)+cm.sum(1);f1=float(np.divide(2*np.diag(cm),denom,out=np.zeros(6),where=denom>0).mean())
   result.append(dict(row_id=c['row_id'],arm=c['arm'],model_seed=c['model_seed'],stage=c['stage'],view='clean',readout='ORIGINAL_HEAD',dimension=dimension,stratum=stratum,query_count=n,accuracy=float(np.trace(cm)/n),macro_f1=f1,confusion=cm.tolist()))
 return result

def checks():
 c=d.config(d.training_rows()[0]);good=frozen_metadata(c);validate_default_freeze(c,good);negative=0
 for key,badvalue in [('original_default_W_retained',False),('joint_source_training',False),('separate_head_fit',True),('separate_head_updates',500),('backbone_updates_after_freeze',1),('all_model_parameters_frozen',False),('query_fit',True),('truth_read',True),('readouts',['FRESH_COSINE_ALL_L']),('head_scale',10),('head_shape',[8,160])]:
  bad=copy.deepcopy(good);bad[key]=badvalue
  try:validate_default_freeze(c,bad)
  except ValueError:negative+=1
  else:raise AssertionError('Illegal default freeze accepted: '+key)
 records=records_for(c);assert validate_records(c,records)==18
 candidates=[records[:-1],records+[records[0]]]
 for key,value in [('model_seed',1),('row_id','wrong'),('readout','FRESH_COSINE_ALL_L'),('view','unknown'),('accuracy',float('nan')),('macro_f1',0.),('query_count',1),('confusion',[[0]*6 for _ in range(6)])]:
  bad=copy.deepcopy(records);bad[0][key]=value;candidates.append(bad)
 for bad in candidates:
  try:validate_records(c,bad)
  except ValueError:negative+=1
  else:raise AssertionError('Incomplete/invalid default score accepted')
 # All9 matrix rows actually flow through finish, so stale arm names fail here.
 with tempfile.TemporaryDirectory() as temp,patch.object(d,'BASE',Path(temp)):
  for r in d.rows():
   cfg=d.config(r);record=records_for(cfg,dict(LEGACY50=.1,PA_CONV=.12,PA_CROSS_DELAY=.08)[r['arm']]);p=d.BASE/r['row_id'];(p/'scores').mkdir(parents=True)
   d.write(p/'scores/metrics.json',dict(records=record,readouts=['ORIGINAL_HEAD'],count_per_readout=d.QUERY_COUNT,truth_last=True,independent_recount='VERIFIED',default_classifier_only=True,Phase2_claim=False));d.write(p/'completion.json',row_completion(cfg,18))
  result=finish();assert result['models']==9 and result['prediction_count']==1512000 and len(result['records'])==162 and len(result['summary'])==3 and len(result['paired'])==3
  assert result['paired'][0]['mean_difference']<0 and result['paired'][1]['mean_difference']>0 and result['paired'][2]['positive_seeds']==3
  assert finish()==result
 # Structural guard: predictor has no source fitting/alternate classifier imports.
 tree=ast.parse(Path(__file__).with_name('predict.py').read_text())
 for node in ast.walk(tree):
  if isinstance(node,ast.ImportFrom):assert not any(s in (node.module or '') for s in ('source_transfer_probe','source_transfer_features','source_identity_readout.run'))
  if isinstance(node,ast.Call):
   name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
   assert name not in ('fit','Adam','AdamW','SGD','backward','step')
 return dict(status='PASS',synthetic_only=True,real_IQ_or_truth_read=False,negative_cases=negative,default_readouts=['ORIGINAL_HEAD'],head_fit_code_absent=True,default_updates=0,score_records_per_row=18,matrix_rows=9,aggregate_records=162,summary_count=3,paired_count=3,prediction_count=1512000)
if __name__=='__main__':print(json.dumps(checks()))
