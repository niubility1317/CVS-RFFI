import json,copy,math
from . import design as d
from .predict import validate_head_freeze,checkpoint_path
from .score import validate_records,row_completion
from .model import build,from_phase1
from experiments.cvs_report_ssl.evaluate import HELDOUT_RX,TARGET_DAYS

def checks():
 c=d.config(d.training_rows()[0]);f=from_phase1(build(c));shapes={k:list(v.shape) for k,v in f.named_parameters()}
 freeze=dict(status='SOURCE_READOUTS_FROZEN',config=c,readouts=list(d.READOUTS),feature_schema='actual joint160; row-wise L2 eps1e-4',feature_dimension=160,old_W_discarded=True,feature_updates=0,feature_parameters_buffers_unchanged=True,fit_L_count=6300,fit_classes=list(range(6)),fit_source_roles='EXACT_MATCH',fit_U_opened=False,fit_V_training=False,target_fit=False,truth_read=False,checkpoint=checkpoint_path(c),seed=c['model_seed']+60013,source_fit=dict(c['fresh_cosine_probe']),fresh_cosine_audit=dict(epochs=20,updates=500,feature_updates=0,losses=[1.]*20),feature_parameters=221569,feature_parameter_shapes=shapes,source_feature_max_difference=0.)
 validate_head_freeze(c,freeze);negative=0
 for k,v in [('old_W_discarded',False),('feature_updates',1),('fit_U_opened',True),('fit_V_training',True),('target_fit',True),('truth_read',True),('source_feature_max_difference',.1),('feature_parameters',176065),('readouts',['ORIGINAL_HEAD']),('fit_source_roles','unknown')]:
  bad=copy.deepcopy(freeze);bad[k]=v
  try:validate_head_freeze(c,bad)
  except ValueError:negative+=1
  else:raise AssertionError('Illegal or incomplete readout freeze accepted: '+k)
 for k,v in [('updates',499),('epochs',21),('feature_updates',1),('losses',[float('nan')]*20)]:
  bad=copy.deepcopy(freeze);bad['fresh_cosine_audit'][k]=v
  try:validate_head_freeze(c,bad)
  except ValueError:negative+=1
  else:raise AssertionError('Wrong readout budget accepted')
 records=[]
 strata=dict(overall=['ALL'],receiver=list(HELDOUT_RX),transmitter=list(d.CLASSES),day=list(TARGET_DAYS))
 for readout in d.READOUTS:
  for dimension,values in strata.items():
   n=d.QUERY_COUNT//len(values)
   for stratum in values:
    cm=[[0]*6 for _ in range(6)]
    for i in range(6):cm[i][i]=n//6+int(i<n%6)
    records.append(dict(row_id=c['row_id'],arm=c['arm'],model_seed=c['model_seed'],stage=c['stage'],view='clean',readout=readout,dimension=dimension,stratum=stratum,query_count=n,accuracy=1.,macro_f1=1.,confusion=cm))
 assert validate_records(c,records)==54
 candidates=[records[:-1],records+[records[0]]]
 for k,v in [('model_seed',1),('row_id','wrong'),('readout','RIDGE'),('view','unknown'),('accuracy',float('nan')),('macro_f1',0.),('query_count',1),('confusion',[[0]*6 for _ in range(6)])]:
  bad=copy.deepcopy(records);bad[0][k]=v;candidates.append(bad)
 for bad in candidates:
  try:validate_records(c,bad)
  except ValueError:negative+=1
  else:raise AssertionError('Malformed/incomplete score accepted')
 completion=row_completion(c,54);assert completion['truth_last'] and completion['Phase2_claim'] is False
 return dict(status='PASS',synthetic_only=True,real_IQ_or_truth_read=False,head_budget=500,backbone_updates_after_freeze=0,score_records=54,negative_cases=negative)
if __name__=='__main__':print(json.dumps(checks()))
