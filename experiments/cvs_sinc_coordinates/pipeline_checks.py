import copy,json,tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch
from . import design as d
from .predict import pipeline
from experiments.cvs_pa_cross_delay.pipeline_checks import records_for
from experiments.cvs_pa_cross_delay.score import validate_records,row_completion,finish

def checks():
 c=d.config(d.training_rows()[0]);p=pipeline();frozen=p.frozen(c);negatives=0
 for key,value in [('separate_head_fit',True),('separate_head_updates',1),('backbone_updates_after_freeze',1),('readouts',['FRESH_COSINE_ALL_L']),('truth_read',True),('query_fit',True),('head_shape',[8,160]),('head_scale',10)]:
  bad=copy.deepcopy(frozen);bad[key]=value
  assert bad!=p.frozen(c);negatives+=1
 with tempfile.TemporaryDirectory() as folder,patch.object(d,'BASE',Path(folder)):
  c=d.config(d.training_rows()[0])
  root=d.BASE/c['row_id']/'prediction';root.mkdir(parents=True);ids=np.arange(d.QUERY_COUNT)
  p.proof=lambda c:Path('/synthetic/source');p.observation=lambda:(ids,{})
  d.write(root/'default_frozen.json',p.frozen(c));d.write(root/'provenance.json',p.provenance(c,Path('/synthetic/source')));d.write(root/'complete.json',p.complete(c))
  np.savez_compressed(root/'predictions.npz',ids=ids,ORIGINAL_HEAD=np.zeros(d.QUERY_COUNT,dtype=np.int64))
  p.preflight(c['row_id'])
  for key,value in [('separate_head_fit',True),('separate_head_updates',1),('backbone_updates_after_freeze',1),('readouts',['FRESH_COSINE_ALL_L']),('truth_read',True),('query_fit',True)]:
   bad=copy.deepcopy(p.frozen(c));bad[key]=value;d.write(root/'default_frozen.json',bad)
   try:p.preflight(c['row_id'])
   except ValueError:negatives+=1
   else:raise AssertionError('Illegal default freeze opened truth')
 record=records_for(c);assert validate_records(c,record,design=d)==18
 for bad in [record[:-1],record+[record[0]]]:
  try:validate_records(c,bad,design=d)
  except ValueError:negatives+=1
  else:raise AssertionError('Illegal score coverage accepted')
 with tempfile.TemporaryDirectory() as temp,patch.object(d,'BASE',Path(temp)):
  for row in d.rows():
   c=d.config(row);records=records_for(c,.1+d.ARMS.index(row['arm'])*.01);out=d.BASE/row['row_id'];(out/'scores').mkdir(parents=True)
   d.write(out/'scores/metrics.json',dict(records=records,readouts=list(d.READOUTS),count_per_readout=d.QUERY_COUNT,truth_last=True,independent_recount='VERIFIED',default_classifier_only=True,Phase2_claim=False))
   d.write(out/'completion.json',row_completion(c,18,design=d))
  result=finish(design=d)
  assert result['models']==9 and result['prediction_count']==1512000 and len(result['records'])==162 and len(result['summary'])==3 and len(result['paired'])==3
  assert finish(design=d)==result
 return dict(status='PASS',synthetic_only=True,query_read=False,models=9,records=162,pairs=3,separate_head_updates=0,negative_checks=negatives)
if __name__=='__main__':print(json.dumps(checks()))
