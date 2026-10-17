"""Focused fixed-matrix and negative-role tests; no torch or real data."""
from copy import deepcopy
from . import design as d

def run():
 negative=0
 for row in d.rows():
  c=d.config(row);d.validate(c)
  assert c['data_contract']==dict(RX=[1,3,4,6,8],days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True)
  assert c['epochs']==20 and c['label_epochs']==13 and c['pseudo_epochs']==7 and c['updates']==4440 and not c['baseline_reuse'] and c['source_falsification'] is None and c['test_views']==list(d.VIEWS)
  for path,value in ((('epochs',),50),(('data_contract','RX'),[3,4,6,8]),(('data_contract','source_RX_holdout'),1),(('data_contract','L'),5040),(('data_contract','U_TX_truth_hidden'),False),(('test_views',),['clean']),(('training_mechanism','alpha'),0.),(('supcon','weight'),.2),(('baseline_reuse',),True),(('checkpoint_sources',),['old']),(('model_seed',),2026092702)):
   bad=deepcopy(c);v=bad
   for k in path[:-1]:v=v[k]
   v[path[-1]]=value
   try:d.validate(bad)
   except ValueError:negative+=1
   else:raise AssertionError('Changed protocol/config accepted')
 return dict(status='PASS',negative_cases=negative,rows=3,full_source_RX=list(d.FIT),L=6300,U=56700,V=27000,epochs_per_model=20,source_RX_holdout=None,IQ=0,CP=0,NN=0,updates=0,target_access=False)
if __name__=='__main__':
 import json
 print(json.dumps(run()))
