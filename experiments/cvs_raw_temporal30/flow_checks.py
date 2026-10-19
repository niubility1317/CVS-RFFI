from unittest.mock import patch
from . import design as d,report
def run():
 def sample(u,clean):return dict(source_accuracies=[[0.,u,u],[0.,u,u]],full_V_clean_accuracy=clean)
 values={a:sample(.70,.98) for a in ('RAW_As','RAW_NoAs','PAIR_As','PAIR_NoAs','FULL_As','FULL_NoAs','TEMP_As','TEMP_NoAs')};values['RAW_As']=sample(.72,.98);values['FULL_As']=sample(.71,.98);values['RAW_NoAs']=sample(.5,.98);values['PAIR_NoAs']=sample(.5,.98);writes=[]
 def read(path):
  assert 'scoring' not in str(path) and 'prediction' not in str(path) and 'truth' not in str(path)
  arm=path.parent.parent.name.split('-s')[0].replace('_JOINT','');return values[arm]
 with patch.object(d,'read',read),patch.object(d,'write',lambda p,v:writes.append(v)):
  assert report.source_decision()['preferred']=='RAW'
  values['RAW_As']=sample(.711,.98);assert report.source_decision()['preferred'] is None
  values['RAW_As']=sample(.72,.977);assert report.source_decision()['preferred'] is None
  values['RAW_As']=sample(.72,.98);values['RAW_NoAs']=sample(.5,.983);assert report.source_decision()['preferred'] is None
 assert all(not x['formal_scores_read'] and not x['automatic_extension'] for x in writes)
 return dict(status='PASS',NN=0,source_only_inputs=True,fixed_FULL_PAIR_clean_and_own_NoAs_rules=True)
if __name__=='__main__':print(run())
