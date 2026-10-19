from unittest.mock import patch
from . import design as d,report
def run():
 def sample(u,clean):return dict(source_accuracies=[[0.,u,u],[0.,u,u]],full_V_clean_accuracy=clean)
 values={a:sample(.70,.98) for a in ('RAW_As','RAW_NoAs','PAIR_As','PAIR_NoAs','FULL_As','FULL_NoAs','TEMP_As','TEMP_NoAs')};writes=[]
 def read(path):
  assert not any(v in str(path) for v in ('scoring','prediction','truth'))
  return values[path.parent.parent.name.split('-s')[0].replace('_JOINT','')]
 with patch.object(d,'read',read),patch.object(d,'write',lambda p,v:writes.append(v)):
  values['RAW_As']=sample(.71,.98);assert report.source_decision()['eligible']==['RAW']
  values['RAW_As']=sample(.704,.98);assert report.source_decision()['preferred'] is None
  values['RAW_As']=sample(.71,.977);assert report.source_decision()['preferred'] is None
  values['RAW_As']=sample(.71,.98);values['RAW_NoAs']=sample(.5,.983);assert report.source_decision()['preferred'] is None
  values['RAW_NoAs']=sample(.5,.98);values['PAIR_As']=sample(.715,.98);assert report.source_decision()['preferred']=='PAIR'
  values['RAW_As']=sample(.715,.98);assert report.source_decision()['preferred']=='RAW'
 assert all(not x['formal_scores_read'] and not x['automatic_extension'] for x in writes)
 return dict(status='PASS',NN=0,source_only_inputs=True,per_style_FULL_increment_clean_ownNoAs_and_fixed_tie_rule=True)
if __name__=='__main__':print(run())
