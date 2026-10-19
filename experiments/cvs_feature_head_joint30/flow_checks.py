import json
from unittest.mock import patch
from . import design as d,report

def run():
 frozen={};joint={}
 for r in d.rows():
  s=r['style'];a=r['As'];f={'FULL':.70,'LORA':.58,'TEMP':.57}[s] if a else .47;j={'FULL':.71,'LORA':.72,'TEMP':.711}[s] if a else .47
  frozen[r['arm']]=dict(source_accuracies=[[f]*3]*2,full_V_clean_accuracy=.98);joint[r['arm']]=dict(source_accuracies=[[j]*3]*2,full_V_clean_accuracy=.98)
 def read(p):
  arm=next(r['arm'] for r in d.rows() if r['arm'] in str(p));return (frozen if str(d.FROZEN_BASE) in str(p) else joint)[arm]
 writes=[]
 with patch.object(d,'read',read),patch.object(d,'write',lambda p,x:writes.append(x)):
  v=report.source_decision();assert v['preferred']=='LORA' and v['criteria']['FULL']['qualified'] and v['criteria']['LORA']['qualified'] and not v['criteria']['TEMP']['qualified'] and not v['formal_scores_read']
  joint['LORA_As']['full_V_clean_accuracy']=.976;v=report.source_decision();assert not v['criteria']['LORA']['qualified'] and v['preferred']=='FULL'
  for arm in joint:joint[arm]=frozen[arm]
  v=report.source_decision();assert not v['eligible_source_order'] and v['preferred'] is None and not v['automatic_extension']
 return dict(status='PASS',head_gain_and_FULL_control_and_clean_rules_verified=True,NoAs_head_effect_reported=True,source_only_decision=True,NN=0,real_IQ=0,checkpoint_loads=0,target_access=False)

if __name__=='__main__':print(json.dumps(run()))
