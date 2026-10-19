import time
import numpy as np
from . import design as d

def source_decision():
 joint={r['arm']:d.read(d.BASE/r['row_id']/'source_qualification/complete.json') for r in d.rows()};frozen={r['arm']:d.read(d.FROZEN_BASE/(r['arm']+'-s'+str(d.SEED))/'source_qualification/complete.json') for r in d.rows()}
 def urban(group,a):return sum(group[a]['source_accuracies'][i][j] for i in range(2) for j in (1,2))/4
 def clean(group,a):return group[a]['full_V_clean_accuracy']
 criteria={};eligible=[]
 for style in ('FULL','LORA','TEMP'):
  a,b=style+'_As',style+'_NoAs';gain=100*(urban(joint,a)-urban(frozen,a));keep=100*(clean(joint,a)-clean(frozen,a));net=100*(clean(joint,a)-clean(joint,b));dfj=100*(urban(joint,a)-urban(joint,'FULL_As'));dff=100*(urban(joint,a)-urban(frozen,'FULL_As'));passed=gain>=.5 and keep>=-.2 and net>=-.2 and (style=='FULL' or min(dfj,dff)>=.5)
  criteria[style]=dict(joint_minus_same_frozen_As_urban_pp=gain,joint_minus_same_frozen_As_fullVclean_pp=keep,joint_As_minus_joint_NoAs_fullVclean_pp=net,urban_minus_FULL_joint_As_pp=dfj,urban_minus_FULL_frozen_As_pp=dff,NoAs_head_effect_urban_pp=100*(urban(joint,b)-urban(frozen,b)),As_specific_head_effect_urban_pp=100*((urban(joint,a)-urban(frozen,a))-(urban(joint,b)-urban(frozen,b))),qualified=passed)
  if passed:eligible.append(style)
 order=sorted(eligible,key=lambda s:(-urban(joint,s+'_As'),-clean(joint,s+'_As'),{'LORA':3264,'TEMP':3344,'FULL':232193}[s]))
 result=dict(status='SOURCE_DECISION_FIXED',timestamp=time.time(),criteria=criteria,eligible_source_order=order,preferred=order[0] if order else None,joint_readouts=joint,frozen_readouts=frozen,formal_scores_read=False,automatic_extension=False,max_cumulative_epochs=50,current_cumulative_epochs=30,model_seed_count=1,goal_complete=False,decision='SOURCE_SUPPORTS_FUTURE_MATCHED_EXTENSION_AT_MOST20' if order else 'CLOSE_FIXED_JOINT_W_COMPARISON;no W-lr/rank/kernel/lambda/seed rescue')
 d.write(d.BASE/'source_matrix_decision.json',result);return result

def collect():
 from experiments.cvs_feature_adapters30 import report as old
 # Reuse the already exercised collection algorithm against only this new run.
 original=old.d;old.d=d
 try:result=old.collect()
 finally:old.d=original
 from experiments.cvs_early_geometry.evaluate import truth_arrays
 pred={};ids=None;frozen_scores={};effects={};pairs={}
 for r in d.rows():
  arm=r['arm'];frozen=d.FROZEN_BASE/(arm+'-s'+str(d.SEED));records=d.read(frozen/'scoring/scores.json')['results'];frozen_scores[arm]={v['view']:v for v in records if v['dimension']=='overall'}
  with np.load(d.BASE/r['row_id']/'prediction/predictions.npz',allow_pickle=False) as f:
   if ids is None:ids=f['ids'].copy()
   if not np.array_equal(ids,f['ids']):raise ValueError('New joint physical prediction IDs differ')
   pred[arm]={v:f[v].copy() for v in d.VIEWS}
 y,rx,day=truth_arrays(d.read(d.TRUTH),ids)
 for r in d.rows():
  arm=r['arm'];effects[arm]={v:100*(result['rows'][arm]['overall'][v]['accuracy']-frozen_scores[arm][v]['accuracy']) for v in d.VIEWS};paired=[]
  with np.load(d.FROZEN_BASE/(arm+'-s'+str(d.SEED))/'prediction/predictions.npz',allow_pickle=False) as f:
   if not np.array_equal(ids,f['ids']):raise ValueError('Frozen/joint observation IDs differ')
   for view in d.VIEWS:
    for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
     for value in values:
      mask=np.ones(len(y),bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;a=pred[arm][view][mask];b=f[view][mask];t=y[mask];repair=int(((a==t)&(b!=t)).sum());damage=int(((a!=t)&(b==t)).sum());net=100*(repair-damage)/int(mask.sum())
      if dim=='overall' and abs(net-effects[arm][view])>1e-10:raise ValueError('Head-effect repair/damage recount differs')
      paired.append(dict(view=view,dimension=dim,stratum=str(value),count=int(mask.sum()),repair=repair,damage=damage,net_pp=net))
  pairs[arm]=paired
 result.update(frozen_head_control_run=d.FROZEN_RUN,frozen_head_overall=frozen_scores,joint_minus_frozen_pp=effects,head_effect_pairs=pairs,head_As_interaction_pp={s:{v:effects[s+'_As'][v]-effects[s+'_NoAs'][v] for v in d.VIEWS} for s in ('FULL','LORA','TEMP')})
 d.write(d.BASE/'analysis.json',result);return result
