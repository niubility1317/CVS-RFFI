"""Describe complete existing Q_ord versus Q0 identity errors;no candidate changes."""
import json,math
from pathlib import Path
from .analyze import PATHS,OUT,VIEWS,read,prediction,dump,decision_transitions

def analyze():
 pp={arm:{v:read(PATHS[arm]/'source_readouts'/(v+'.json')) for v in VIEWS} for arm in ('Q0','Q_ord')};scores={arm:{v:prediction(p) for v,p in views.items()} for arm,views in pp.items()}
 raw=pp['Q0']['clean'];asclean=pp['Q_ord']['clean'];assert raw['labels']==asclean['labels'] and raw['physical_ids']==asclean['physical_ids'];y=raw['labels'];b=raw['predictions'];s=asclean['predictions'];groups={k:[] for k in ('both_clean_correct','As_clean_repaired','As_clean_broken','both_clean_wrong')}
 for i,t in enumerate(y):groups['both_clean_correct' if b[i]==t and s[i]==t else 'As_clean_repaired' if b[i]!=t and s[i]==t else 'As_clean_broken' if b[i]==t and s[i]!=t else 'both_clean_wrong'].append(i)
 assert sum(map(len,groups.values()))==1260
 views={}
 for view in VIEWS:
  r,q=pp['Q0'][view],pp['Q_ord'][view];assert r['labels']==q['labels']==y and r['physical_ids']==q['physical_ids']==raw['physical_ids'];full=decision_transitions(pp['Q0'],pp['Q_ord'],view);parts={}
  for name,take in groups.items():
   repair=sum(r['predictions'][i]!=y[i] and q['predictions'][i]==y[i] for i in take);broken=sum(r['predictions'][i]==y[i] and q['predictions'][i]!=y[i] for i in take)
   parts[name]=dict(count=len(take),repair=repair,broken=broken,net_correct=repair-broken,contribution_pp=100*(repair-broken)/1260)
  assert sum(v['net_correct'] for v in parts.values())==full['all']['net_correct']
  delta=100*(scores['Q_ord'][view]['accuracy']-scores['Q0'][view]['accuracy']);assert math.isclose(delta,full['all']['delta_pp'],abs_tol=1e-10)
  views[view]=dict(delta_accuracy_pp=delta,delta_macroF1_pp=100*(scores['Q_ord'][view]['macro_F1']-scores['Q0'][view]['macro_F1']),paired=full,disjoint_clean_groups=parts)
 value=dict(status='ANALYZED_FIXED_CONTROL_CONTRAST',source_only=True,reference='Q0',As_control='Q_ord',model_seed=2026092701,source_RX=1,source_packets=1260,repeated_scene_decisions=17640,views=views,scores=scores,new_NN=0,new_IQ_reads=0,new_checkpoint_reads=0,new_updates=0,target_access=False,candidate_or_threshold_changed=False,scope='One exposed sourcefold/seed;descriptive fullpairedcounts,not propagationcausality/crossseed/Ag/formaltarget/fulljoint proof')
 OUT.mkdir(exist_ok=True);dump(OUT/'control_As_effect.json',value);print(json.dumps({v:{'gain_pp':z['delta_accuracy_pp'],'repair':z['paired']['all']['repaired'],'broken':z['paired']['all']['broken'],'both_clean_correct':z['disjoint_clean_groups']['both_clean_correct']} for v,z in views.items()},indent=2));return value
if __name__=='__main__':analyze()
