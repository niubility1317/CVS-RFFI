"""Independent source-only recount from fixed logits;no model/IQ/checkpoint input."""
import json,math,time
from . import design as d

def metric(y,p):
 cm=[[0]*6 for _ in range(6)]
 for a,b in zip(y,p):cm[a][b]+=1
 n=len(y);correct=sum(cm[i][i] for i in range(6));r=[sum(v) for v in cm];c=[sum(cm[j][i] for j in range(6)) for i in range(6)]
 f1=sum(2*cm[i][i]/(r[i]+c[i]) if r[i]+c[i] else 0. for i in range(6))/6
 return dict(count=n,correct=correct,accuracy=correct/n,macro_F1=f1,confusion=cm)

def run():
 started=time.perf_counter();out=d.resolve(d.BASE);complete=d.read(out/'prediction_complete.json');c=d.config();d.validate(complete['config'])
 if complete['status']!='SOURCE_PREDICTIONS_COMPLETE' or complete['NN']!=21604 or not complete['model_state_unchanged'] or complete['target_access'] or complete['updates']:raise ValueError('All declared source predictions must finish')
 a=d.read(out/'predictions.json');ids=a['ids'];y=a['y'];rx=a['rx'];days=a['day']
 if len(ids)!=1800 or len(set(ids))!=1800 or len(y)!=1800 or set(y)!=set(range(6)) or set(rx)!=set(d.FIT) or set(days)!={1,2,3}:raise ValueError('Fixed all90cell V grid required')
 rows=[];summary={};pairs=[];criteria={};curves={}
 for arm in d.MODELS:
  if set(a['logits'][arm])!={b+'/'+str(x) for b in c['base_views'] for x in c['operator']['alpha_values']}:raise ValueError('Missing/extra source view')
  summary[arm]={}
  for base in c['base_views']:
   predictions={}
   for alpha in c['operator']['alpha_values']:
    key=base+'/'+str(alpha);logits=a['logits'][arm][key]
    if len(logits)!=1800 or any(len(z)!=6 or any(not math.isfinite(v) for v in z) for z in logits):raise ValueError('Finite default6 logits required')
    pred=[max(range(6),key=lambda k:z[k]) for z in logits];predictions[str(alpha)]=pred;overall=metric(y,pred);summary[arm][key]=overall
    for dimension,groups in [('overall',{'ALL':list(range(1800))}),('RX',{str(r):[i for i in range(1800) if rx[i]==r] for r in d.FIT}),('TX',{str(t):[i for i in range(1800) if y[i]==t] for t in range(6)}),('day',{str(z):[i for i in range(1800) if days[i]==z] for z in (1,2,3)}),('TX_RX_day',{f'{t}/{r}/{z}':[i for i in range(1800) if (y[i],rx[i],days[i])==(t,r,z)] for t in range(6) for r in d.FIT for z in (1,2,3)})]:
     for name,indices in groups.items():rows.append(dict(arm=arm,base=base,alpha=alpha,dimension=dimension,stratum=name,**metric([y[i] for i in indices],[pred[i] for i in indices])))
   p0=predictions['0.0'];correct0=[p0[i]==y[i] for i in range(1800)];n0=sum(correct0)
   for alpha in (-.1,.1):
    pred=predictions[str(alpha)];correct=[pred[i]==y[i] for i in range(1800)];repairs=sum(v and not b for v,b in zip(correct,correct0));new=sum(b and not v for v,b in zip(correct,correct0));delta=(sum(correct)-n0)/1800*100;retention=sum(v and b for v,b in zip(correct,correct0))/n0 if n0 else 0.
    entry=dict(arm=arm,base=base,alpha=alpha,delta_accuracy_pp=delta,repairs=repairs,new_errors=new,baseline_correct=n0,correct_retention=retention)
    entry['RX_delta_pp']={str(r):100*(sum(correct[i] for i in range(1800) if rx[i]==r)-sum(correct0[i] for i in range(1800) if rx[i]==r))/sum(v==r for v in rx) for r in d.FIT}
    entry['TX_delta_pp']={str(t):100*(sum(correct[i] for i in range(1800) if y[i]==t)-sum(correct0[i] for i in range(1800) if y[i]==t))/sum(v==t for v in y) for t in range(6)}
    pairs.append(entry);criteria[f'{arm}/{base}/{alpha}/accuracy']=delta>=-.5;criteria[f'{arm}/{base}/{alpha}/retention']=retention>=.995;criteria[f'{arm}/{base}/{alpha}/RX']=all(v>=-1. for v in entry['RX_delta_pp'].values());criteria[f'{arm}/{base}/{alpha}/TX']=all(v>=-1. for v in entry['TX_delta_pp'].values())
 for r in complete['numerics']:
  criteria[r['base']+'/'+str(r['alpha'])+'/energy']=r['max_energy_relative_error']<=1e-5;criteria[r['base']+'/'+str(r['alpha'])+'/inverse']=r['max_inverse_relative_error']<=1e-5
 passed=all(criteria.values());value=dict(status='ANALYZED_SOURCE_IDENTITY_PROTECTION_ONLY',config=c,source_criteria_passed=passed,decision='SOURCE_PROTECTION_ONLY_NOT_DG_EFFICACY' if passed else 'CLOSED_FIXED_UNITARY_PROFILE_FAILED_SOURCE_PROTECTION',criteria=criteria,summary=summary,paired=pairs,strata=rows,numerics=complete['numerics'],ordinary_actual_stage=complete['ordinary_actual_stage'],cost={k:complete[k] for k in ('NN','source_cache_loads','CP','ordinary_generations','operator_applications','forward_operator_applications','inverse_check_applications','teacher_NN','updates','head_fit_updates','wall_seconds')},analysis_wall_seconds=time.perf_counter()-started,source_RX_holdout=None,model_state_unchanged=True,target_access=False,new_Ag_efficacy=False,joint_goal_complete=False)
 d.write(out/'result.json',value);d.write(out/'completion.json',{k:v for k,v in value.items() if k!='strata'});print('SOURCE_IDENTITY_GUARD_COMPLETE '+value['decision'],flush=True)
if __name__=='__main__':run()
