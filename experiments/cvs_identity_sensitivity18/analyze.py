"""Separate source scoring from saved logits/derivatives;no IQ/model/checkpoint reads."""
import math,time
from collections import Counter
from . import design as d

def metric(y,p):
 cm=[[0]*6 for _ in range(6)]
 for a,b in zip(y,p):cm[a][b]+=1
 n=len(y);correct=sum(cm[i][i] for i in range(6));r=[sum(v) for v in cm];c=[sum(cm[j][i] for j in range(6)) for i in range(6)]
 return dict(count=n,correct=correct,accuracy=correct/n,macro_F1=sum(2*cm[i][i]/(r[i]+c[i]) if r[i]+c[i] else 0. for i in range(6))/6,confusion=cm)

def rerank(fit):
 a=fit['derivatives'];y=fit['y'];r=fit['rx'];z=fit['day']
 if len(a)!=180 or len(set(fit['ids']))!=180 or Counter(zip(y,r,z))!=Counter({(t,rx,day):2 for t in range(6) for rx in d.FIT for day in (1,2,3)}):raise ValueError('Fixed source L grid required')
 relative=[];zero=0
 for row in a:
  if len(row)!=16 or any(not math.isfinite(v) for v in row):raise ValueError('All finite L derivatives required')
  denom=sum(abs(v) for v in row);zero+=int(denom==0);relative.append([abs(v)/denom for v in row] if denom else [1/16]*16)
 cells=[]
 for tx in range(6):
  for rx in d.FIT:
   for day in (1,2,3):
    idx=[i for i in range(180) if (y[i],r[i],z[i])==(tx,rx,day)];cells.append([sum(relative[i][b] for i in idx)/2 for b in range(16)])
 scores=[max(row[b] for row in cells) for b in range(16)];order=sorted(range(16),key=lambda b:(scores[b],b));result=dict(scores=scores,low_blocks=order[:4],high_blocks=order[-4:],source_zero_gradient_packets=zero,cell_scores=cells)
 registered=fit['ranking']
 if any(result[k]!=registered[k] for k in ('low_blocks','high_blocks','source_zero_gradient_packets')) or max(abs(a-b) for a,b in zip(scores,registered['scores']))>1e-12:raise ValueError('Fixed L-only ranking recount differs')
 return result

def qualify(pairs,ranking,numerics,unavailable):
 c=d.config()['criteria'];criteria={};by_arm={};low=set(ranking['low_blocks']);high=set(ranking['high_blocks'])
 scores=ranking['scores'];lo=sum(scores[b] for b in low)/4;hi=sum(scores[b] for b in high)/4
 criteria['L_score_separation']=hi>=c['source_high_score_over_low_min']*lo+1e-12;criteria['input_gradient_available']=not unavailable
 for arm in d.MODELS:
  entries=[p for p in pairs if p['arm']==arm];selected=[p for p in entries if p['block'] in low]
  if len(entries)!=32 or len(selected)!=8:raise ValueError('All32 V views required per default head')
  uniform=sum(p['new_errors'] for p in entries);low_errors=sum(p['new_errors'] for p in selected);high_errors=sum(p['new_errors'] for p in entries if p['block'] in high);n0=entries[0]['baseline_correct']
  if not n0 or any(p['baseline_correct']!=n0 for p in entries):raise ValueError('Common nonzero baseline required')
  by_arm[arm]=dict(uniform_new_errors=uniform,low_new_errors=low_errors,high_new_errors=high_errors,uniform_error_opportunities=n0*32,low_error_opportunities=n0*8,high_error_opportunities=n0*8,low_over_uniform=(low_errors*4/uniform if uniform else None),low_over_high=(low_errors/high_errors if high_errors else None))
  criteria[arm+'/informative_controls']=uniform>=c['uniform_new_errors_each_readout_min']
  criteria[arm+'/low_over_uniform']=uniform>0 and low_errors*4<=c['low_error_rate_over_uniform_max']*uniform
  criteria[arm+'/low_over_high']=high_errors>0 and low_errors<=c['low_error_rate_over_high_max']*high_errors
  for p in selected:
   name=arm+'/'+str(p['block'])+'/'+str(p['sign'])
   criteria[name+'/accuracy']=p['delta_accuracy_pp']>=c['each_low_view_accuracy_delta_pp_min']
   criteria[name+'/retention']=p['correct_retention']>=c['each_low_view_correct_retention_min']
   criteria[name+'/RX']=all(v>=c['each_RX_or_TX_accuracy_delta_pp_min'] for v in p['RX_delta_pp'].values())
   criteria[name+'/TX']=all(v>=c['each_RX_or_TX_accuracy_delta_pp_min'] for v in p['TX_delta_pp'].values())
 if len(numerics)!=32 or {(v['block'],v['sign']) for v in numerics}!={(b,s) for b in range(16) for s in (-1,1)}:raise ValueError('All energy controls required')
 for v in numerics:
  k=str(v['block'])+'/'+str(v['sign']);criteria[k+'/gain']=v['max_gain']<=.1+1e-12;criteria[k+'/step']=v['step_error']<=1e-6;criteria[k+'/outside']=v['outside_unchanged'] is True;criteria[k+'/relative_step']=v['max_relative_change']<=.01+1e-6
 return dict(passed=all(criteria.values()),criteria=criteria,controls=by_arm,L_score_mean_low=lo,L_score_mean_high=hi)

def run():
 started=time.perf_counter();out=d.resolve(d.BASE);complete=d.read(out/'prediction_complete.json');d.validate(complete['config'])
 if complete['status']!='SOURCE_PREDICTIONS_COMPLETE' or complete['NN']!=118984 or not complete['model_state_unchanged'] or complete['target_access'] or complete['updates']:raise ValueError('All fixed source predictions required')
 fit=d.read(out/'source_L_ranking.json');ranking=rerank(fit);a=d.read(out/'predictions.json');y=a['y'];rx=a['rx'];days=a['day'];ids=a['ids']
 if len(ids)!=1800 or len(set(ids))!=1800 or set(ids)&set(fit['ids']) or Counter(zip(y,rx,days))!=Counter({(t,r,z):20 for t in range(6) for r in d.FIT for z in (1,2,3)}):raise ValueError('Fixed disjoint all90cell V grid required')
 groups=[('overall',{'ALL':list(range(1800))}),('RX',{str(r):[i for i in range(1800) if rx[i]==r] for r in d.FIT}),('TX',{str(t):[i for i in range(1800) if y[i]==t] for t in range(6)}),('day',{str(z):[i for i in range(1800) if days[i]==z] for z in (1,2,3)}),('TX_RX_day',{str(t)+'/'+str(r)+'/'+str(z):[i for i in range(1800) if (y[i],rx[i],days[i])==(t,r,z)] for t in range(6) for r in d.FIT for z in (1,2,3)})]
 strata=[];pairs=[];summary={}
 for arm in d.MODELS:
  expected={'raw'}|{str(b)+'/'+str(s) for b in range(16) for s in (-1,1)}
  if set(a['logits'][arm])!=expected:raise ValueError('All33 fixed source views required')
  predictions={};summary[arm]={}
  for key,logits in a['logits'][arm].items():
   if len(logits)!=1800 or any(len(v)!=6 or any(not math.isfinite(t) for t in v) for v in logits):raise ValueError('Finite6 default logits required')
   p=[max(range(6),key=lambda k:v[k]) for v in logits];predictions[key]=p;summary[arm][key]=metric(y,p)
   for dimension,parts in groups:
    for name,idx in parts.items():strata.append(dict(arm=arm,view=key,dimension=dimension,stratum=name,**metric([y[i] for i in idx],[p[i] for i in idx])))
  base=[p==t for p,t in zip(predictions['raw'],y)];n0=sum(base)
  for block in range(16):
   for sign in (-1,1):
    good=[p==t for p,t in zip(predictions[str(block)+'/'+str(sign)],y)];new=sum(b and not v for b,v in zip(base,good));repairs=sum(v and not b for b,v in zip(base,good))
    entry=dict(arm=arm,block=block,sign=sign,baseline_correct=n0,new_errors=new,repairs=repairs,delta_accuracy_pp=(repairs-new)/18,correct_retention=(n0-new)/n0 if n0 else 0.)
    for dimension,values,valid in [('RX',rx,d.FIT),('TX',y,range(6))]:
     entry[dimension+'_delta_pp']={str(v):100*sum(int(good[i])-int(base[i]) for i in range(1800) if values[i]==v)/sum(t==v for t in values) for v in valid}
    pairs.append(entry)
 q=qualify(pairs,ranking,complete['numerics'],complete['gradient_unavailable']);decision='SOURCE_RANKING_ONLY_NOT_DG_EFFICACY' if q['passed'] else 'CLOSED_FIXED_IDENTITY_SENSITIVITY_SOURCE_RANKING_FAILED'
 value=dict(status='ANALYZED_SOURCE_RANKING_ONLY',config=d.config(),decision=decision,source_criteria_passed=q['passed'],qualification=q,ranking=ranking,summary=summary,paired=pairs,strata=strata,numerics=complete['numerics'],cost={k:complete[k] for k in ('NN','CP','source_cache_loads','source_L_selected','source_V_selected','input_gradient_packets','input_backward_tensor_calls','operator_applications','ordinary_generations','teacher_NN','updates','head_fit_updates','wall_seconds')},analysis_wall_seconds=time.perf_counter()-started,source_RX_holdout=None,target_access=False,new_Ag_efficacy=False,new_As_efficacy=False,joint_goal_complete=False)
 d.write(out/'result.json',value);d.write(out/'completion.json',{k:v for k,v in value.items() if k!='strata'});print('SOURCE_SENSITIVITY_COMPLETE '+decision,flush=True)
if __name__=='__main__':run()
