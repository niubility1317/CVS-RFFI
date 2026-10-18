"""Independent NumPy recount of all fixed source outputs;never opens IQ or CP."""
import json,math
from pathlib import Path
import numpy as np
RUN='20261018-diagnostic-F-identitySensitivity-sourceLtoV-s2026092701-m1-r01'
def run(project):
 out=Path(project)/'runs'/RUN
 def read(name):return json.loads((out/name).read_text())
 result=read('result.json');a=read('predictions.json');fit=read('source_L_ranking.json');complete=read('prediction_complete.json');errors=[];counts=dict(confusion_matrices=0,paired_views=0,partition_reconciliations=0,ranking_values=0,criteria=0)
 def equal(actual,expected):
  error=float(np.max(np.abs(np.asarray(actual,dtype=float)-np.asarray(expected,dtype=float))))
  if not math.isfinite(error):raise ValueError('Nonfinite recount')
  errors.append(error)
 y=np.asarray(a['y']);rx=np.asarray(a['rx']);day=np.asarray(a['day']);pred={arm:{key:np.asarray(v).argmax(1) for key,v in views.items()} for arm,views in a['logits'].items()}
 def metric(p,mask):
  cm=np.bincount((6*y[mask]+p[mask]).astype(int),minlength=36).reshape(6,6);count=int(cm.sum());correct=int(np.trace(cm));den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()
  return dict(count=count,correct=correct,accuracy=correct/count,macro_F1=float(f1),confusion=cm.tolist())
 for row in result['strata']:
  dimension=row['dimension'];name=row['stratum'];mask=np.ones(1800,dtype=bool)
  if dimension=='RX':mask=rx==int(name)
  elif dimension=='TX':mask=y==int(name)
  elif dimension=='day':mask=day==int(name)
  elif dimension=='TX_RX_day':t,r,d=map(int,name.split('/'));mask=(y==t)&(rx==r)&(day==d)
  elif dimension!='overall':raise ValueError('Unknown dimension')
  m=metric(pred[row['arm']][row['view']],mask)
  for k,v in m.items():equal(row[k],v)
  counts['confusion_matrices']+=1
 for arm,views in pred.items():
  for key,p in views.items():
   m=metric(p,np.ones(1800,dtype=bool))
   for k,v in m.items():equal(result['summary'][arm][key][k],v)
   for dimension in ('RX','TX','day','TX_RX_day'):
    rows=[s for s in result['strata'] if (s['arm'],s['view'],s['dimension'])==(arm,key,dimension)];equal(np.sum([s['confusion'] for s in rows],axis=0),m['confusion']);counts['partition_reconciliations']+=1
 pairs=[]
 for row in result['paired']:
  arm=row['arm'];base=pred[arm]['raw']==y;good=pred[arm][str(row['block'])+'/'+str(row['sign'])]==y;new=int(np.sum(base&~good));repair=int(np.sum(~base&good));n0=int(base.sum());m=dict(new_errors=new,repairs=repair,baseline_correct=n0,delta_accuracy_pp=100*(repair-new)/1800,correct_retention=(n0-new)/n0)
  for k,v in m.items():equal(row[k],v)
  for dim,values,unique in [('RX',rx,(1,3,4,6,8)),('TX',y,range(6))]:
   m[dim+'_delta_pp']={str(v):float(100*(good[values==v].sum()-base[values==v].sum())/np.sum(values==v)) for v in unique}
   for k,v in m[dim+'_delta_pp'].items():equal(row[dim+'_delta_pp'][k],v)
  pairs.append(dict(arm=arm,block=row['block'],sign=row['sign'],**m));counts['paired_views']+=1
 values=np.abs(np.asarray(fit['derivatives']));den=values.sum(1,keepdims=True);values=np.divide(values,den,out=np.full_like(values,1/16),where=den>0);fy,fr,fd=[np.asarray(fit[k]) for k in ('y','rx','day')];cells=np.array([values[(fy==t)&(fr==r)&(fd==d)].mean(0) for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)]);scores=cells.max(0);order=np.lexsort((np.arange(16),scores));low=set(order[:4].tolist());high=set(order[-4:].tolist());ranking=result['ranking'];equal(ranking['scores'],scores);equal(ranking['cell_scores'],cells);equal(ranking['low_blocks'],order[:4]);equal(ranking['high_blocks'],order[-4:]);equal(ranking['source_zero_gradient_packets'],int((den==0).sum()));counts['ranking_values']=16+90*16+8+1
 criteria={};criteria['L_score_separation']=float(scores[list(high)].mean())>=1.5*float(scores[list(low)].mean())+1e-12;criteria['input_gradient_available']=not complete['gradient_unavailable']
 for arm in ('REPEAT','CONCAT'):
  rows=[p for p in pairs if p['arm']==arm];selected=[p for p in rows if p['block'] in low];uniform=sum(p['new_errors'] for p in rows);le=sum(p['new_errors'] for p in selected);he=sum(p['new_errors'] for p in rows if p['block'] in high);controls=result['qualification']['controls'][arm];equal(controls['uniform_new_errors'],uniform);equal(controls['low_new_errors'],le);equal(controls['high_new_errors'],he)
  criteria[arm+'/informative_controls']=uniform>=20;criteria[arm+'/low_over_uniform']=uniform>0 and le*4<=.75*uniform;criteria[arm+'/low_over_high']=he>0 and le<=.5*he
  for p in selected:
   name=arm+'/'+str(p['block'])+'/'+str(p['sign']);criteria[name+'/accuracy']=p['delta_accuracy_pp']>=-.5;criteria[name+'/retention']=p['correct_retention']>=.995;criteria[name+'/RX']=all(v>=-1 for v in p['RX_delta_pp'].values());criteria[name+'/TX']=all(v>=-1 for v in p['TX_delta_pp'].values())
 for v in complete['numerics']:
  k=str(v['block'])+'/'+str(v['sign']);criteria[k+'/gain']=v['max_gain']<=.1+1e-12;criteria[k+'/step']=v['step_error']<=1e-6;criteria[k+'/outside']=v['outside_unchanged'] is True;criteria[k+'/relative_step']=v['max_relative_change']<=.01+1e-6
 if criteria!=result['qualification']['criteria']:raise ValueError('Independent source qualification differs')
 counts['criteria']=len(criteria);equal(result['source_criteria_passed'],all(criteria.values()));equal(complete['NN'],4+180+1800*33*2)
 if max(errors)>1e-10 or counts['confusion_matrices']!=6930 or counts['paired_views']!=64:raise ValueError('All-output independent recount differs')
 summary={k:v for k,v in result.items() if k!='strata'};log=(out/'worker.stdout.log').read_text();view_lines=[v for v in log.splitlines() if v.startswith('VIEW_COMPLETE ')];expected={'VIEW_COMPLETE raw'}|{'VIEW_COMPLETE '+str(b)+'/'+str(s) for b in range(16) for s in (-1,1)}
 if len(view_lines)!=33 or set(view_lines)!=expected or 'SMOKE PASS 2CP/defaulthead/noquery' not in log or 'SOURCE_SENSITIVITY_COMPLETE '+result['decision'] not in log:raise ValueError('Full source execution log incomplete')
 return dict(result=summary,verification=dict(status='VERIFIED',**counts,max_absolute_error=max(errors),all33_view_log_lines_once=True,source_only=True,raw_IQ_reads=0,CP_loads=0,NN=0,updates=0,target_truth_reads=0))
