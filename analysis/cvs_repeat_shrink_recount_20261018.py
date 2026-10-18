"""Independent full source-logit recount;no IQ/checkpoint/model or target input."""
import argparse,json,math
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def run(directory):
 p=Path(directory);pred=read(p/'predictions.json');result=read(p/'result.json');done=read(p/'prediction_complete.json');y=np.asarray(pred['y']);rx=np.asarray(pred['rx']);day=np.asarray(pred['day']);n=len(y)
 assert n==1800 and len(set(pred['ids']))==n and set(rx)=={1,3,4,6,8} and set(day)=={1,2,3} and set(y)==set(range(6))
 assert all(np.count_nonzero((y==t)&(rx==r)&(day==z))==20 for t in range(6) for r in (1,3,4,6,8) for z in (1,2,3))
 assert done['NN']==14404 and done['CP']==2 and done['model_state_unchanged'] and not done['target_access'] and done['updates']==done['teacher_NN']==done['head_fit_updates']==0
 outputs={};maxerr=0.;cm_count=0
 def equal(a,b):
  nonlocal maxerr
  error=abs(float(a)-float(b));maxerr=max(maxerr,error);assert error<1e-12,(a,b)
 def matrix(yy,pp):
  cm=np.zeros((6,6),dtype=np.int64);np.add.at(cm,(yy,pp),1);return cm
 def compare(v,cm):
  nonlocal cm_count
  cm_count+=1;assert v['confusion']==cm.tolist() and v['count']==int(cm.sum()) and v['correct']==int(np.trace(cm))
  equal(v['accuracy'],np.trace(cm)/cm.sum());den=cm.sum(1)+cm.sum(0);f=np.divide(2*np.diag(cm),den,out=np.zeros(6,dtype=float),where=den!=0).mean();equal(v['macro_F1'],f)
 for arm,views in pred['logits'].items():
  assert arm in ('REPEAT','CONCAT') and set(views)=={'raw/0.0','raw/0.1','ordinary/0.0','ordinary/0.1'}
  for key,logits in views.items():
   a=np.asarray(logits);assert a.shape==(1800,6) and np.isfinite(a).all();outputs[(arm,key)]=a.argmax(1)
 for v in result['strata']:
  q=outputs[(v['arm'],v['base']+'/'+str(v['strength']))];dimension=v['dimension'];s=v['stratum']
  if dimension=='overall':mask=np.ones(n,dtype=bool)
  elif dimension=='RX':mask=rx==int(s)
  elif dimension=='TX':mask=y==int(s)
  elif dimension=='day':mask=day==int(s)
  elif dimension=='TX_RX_day':
   t,r,z=map(int,s.split('/'));mask=(y==t)&(rx==r)&(day==z)
  else:raise ValueError(dimension)
  compare(v,matrix(y[mask],q[mask]))
 assert cm_count==840
 for arm,views in result['summary'].items():
  for key,v in views.items():compare(v,matrix(y,outputs[(arm,key)]))
 criteria={};paired=[]
 for arm in ('REPEAT','CONCAT'):
  for base in ('raw','ordinary'):
   before=outputs[(arm,base+'/0.0')]==y;after=outputs[(arm,base+'/0.1')]==y;repairs=int(np.count_nonzero(after&~before));new=int(np.count_nonzero(before&~after));delta=(repairs-new)*100/n;retention=np.count_nonzero(before&after)/before.sum();key=arm+'/'+base+'/0.1'
   item=next(v for v in result['paired'] if v['arm']==arm and v['base']==base and v['strength']==.1)
   assert item['repairs']==repairs and item['new_errors']==new and item['baseline_correct']==int(before.sum());equal(item['delta_accuracy_pp'],delta);equal(item['correct_retention'],retention)
   criteria[key+'/accuracy']=bool(delta>=-.5);criteria[key+'/retention']=bool(retention>=.995)
   for dim,values in [('RX',rx),('TX',y)]:
    differences={str(v):float((after[values==v].sum()-before[values==v].sum())*100/(values==v).sum()) for v in np.unique(values)}
    assert set(differences)==set(item[dim+'_delta_pp'])
    for v,change in differences.items():equal(change,item[dim+'_delta_pp'][v])
    criteria[key+'/'+dim]=all(change>=-1. for change in differences.values())
   paired.append(dict(arm=arm,base=base,accuracy0=float(before.mean()),accuracy1=float(after.mean()),delta_accuracy_pp=delta,correct_retention=float(retention),repairs=repairs,new_errors=new))
 for v in done['numerics']:
  key=v['base']+'/'+str(v['strength']);criteria[key+'/max_energy_increase_relative']=v['max_energy_increase_relative']<=1e-6;criteria[key+'/max_cycle_mean_change_relative']=v['max_cycle_mean_change_relative']<=1e-6;criteria[key+'/max_relative_change']=v['max_relative_change']<=.100001
  for field in ('outside_unchanged','cycle_boundaries_unchanged'):criteria[key+'/'+field]=v[field] is True
 assert result['criteria']==criteria and result['source_criteria_passed']==all(criteria.values())
 assert (result['decision']=='SOURCE_PROTECTION_ONLY_NOT_DG_EFFICACY')==all(criteria.values())
 return dict(status='VERIFIED',compared_strata_CM=840,compared_summary_CM=8,paired=paired,criteria_count=len(criteria),failed_criteria=[k for k,v in criteria.items() if not v],all_conditions_passed=all(criteria.values()),max_error=maxerr,model_state_unchanged=True,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,teacher_NN=0,target_access=False)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('directory');a.add_argument('--output',required=True);args=a.parse_args();v=run(args.directory);Path(args.output).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps({k:v[k] for k in ('status','compared_strata_CM','criteria_count','all_conditions_passed','max_error')}))
