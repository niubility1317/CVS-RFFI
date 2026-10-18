"""Synthetic complete-draw/source threshold counterexamples;0realIQ/CP/NN/truth."""
import argparse,collections,copy,gzip,json,math
from pathlib import Path
from statistics import NormalDist
from analysis import cvs_shadow_lhs_recount_20261018 as r
def run(out):
 root=Path(out);root.mkdir(parents=True,exist_ok=False);negatives=0;cases=[]
 def bad(call):
  nonlocal negatives
  try:call()
  except (ValueError,KeyError,TypeError):negatives+=1
  else:raise AssertionError('Invalid shadow/source evidence accepted')
 def gz(path,rows):
  with gzip.open(path,'wt',encoding='utf8') as f:
   for v in rows:f.write(json.dumps(v)+'\n')
 for mode in ('AS_NATIVE','AS_IID','AS_LHS'):
  S=root/mode;S.mkdir();kind=mode.removeprefix('AS_');rr=[];sequence=[];changed=0;counter=collections.Counter()
  for step in range(2):
   items=[];ids=[];group_ids=[0]*50+[4]*40+[8]*38;groups=[]
   for key in sorted(set(group_ids)):
    ix=[i for i,k in enumerate(group_ids) if k==key];groups.append(dict(group=key,count=len(ix),full_probability_strata=kind=='LHS'))
    for j,i in enumerate(ix):
     pid=[1,(1,3,4,6,8)[i%5],1+i%3,i,step*128+i];u=None if kind=='NATIVE' else (j+.35)/len(ix) if kind=='LHS' else (i+.25)/128;old=(i-64)/64;z=old if u is None else NormalDist().inv_cdf(u);changed+=int(z!=old);items.append(dict(sample_id=pid,group=key,group_size=len(ix),mode=kind,shadow_original=old,shadow_used=z,uniform=u,stratum=j if kind=='LHS' else None,replacement_applied=kind!='NATIVE'));ids.append(pid);counter[key]+=1
   rr.append(dict(step=4441+step,epoch=21,mode=kind,packets=128,groups=groups,records=items,sampling_seconds=.001));sequence.append(dict(role='L',ids=ids))
  gz(S/'training_sequence.jsonl.gz',sequence);cost=dict(channel=dict(strata=[dict(scene=('practical_low_suburban','practical_mid_urban','practical_low_urban')[key//3],state=('LOS','PARTIAL','BLOCKED')[key%3],count=n) for key,n in counter.items()],shadow_sampling=dict(mode=kind,changed=changed,strata_failures=0,sampling_seconds=.002),calls=2,packets=256,session_count=15,dropped=0,confidence_filter=False,ordinary_before_satellite=False,formal_test_changed=False))
  def write(rows):gz(S/'shadow_steps.jsonl.gz',rows)
  write(rr);v=r.shadow_records(S,mode,cost,expected_records=2);json.dumps(v,allow_nan=False);assert v['packets']==256;cases.append(dict(mode=kind,records=2))
  z=copy.deepcopy(rr);z[0]['records'][0]['sample_id'][-1]=-999;write(z);bad(lambda:r.shadow_records(S,mode,cost,2))
  z=copy.deepcopy(rr);z[0]['records'][0]['shadow_used']+=.1;write(z);bad(lambda:r.shadow_records(S,mode,cost,2))
  if kind=='LHS':z=copy.deepcopy(rr);z[0]['records'][0]['stratum']=4;write(z);bad(lambda:r.shadow_records(S,mode,cost,2))
  write(rr[:-1]);bad(lambda:r.shadow_records(S,mode,cost,2))
 # Build complete 90cell IDs and independently specified integer-count boundaries.
 B=root/'runs'/r.RUN;B.mkdir(parents=True);meta=dict(ids=[],TX=[],RX=[],day=[])
 for t in range(6):
  for rx in (1,3,4,6,8):
   for d in (1,2,3):
    for j in range(20):
     meta['ids'].append(f'{t}:{rx}:{d}:{j}');meta['TX'].append(t);meta['RX'].append(rx);meta['day'].append(d)
 def cm(n):
  c=[[0]*6 for _ in range(6)]
  for i in range(6):q=n//6+(i<n%6);c[i][i]=q;c[i][(i+1)%6]=300-q
  return c
 values={};counts={}
 for mode in r.MODES:
  rec=[]
  for view in r.VIEWS:
   for draw in range(1 if view=='clean' else 2):
    n=1700+(5 if mode=='AS_LHS' else 0) if view=='clean' else 1000+(9 if mode=='AS_LHS' else 0) if view=='practical_low_urban' else 1200
    rec.append(dict(view=view,draw=draw,count=1800,correct=n,confusion=cm(n),per_RX={str(rx):dict(count=360,correct=n//5+(i<n%5)) for i,rx in enumerate((1,3,4,6,8))}))
  metrics={v:dict(accuracy=sum(x['correct'] for x in rec if x['view']==v)/(1800*sum(x['view']==v for x in rec)),decisions=1800*sum(x['view']==v for x in rec)) for v in r.VIEWS}
  values[mode]=dict(status='SOURCE_QUALIFIED_FROZEN_READOUT_ONLY',source_V_unique=1800,source_cells=90,all_RX=[1,3,4,6,8],source_RX_holdout=None,CP=1,NN=23400,physical_channel_calls=21600,updates=0,teacher_NN=0,head_fit_updates=0,U_truth_read=False,target=False,model_state_unchanged=True,cache_loads=1,wall_seconds=0.,records=rec,metrics=metrics)
  P=B/(mode+'-s2026092701')/'source_qualification';P.mkdir(parents=True);(P/'source_ids.json').write_text(json.dumps(meta));(P/'result.json').write_text(json.dumps(values[mode]))
 expected={name+'_'+metric:True for name in ('AS_NATIVE','AS_IID') for metric in ('clean','midurban','lowurban','suburban')};expected['joint_clean']=True
 def saved(conditions):
  a=values['AS_LHS']['metrics'];comparisons={}
  for control in ('AS_NATIVE','AS_IID'):
   b=values[control]['metrics'];delta={v:100*(a[v]['accuracy']-b[v]['accuracy']) for v in r.VIEWS}
   def dd(view,k):return (next(x['correct'] for x in values['AS_LHS']['records'] if x['view']==view and x['draw']==k)-next(x['correct'] for x in values[control]['records'] if x['view']==view and x['draw']==k))/18
   comparisons[control]=dict(delta_pp=delta,urban_draw_delta_pp={v:[dd(v,k) for k in range(2)] for v in ('practical_mid_urban','practical_low_urban')},suburban_draw_delta_pp={v:[dd(v,k) for k in range(2)] for v in r.VIEWS[1:4]})
  (B/'source_matrix_decision.json').write_text(json.dumps(dict(conditions=conditions,comparisons=comparisons,joint_clean_delta_pp=100*(a['clean']['accuracy']-values['NOAS']['metrics']['clean']['accuracy']),source_pass=all(conditions.values()),target_access=False,decision_before_primary_formal_metric_read=True,decision='FIXTURE')))
 saved(expected);assert r.source_qualification(root)['source_pass'];cases.append(dict(source_integer_boundary='PASS',low_each_draw_correct_gain=9))
 path=B/'AS_LHS-s2026092701/source_qualification/result.json';z=copy.deepcopy(values['AS_LHS']);next(x for x in z['records'] if x['view']=='practical_low_urban' and x['draw']==0)['correct']-=1;path.write_text(json.dumps(z));bad(lambda:r.source_qualification(root))
 z=copy.deepcopy(values['AS_LHS']);q=next(x for x in z['records'] if x['view']=='practical_low_urban' and x['draw']==0);q.update(correct=1008,confusion=cm(1008),per_RX={str(rx):dict(count=360,correct=1008//5+(i<1008%5)) for i,rx in enumerate((1,3,4,6,8))});z['metrics']['practical_low_urban']['accuracy']=2017/3600;values['AS_LHS']=z;path.write_text(json.dumps(z));saved(expected);bad(lambda:r.source_qualification(root))
 expected['AS_NATIVE_lowurban']=False;expected['AS_IID_lowurban']=False;saved(expected);assert not r.source_qualification(root)['source_pass'];cases.append(dict(one_bad_low_draw='CLOSED'))
 return dict(status='PASS',shadow_fixture_records=6,negative_cases=negatives,cases=cases,IQ=0,CP=0,NN=0,updates=0,raw_truth_reads=0,target_access=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run(a.out),allow_nan=False))
