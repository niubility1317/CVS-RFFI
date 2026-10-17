"""Synthetic carrier counters/source integer thresholds;zero IQ/CP/NN/truth."""
import argparse,copy,gzip,json,math
from pathlib import Path
from analysis import cvs_carrier_gate_recount_20261018 as r

def run(out):
 root=Path(out);root.mkdir(parents=True,exist_ok=False);negatives=0;cases=[]
 def bad(call):
  nonlocal negatives
  try:call()
  except (ValueError,KeyError,TypeError):negatives+=1
  else:raise AssertionError('Invalid carrier evidence accepted')
 for mode in ('NOAS_BATCH','AS_PACKET'):
  kind='PACKET' if mode.endswith('PACKET') else 'BATCH';S=root/mode;S.mkdir();rr=[]
  for i in range(2220):
   row=dict(call=i+1,mode=kind,packets=128)
   for n,p,m in [('phase',.12,2*math.pi),('cfo',.09,.0001)]:
    gate=i%10==0;k=(128 if gate else 0) if kind=='BATCH' else (15 if n=='phase' else 11)
    row.update({n+'_probability':p,n+'_maximum':m,n+'_native_gate':gate,n+'_selected':k,n+'_pairs_selected':k*(k-1)})
   rr.append(row)
  def write(v):
   with gzip.open(S/'carrier_steps.jsonl.gz','wt',encoding='utf8') as f:
    for x in v:f.write(json.dumps(x)+'\n')
  summary=dict(mode=kind,calls=2220,phase_calls=2220,cfo_calls=2220,packets=284160,new_IQ_operator=False,new_parameters=0,new_neural_forwards=0,labels_used=False,U_used=False,range_changed=False,probabilities_changed=False)
  for n in ('phase','cfo'):
   k=[x[n+'_selected'] for x in rr];summary[n]=dict(selected=sum(k),selected_pairs=sum(x*(x-1) for x in k),pair_denominator=2220*128*127,selected_count_min=min(k),selected_count_max=max(k),nontrivial_batches=sum(0<x<128 for x in k))
  cost=dict(carrier_organization=summary,one_shared_NoAs20_parent=True,not_new_IQ_Ag=True);write(rr);v=r.carrier_records(S,mode,cost);cases.append(dict(mode=kind,records=v['records']))
  for key,value in [('phase_pairs_selected',3),('cfo_probability',.08),('call',5),('mode','WRONG')]:
   z=copy.deepcopy(rr);z[1][key]=value;write(z);bad(lambda:r.carrier_records(S,mode,cost))
  write(rr[:-1]);bad(lambda:r.carrier_records(S,mode,cost))
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
    n=1700+(5 if mode.endswith('PACKET') else 0) if view=='clean' else 1000+(9 if mode=='AS_PACKET' else 0) if view=='practical_low_urban' else 1200
    rec.append(dict(view=view,draw=draw,count=1800,correct=n,confusion=cm(n),per_RX={str(rx):dict(count=360,correct=n//5+(i<n%5)) for i,rx in enumerate((1,3,4,6,8))}))
  metrics={v:dict(accuracy=sum(x['correct'] for x in rec if x['view']==v)/(1800*sum(x['view']==v for x in rec)),decisions=1800*sum(x['view']==v for x in rec)) for v in r.VIEWS}
  values[mode]=dict(status='SOURCE_QUALIFIED_FROZEN_READOUT_ONLY',source_V_unique=1800,source_cells=90,all_RX=[1,3,4,6,8],source_RX_holdout=None,CP=1,NN=23400,physical_channel_calls=21600,updates=0,teacher_NN=0,head_fit_updates=0,U_truth_read=False,target=False,model_state_unchanged=True,cache_loads=1,wall_seconds=0.,records=rec,metrics=metrics)
  P=B/(mode+'-s2026092701')/'source_qualification';P.mkdir(parents=True);(P/'source_ids.json').write_text(json.dumps(meta));(P/'result.json').write_text(json.dumps(values[mode]))
 expected=dict(NOAS_clean=True,AS_clean=True,joint_clean=True,midurban=True,lowurban=True,suburban=True)
 def saved(conditions):
  a=values['AS_PACKET']['metrics'];b=values['AS_BATCH']['metrics'];delta={v:100*(a[v]['accuracy']-b[v]['accuracy']) for v in r.VIEWS};interaction={v:100*(a[v]['accuracy']-values['NOAS_PACKET']['metrics'][v]['accuracy']-b[v]['accuracy']+values['NOAS_BATCH']['metrics'][v]['accuracy']) for v in r.VIEWS}
  (B/'source_matrix_decision.json').write_text(json.dumps(dict(conditions=conditions,AS_PACKET_vs_AS_BATCH_delta_pp=delta,source_pass=all(conditions.values()),target_access=False,decision_before_primary_formal_metric_read=True,decision='FIXTURE',organization_x_As_interaction_pp=interaction)))
 saved(expected);assert r.source_qualification(root)['source_pass'];cases.append(dict(source_integer_boundary='PASS',NOAS_clean_correct_gain=5,low_each_draw_gain=9))
 path=B/'AS_PACKET-s2026092701/source_qualification/result.json';z=copy.deepcopy(values['AS_PACKET']);next(x for x in z['records'] if x['view']=='practical_low_urban' and x['draw']==0)['correct']-=1;path.write_text(json.dumps(z));bad(lambda:r.source_qualification(root))
 # A saved claimed PASS cannot mask one insufficient lowurban draw,even with consistent CM/mean.
 z=copy.deepcopy(values['AS_PACKET']);q=next(x for x in z['records'] if x['view']=='practical_low_urban' and x['draw']==0);q.update(correct=1008,confusion=cm(1008),per_RX={str(rx):dict(count=360,correct=1008//5+(i<1008%5)) for i,rx in enumerate((1,3,4,6,8))});z['metrics']['practical_low_urban']['accuracy']=2017/3600;values['AS_PACKET']=z;path.write_text(json.dumps(z));saved(expected);bad(lambda:r.source_qualification(root))
 expected['lowurban']=False;saved(expected);assert not r.source_qualification(root)['source_pass'];cases.append(dict(one_bad_low_draw='CLOSED'))
 return dict(status='PASS',carrier_fixture_records=4440,negative_cases=negatives,cases=cases,IQ=0,CP=0,NN=0,updates=0,raw_truth_reads=0,target_access=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run(a.out)))
