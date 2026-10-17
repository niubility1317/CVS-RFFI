"""Independent complete source-artifact recount;no model/IQ or target access."""
import json,math
from collections import Counter
from pathlib import Path
from . import design as d
OUT=d.ROOT/d.LOCAL/'final'
VIEWS=('raw','direction','permuted')
FIT=(3,4,6,8)
BUDGET=dict(folds=12,reference_packets_per_fold=2520,held_packets_per_fold=420,reference_forward_packets=60480,input_backward_packets=30240,input_backward_calls=240,source_eval_packets=15120,noquery_smoke_packets=2,total_forward_packets=75602,optimizer_updates=0)

def counts(y,p,take=None):
 take=list(range(len(y))) if take is None else list(take)
 assert take and len(y)==len(p)
 cm=[[0]*6 for _ in range(6)]
 for i in take:
  assert type(y[i]) is int and type(p[i]) is int and y[i] in range(6) and p[i] in range(6)
  cm[y[i]][p[i]]+=1
 return dict(count=len(take),correct=sum(cm[t][t] for t in range(6)),accuracy=sum(cm[t][t] for t in range(6))/len(take),macro_F1=sum(2*cm[t][t]/(sum(cm[t])+sum(v[t] for v in cm)) if sum(cm[t])+sum(v[t] for v in cm) else 0. for t in range(6))/6,confusion=cm)
def paired(y,raw,p,take):
 take=list(take);base=[i for i in take if raw[i]==y[i]]
 assert base
 repair=sum(raw[i]!=y[i] and p[i]==y[i] for i in take);broken=sum(raw[i]==y[i] and p[i]!=y[i] for i in take)
 return dict(count=len(take),raw_correct=len(base),retained=len(base)-broken,retention=(len(base)-broken)/len(base),repair=repair,broken=broken,net_correct=repair-broken,delta_accuracy_pp=100*(repair-broken)/len(take))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11),(a,b)
def analyze():
 result=d.read(OUT/'result.json');complete=d.read(OUT/'completion.json');resolved=d.read(OUT/'resolved_config.json');smoke=d.read(OUT/'real_checkpoint_smoke.json');matrix=d.read(OUT/'folds.json');ids=d.read(OUT/'fit_L_ids.json')
 assert resolved['config']==d.config() and resolved['commit']==(OUT/'release_commit.txt').read_text().strip()
 assert resolved['U_truth_read'] is False and resolved['held_RX1_used_in_math'] is False and resolved['target_access'] is False
 assert smoke['status']=='PASS' and smoke['strict_load'] and smoke['matching_own_source_final20'] and smoke['synthetic_NN_packets']==2 and smoke['source_IQ_reads']==0 and smoke['target_access'] is False
 assert not (OUT/'failure.json').exists() and len(matrix)==12 and len(ids)==len(set(ids))==5040
 files=sorted((OUT/'predictions').glob('RX*_day*.json'));assert len(files)==12
 y=[None]*5040;rx=[None]*5040;days=[None]*5040;p={v:[None]*5040 for v in VIEWS};zs={v:[None]*5040 for v in VIEWS};rows=[];vectors=[]
 refs=back=backcalls=evals=0;identifiable=0
 for fold in matrix:
  r,day=fold['RX'],fold['day'];row=d.read(OUT/'predictions'/('RX%d_day%d.json'%(r,day)));held=fold['held'];reference=fold['reference']
  assert row['RX']==r and row['day']==day and len(held)==len(set(held))==row['held_count']==420 and len(reference)==len(set(reference))==row['reference_count']==2520 and not set(held)&set(reference)
  assert row['held_indices']==held and row['physical_ids']==[ids[i] for i in held] and len(row['labels'])==420
  assert Counter(row['labels'])==Counter({i:70 for i in range(6)}) and set(row['views'])==set(VIEWS)
  op=row['operator'];assert op['reference_packets']==2520 and op['eval_forward_packets']==5040 and op['input_backward_packets']==2520 and op['input_backward_calls']==20
  assert op['parameter_optimizer_updates']==op['teacher_copies']==op['new_training_loss_terms']==0 and op['labels_consumed'] is False and op['one_shared_direction'] is True
  refs+=row['reference_forward_packets'];back+=row['input_backward_packets'];backcalls+=row['input_backward_calls'];identifiable+=int(op['identifiable'])
  assert row['reference_forward_packets']==5040 and row['input_backward_packets']==2520 and row['input_backward_calls']==20
  a,b=row['direction'],row['permuted'];assert len(a)==len(b)==2 and all(len(v)==256 for v in a+b) and all(math.isfinite(z) for v in a+b for z in v)
  rms=math.sqrt(sum(z*z for v in a for z in v)/256);peak=max(abs(z) for v in a for z in v);crms=math.sqrt(sum(z*z for v in b for z in v)/256);cpeak=max(abs(z) for v in b for z in v)
  assert all(sorted(a[c])==sorted(b[c]) for c in range(2));close(rms,row['direction_RMS']);close(crms,row['control_RMS']);close(peak,row['direction_peak']);close(cpeak,row['control_peak'])
  assert rms<=.001*(1+1e-6) and peak<=.003*(1+1e-6) and max(abs(sum(v)/256) for v in a+b)<=1e-9
  assert op['identifiable']==(op['gradient_RMS']>1e-24) and (rms>0)==op['identifiable']
  vectors.append(dict(RX=r,day=day,RMS=rms,peak=peak,gradient_RMS=op['gradient_RMS'],probe_KL_mean=op['probe_KL_mean'],seconds=row['seconds']))
  for j,i in enumerate(held):
   assert y[i] is None;y[i]=row['labels'][j];rx[i]=r;days[i]=day
  for view in VIEWS:
   v=row['views'][view];assert len(v['predictions'])==len(v['logits'])==420 and v['raw_complex_power_min']>0 and math.isfinite(v['relative_added_RMS_max'])
   for j,i in enumerate(held):
    z=v['logits'][j];assert len(z)==6 and all(math.isfinite(x) for x in z) and v['predictions'][j]==max(range(6),key=z.__getitem__) and p[view][i] is None
    p[view][i]=v['predictions'][j];zs[view][i]=z
   evals+=420
  assert all(row['views'][v]['raw_complex_power_min']==row['views']['raw']['raw_complex_power_min'] and row['views'][v]['raw_complex_power_max']==row['views']['raw']['raw_complex_power_max'] for v in VIEWS)
  rows.append(row)
 assert all(v is not None for v in y+rx+days) and Counter(zip(y,rx,days))==Counter({(t,r,day):70 for t in range(6) for r in FIT for day in (1,2,3)})
 for fold in matrix:
  assert fold['held']==[i for i in range(5040) if rx[i]==fold['RX'] and days[i]==fold['day']]
  assert fold['reference']==[i for i in range(5040) if rx[i]!=fold['RX'] and days[i]!=fold['day']]
 scores={};strata={};pairs={}
 for view in VIEWS:
  a=counts(y,p[view]);saved=result['scores'][view];assert a['confusion']==saved['confusion'];close(a['accuracy'],saved['accuracy']);close(a['macro_F1'],saved['macro_F1'])
  pair=paired(y,p['raw'],p[view],range(5040));close(pair['retention'],saved['retention']);assert pair['raw_correct']==saved['raw_correct_count']
  a['mean_CE']=sum(max(z)+math.log(sum(math.exp(v-max(z)) for v in z))-z[t] for t,z in zip(y,zs[view]))/5040;scores[view]=a;pairs[view]={'overall':pair};strata[view]={}
  for name,values,allowed in (('RX',rx,FIT),('TX',y,range(6)),('day',days,(1,2,3))):
   strata[view][name]={};pairs[view][name]={}
   for v in allowed:
    take=[i for i,z in enumerate(values) if z==v];s=counts(y,p[view],take);pa=paired(y,p['raw'],p[view],take);strata[view][name][str(v)]=s;pairs[view][name][str(v)]=pa
    if name=='RX':close(s['accuracy'],saved['RX_accuracy'][str(v)]);close(pa['retention'],saved['RX_retention'][str(v)])
    if name=='TX':close(pa['retention'],saved['TX_retention'][str(v)])
    if name=='day':assert s['confusion']==saved['by_day'][str(v)]['confusion'];close(s['macro_F1'],saved['by_day'][str(v)]['macro_F1'])
  strata[view]['cells']=[dict(RX=r,day=day,TX=t,**counts(y,p[view],[i for i in range(5040) if (rx[i],days[i],y[i])==(r,day,t)])) for r in FIT for day in (1,2,3) for t in range(6)]
 gap=100*(scores['permuted']['accuracy']-scores['direction']['accuracy']);gaps={str(r):100*(strata['permuted']['RX'][str(r)]['accuracy']-strata['direction']['RX'][str(r)]['accuracy']) for r in FIT};eps=1e-10
 actual=dict(BUDGET,reference_forward_packets=refs,input_backward_packets=back,input_backward_calls=backcalls,source_eval_packets=evals,total_forward_packets=2+refs+evals)
 assert actual==BUDGET and all(result['execution'][k]==v for k,v in actual.items()) and result['execution']['identifiable_folds']==identifiable
 conditions=dict(raw_accuracy=scores['raw']['accuracy']>=.98-eps,directional_gap=gap>=2-eps,transfer_RX_count=sum(v>=1-eps for v in gaps.values())>=3,directional_retention=pairs['direction']['overall']['retention']>=.95-eps,directional_each_RX_retention=min(v['retention'] for v in pairs['direction']['RX'].values())>=.95-eps,directional_each_TX_retention=min(v['retention'] for v in pairs['direction']['TX'].values())>=.95-eps,permuted_retention=pairs['permuted']['overall']['retention']>=.99-eps,permuted_each_RX_retention=min(v['retention'] for v in pairs['permuted']['RX'].values())>=.98-eps,all12_directions_identifiable=identifiable==12,complete_budget=actual==BUDGET,no_target=result['execution']['target_access'] is False)
 decision='PASS_SENSITIVITY_ONLY' if all(conditions.values()) else 'FAIL_FIXED_SHARED_DIRECTION_CLOSED'
 assert conditions==result['qualification']['conditions'] and all(conditions.values())==result['qualification']['supported_for_separate_followup'] and decision==complete['scientific_decision']
 assert complete['folds']==12 and complete['source_decisions']==15120 and complete['actual_NN_packets']==75602 and complete['input_backward_packets']==30240 and complete['input_backward_calls']==240 and complete['automatic_training'] is False
 for k in ('actual_optimizer_updates','head_fit_updates','teacher_forwards','new_training_loss_terms','As_generations'):assert result[k]==0
 for k in ('U_used','V_used','held_RX1_used','target_access'):assert result[k] is False
 assert result['model_state_unchanged'] and result['default_head_only'] and result['NoExplicitCFO']
 log=(OUT/'shared_source.stdout.log').read_text();assert 'SMOKE PASS matching real checkpoint' in log and 'INPUT PASS' in log and log.count('FOLD RX')==12 and 'COMPLETE ' in log and 'Traceback (most recent call last)' not in log
 summary=dict(status='ANALYZED_SOURCE_ONLY_VERIFIED',run_id=d.RUN,commit=resolved['commit'],decision=decision,conditions=conditions,scores=scores,strata=strata,paired=pairs,directional_gap_pp=gap,per_RX_gap_pp=gaps,vectors=vectors,complete_logit_decisions=15120,source_L_unique=5040,cells=216,execution=actual,elapsed_seconds=result['elapsed_seconds'],optimizer_updates=0,new_teacher_models=0,head_fit_updates=0,As_generations=0,U_used=False,V_used=False,held_RX1_used=False,target_access=False,Ag_efficacy_proved=False,identity_preservation_proved=False,joint_goal_complete=False,automatic_training=False)
 d.write(OUT/'analysis_summary.json',summary)
 print(json.dumps({k:summary[k] for k in ('status','decision','conditions','directional_gap_pp','per_RX_gap_pp','execution','elapsed_seconds')},indent=2));return summary

def figures(a):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout='constrained');colors=('#306b9c','#bf6434','#66894d')
 for view,offset,color in zip(VIEWS,(-.25,0,.25),colors):ax[0].bar([i+offset for i in range(4)],[100*a['strata'][view]['RX'][str(r)]['accuracy'] for r in FIT],.24,label=view,color=color)
 ax[0].set(xticks=range(4),xticklabels=FIT,xlabel='Source RX',ylabel='Frozen default-head fit-L accuracy (%)',ylim=(90,100));ax[0].legend(frameon=False)
 vals=[a['per_RX_gap_pp'][str(r)] for r in FIT];ax[1].bar(range(4),vals,color='#bf6434');ax[1].axhline(1,ls='--',color='#666',label='Fixed per-RX 1 pp requirement');ax[1].set(xticks=range(4),xticklabels=FIT,xlabel='Source RX',ylabel='Permuted minus direction accuracy (pp)');ax[1].legend(frameon=False)
 fig.suptitle('F+CROSS_RX / shared additive IQ / '+a['decision']+'\nFrozen source sensitivity; no training, no target, not Ag efficacy')
 for suffix in ('png','pdf'):fig.savefig(OUT/('source_direction_results.'+suffix),dpi=180)
 plt.close(fig)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--figures',action='store_true',help='Also draw PNG/PDF;requires existing matplotlib environment')
 args=parser.parse_args();a=analyze()
 if args.figures:figures(a)
