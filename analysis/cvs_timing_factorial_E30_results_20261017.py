"""Same-row independent four-factor analysis; no IQ/CP/model forward/training."""
import argparse,gzip,json,time
from collections import Counter
from pathlib import Path
import numpy as np
import cvs_fullsource_E30_results_20261017 as base
RUN='20261017-phase1-FcrossRX-timing-factorial-e30-s2026092701-m2-r01'
REFERENCE_RUN='20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01'
ORDER=('B00','B10','B01','B11')
PAIRS=(('B01','B00'),('B10','B00'),('B11','B00'),('B11','B10'),('B11','B01'))
base.ROWS={a:(RUN if a in ('B10','B11') else REFERENCE_RUN,a+'-s2026092701') for a in ORDER}
base.COLORS.update(B10='#16a34a',B11='#dc2626')
base.COMPARISON_PAIRS=PAIRS
PANEL=base.PROJECT/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
SOURCE_BASE=base.PROJECT/'runs/20261017-phase1-FcrossRX-fullsource-boundarymix-e30-s2026092701-m1-r01/source_boundary_mechanism_r01'
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
def roots():return {a:base.PROJECT/'runs'/run/row for a,(run,row) in base.ROWS.items()}
def factorial(values):
 if set(values)!=set(ORDER) or any(set(v)!=set(base.VIEWS) for v in values.values()):raise ValueError('All four arms and seven views required')
 if any(not isinstance(x,(float,int)) or not np.isfinite(x) or not 0<=x<=1 for v in values.values() for x in v.values()):raise ValueError('Finite accuracy probabilities required')
 return {v:100*(values['B11'][v]-values['B10'][v]-values['B01'][v]+values['B00'][v]) for v in base.VIEWS}
def local_trace(root,result):
 """Independently recount private bounded offsets; no waveform or CP reads."""
 import hashlib
 if not result['config']['Ag']:
  return dict(reused_control=True,raw_secondary_packets=852480,new_IQ_augment_views=0,original_control_config=True)
 packets=records=0;physical=set();counts=Counter();relative=[];power=0.;ordinary_mse=0.
 with gzip.open(root/'source/timing_views.jsonl.gz','rt',encoding='utf8') as f:
  for i,line in enumerate(f,1):
   r=json.loads(line);tag='trainL/epoch'+str((i-1)//222+1)+'/step'+str(i)+'/secondary';pids=r['physical_ids'];ids=r['ids'];info=r['meta']
   if r['step']!=i or r['view']!=tag or not r['Ag'] or len(pids)!=128 or len(set(ids))!=128:raise ValueError('Full ordered bounded timing trace required')
   occurrences={};expected=[]
   for pid in pids:
    fields=list(map(int,pid.split('/')))
    if len(fields)!=5 or fields[0]!=1 or fields[1] not in (1,3,4,6,8) or fields[2] not in (1,2,3):raise ValueError('Only original physical source L allowed')
    n=occurrences.get(pid,0);occurrences[pid]=n+1;expected.append(pid+'/occ'+str(n))
   if ids!=expected or any(len(info[k])!=128 for k in ('offset_samples','power_relative_error','relative_L2')) or info['seed']!=202610174101 or info['view']!=tag or info['timing_bound']!=16 or info['padding']!='reflect_no_wrap' or info['label_selection'] or info['full_As_reference_raw_dependency']!=[3,176]:raise ValueError('Fixed label-independent offset recipe required')
   offsets=[int.from_bytes(hashlib.blake2b(('202610174101\0'+tag+'\0'+pid).encode('utf8'),digest_size=8).digest(),'little')%33-16 for pid in ids]
   if offsets!=info['offset_samples']:raise ValueError('Private view/physical-occurrence offsets differ')
   counts.update(offsets);physical.update(pids);records+=1;packets+=128;relative.extend(info['relative_L2']);power=max(power,max(info['power_relative_error']));ordinary_mse+=r['ordinary_raw_MSE']
 if records!=6660 or packets!=852480 or len(physical)!=6300 or set(counts)!=set(range(-16,17)) or power>1e-6:raise ValueError('Complete timing coverage required')
 saved=result['timing_route']
 if saved['calls']!=records or saved['packets']!=packets or saved['unique_physical_L']!=len(physical) or saved['new_IQ_augment_views']!=packets or saved['offset_count']!=33 or abs(saved['relative_L2_mean']-float(np.mean(relative)))>1e-12 or saved['max_relative_power_error']!=power:raise ValueError('Independent route summary differs')
 return dict(reused_control=False,records=records,raw_secondary_packets=packets,IQ_augmented_packets=packets,physical_IDs=len(physical),offset_counts={str(k):v for k,v in sorted(counts.items())},relative_L2_quantiles=np.quantile(relative,[0,.1,.5,.9,1]).tolist(),max_relative_power_error=power,mean_ordinary_raw_MSE=ordinary_mse/records,private_offsets_exact=True,identity_preservation_proved=False,no_waveform_reads=True)
def same_local_stream(aa,bb):
 from itertools import zip_longest
 count=metadata=stats=offsets=0
 with gzip.open(aa/'source/timing_views.jsonl.gz','rt',encoding='utf8') as a,gzip.open(bb/'source/timing_views.jsonl.gz','rt',encoding='utf8') as b:
  for x,y in zip_longest(a,b):
   if x is None or y is None:raise ValueError('Two complete timing streams required')
   x=json.loads(x);y=json.loads(y);count+=1;metadata+=any(x[k]!=y[k] for k in ('step','view','ids','physical_ids'));stats+=x['meta']!=y['meta'];offsets+=x['meta']['offset_samples']!=y['meta']['offset_samples']
 return dict(records=count,physical_metadata_mismatches=metadata,numerical_stats_mismatches=stats,offset_mismatches=offsets)

def source_readout(out):
 data={};qualified={};role=base.read(PANEL/'data_roles.json');obs=base.read(PANEL/'observations/completion.json')
 if role['role']!='V' or role['V_panel']!=1800 or role['held_RX'] is not None or role['extra_Ldev'] or role['U_truth_used'] or role['target_access'] or obs['target_access'] or obs['draws']!=2 or obs['conditions'][0]!='full':raise ValueError('Fixed original source panel required')
 for a,root in roots().items():
  p=root/'source_qualification/complete.json'
  if not p.is_file():qualified[a]=base.read(root/'source_qualification_failure.json');continue
  q=base.read(p)
  if q['status']!=('SOURCE_QUALIFIED_TIMING_READOUT' if a in ('B10','B11') else 'SOURCE_QUALIFIED_READOUT') or q['NN']!=(16202 if a in ('B10','B11') else 12602) or q['training_updates'] or q['head_fit_updates'] or q['target_access'] or q['formal_scores_read'] or not q['state_unchanged']:raise ValueError('Frozen full-source source qualification required')
  qualified[a]=q
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as z:data[a]={k:z[k].copy() for k in z.files}
 if set(data)!=set(ORDER):
  base.write(out/'source_summary.json',dict(status='SOURCE_FACTORIAL_PARTIAL',source_qualification=qualified,missing=sorted(set(ORDER)-set(data)),NN=0,goal_complete=False));return dict(status='SOURCE_FACTORIAL_PARTIAL',source_qualification=qualified,rows=[],source_clean={})
 for a,x in data.items():
  q=qualified[a];acc=(x['satellite'].argmax(-1)==x['TX']).mean(-1)
  if not np.allclose(acc,q['source_accuracies'],atol=1e-12,rtol=0) or abs(float((x['clean'].argmax(1)==x['TX']).mean())-q['clean_accuracy'])>1e-12:raise ValueError('Independent source probabilities differ')
  if not np.array_equal(x['ids'],data['B00']['ids']) or not np.array_equal(x['ids'],np.asarray(role['physical_ids'])) or not np.array_equal(x['TX'],data['B00']['TX']) or x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Matched original finite source predictions required')
 with np.load(PANEL/'observations/physics.npz',allow_pickle=False) as p:physics={k:p[k].copy() for k in ('state','lock','signal_to_added_noise_db')}
 y=data['B00']['TX'];rx=data['B00']['RX'];day=data['B00']['day']
 with np.load(SOURCE_BASE/'B00/predictions.npz',allow_pickle=False) as p:
  if not np.array_equal(p['ids'],data['B00']['ids']) or not np.array_equal(p['TX'],y):raise ValueError('Fixed historical reference cohort differs')
  cohort=p['clean'].argmax(1)==y
 rows=[]
 for a,x in data.items():
  for si,scene in enumerate(SCENES):
   for dd in (-1,0,1):
    draws=(0,1) if dd==-1 else (dd,);yy=np.tile(y,len(draws));rr=np.tile(rx,len(draws));days=np.tile(day,len(draws));cc=np.tile(cohort,len(draws));z=np.concatenate([x['satellite'][d,si] for d in draws]);pred=z.argmax(1);old=np.concatenate([data['B01']['satellite'][d,si].argmax(1) for d in draws]);state=np.concatenate([physics['state'][d,si,0] for d in draws]);lock=np.concatenate([physics['lock'][d,si,0] for d in draws]);quality=np.concatenate([physics['signal_to_added_noise_db'][d,si,0] for d in draws]);others=z.copy();others[np.arange(len(yy)),yy]=-np.inf;margin=(z[np.arange(len(yy)),yy]-others.max(1))/30
    masks=[('overall','ALL',np.ones(len(yy),bool))]+[('RX',str(r),rr==r) for r in sorted(set(rx))]+[('day',str(d),days==d) for d in sorted(set(day))]+[('TX',str(t),yy==t) for t in range(6)]+[('state',s,state==k) for k,s in enumerate(('LOS','PARTIAL','BLOCKED'))]+[('lock',s,lock==k) for k,s in enumerate(('locked','degraded','unlocked'))]+[('state_lock',s+'/'+l,(state==k)&(lock==j)) for k,s in enumerate(('LOS','PARTIAL','BLOCKED')) for j,l in enumerate(('locked','degraded','unlocked'))]+[('added_noise_ratio',label,(quality>=lo)&(quality<hi)) for label,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]]
    for dim,value,mask in masks:
     if not mask.any():continue
     q=base.metric(yy[mask],pred[mask]);den=int((mask&cc).sum());rows.append(dict(arm=a,scene=scene,draw=dd,dimension=dim,stratum=value,**q,mean_margin_over30=float(margin[mask].mean()),fixed_P25_B00_clean_correct_count=den,retention_on_P25_B00_clean_correct=float((pred==yy)[mask&cc].mean()) if den else None,repairs_vs_B01=int(((pred==yy)&(old!=yy)&mask).sum()),breaks_vs_B01=int(((pred!=yy)&(old==yy)&mask).sum())))
 base.write_csv(out/'source_strata.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in rows]);summary=dict(status='SOURCE_FACTORIAL_READOUT' if set(data)==set(ORDER) else 'SOURCE_FACTORIAL_PARTIAL',rows=rows,source_clean={a:base.metric(y,x['clean'].argmax(1)) for a,x in data.items()},source_qualification=qualified,fixed_P25_B00_clean_correct_count=int(cohort.sum()),unique_source_V=1800,draws=2,source_observation='same original full reflect; no Ag at inference',added_noise_ratio_is_total_SNR=False,NN=0,IQ_reads=0,CP_reads=0,training_updates=0,target_feedback=False,crossseed_confirmed=False)
 base.write(out/'source_summary.json',summary);return summary

def source_plots(out,summary):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=summary['rows'];arms=[a for a in ORDER if a in summary['source_clean']];files=[]
 def row(a,scene,dim='overall',stratum='ALL'):return next(r for r in rows if (r['arm'],r['scene'],r['draw'],r['dimension'],r['stratum'])==(a,scene,-1,dim,stratum))
 def save(fig,name):
  fig.tight_layout()
  for suffix in ('png','pdf'):
   path=out/(name+'.'+suffix);fig.savefig(path,dpi=200);files.append(str(path))
  plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11.5,4.7));bins=('<3','3-8','8-15','>=15')
 for ax,scene,title in zip(axes,SCENES[1:],('Mid urban','Low urban')):
  for a in arms:ax.plot(range(4),[100*row(a,scene,'added_noise_ratio',b)['accuracy'] for b in bins],'o-',label=a,color=base.COLORS[a])
  counts=[row('B01',scene,'added_noise_ratio',b)['count'] for b in bins];ax.set(xticks=range(4),xticklabels=[b+' dB\nn='+str(n) for b,n in zip(bins,counts)],ylim=(0,100),title=title,ylabel='Source accuracy (%)',xlabel='Signal / added noise ratio; not total SNR');ax.legend(frameon=False);ax.grid(alpha=.2)
 fig.suptitle('Same fixed source V1800 / two channel draws / default head');save(fig,'source_added_noise_accuracy')
 fig,axes=plt.subplots(1,2,figsize=(13,5.1))
 for ax,scene,title in zip(axes,SCENES[1:],('Mid urban','Low urban')):
  strata=[r['stratum'] for r in rows if (r['arm'],r['scene'],r['draw'],r['dimension'])==('B01',scene,-1,'state_lock')];matrix=np.asarray([[100*row(a,scene,'state_lock',s)['accuracy'] for s in strata] for a in arms]);im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto');ax.set(yticks=range(len(arms)),yticklabels=arms,xticks=range(len(strata)),xticklabels=[s+'\nn='+str(row('B01',scene,'state_lock',s)['count']) for s in strata],title=title);ax.tick_params(axis='x',rotation=55)
  for j in range(len(arms)):
   for k in range(len(strata)):ax.text(k,j,f'{matrix[j,k]:.1f}',ha='center',va='center',fontsize=8,color='white' if matrix[j,k]>65 else 'black')
  fig.colorbar(im,ax=ax,fraction=.035,label='Source accuracy (%)')
 fig.suptitle('Source state/lock strata; tiny cells descriptive, not formal test states');save(fig,'source_state_lock_accuracy');return files

def independent_source_rule(out,sources):
 root=base.PROJECT/'runs'/RUN;saved=base.read(root/'source_matrix_decision.json')
 if not saved['source_only'] or saved['formal_scores_read'] or saved['automatic_seed_extension']:raise ValueError('Source rule must be fixed before formal consumption')
 if sources['status']!='SOURCE_FACTORIAL_READOUT':
  answer=dict(status='SOURCE_PARTIAL',source_replication_evidence=False,source_only=True,formal_truth_read=False,NN=0)
  base.write(out/'independent_source_decision.json',answer);return answer
 cv={a:base.read(r/'source/completion.json')['final_source_metrics']['source_val_accuracy'] for a,r in roots().items()};q=sources['source_qualification'];ref=base.read(root/'reference_timing/complete.json')
 if ref['status']!='REFERENCE_TIMING_SOURCE_COMPLETE' or ref['NN']!=3602 or ref['training_updates'] or ref['head_fit_updates'] or ref['target_access'] or ref['formal_scores_read'] or not ref['state_unchanged']:raise ValueError('Frozen original B00 timing reference required')
 with np.load(root/'reference_timing/predictions.npz',allow_pickle=False) as f:reference={k:f[k].copy() for k in f.files}
 data={'B00':reference}
 for arm in ('B10','B11'):
  with np.load(roots()[arm]/'source_qualification/predictions.npz',allow_pickle=False) as f:data[arm]={k:f[k].copy() for k in f.files}
 timing_rows=[]
 for arm,x in data.items():
  if x['timing'].shape!=(2,1800,6) or x['offset_samples'].shape!=(2,1800) or not np.isfinite(x['timing']).all() or not np.array_equal(x['ids'],reference['ids']) or not np.array_equal(x['TX'],reference['TX']) or not np.array_equal(x['offset_samples'],reference['offset_samples']):raise ValueError('Same finite private timing source views required')
  yy=np.tile(x['TX'],2);pred=x['timing'].argmax(-1).reshape(-1);offset=x['offset_samples'].reshape(-1);rr=np.tile(x['RX'],2)
  measured=(x['timing'].argmax(-1)==x['TX']).mean(-1)
  expected=ref['timing_accuracies'] if arm=='B00' else q[arm]['timing_accuracies']
  if not np.allclose(measured,expected,atol=1e-12,rtol=0):raise ValueError('Timing accuracy independent recount differs')
  for dimension,values in [('overall',['ALL']),('offset',list(range(-16,17))),('RX',sorted(set(rr)))]:
   for value in values:
    mask=np.ones(3600,bool) if dimension=='overall' else offset==value if dimension=='offset' else rr==value
    if mask.any():timing_rows.append(dict(arm=arm,dimension=dimension,stratum=str(value),**base.metric(yy[mask],pred[mask])))
 delta_t=100*(np.asarray(q['B10']['timing_accuracies'])-np.asarray(ref['timing_accuracies']));delta_u=100*(np.asarray(q['B11']['source_accuracies'])[:,1:]-np.asarray(q['B01']['source_accuracies'])[:,1:])
 checks=dict(Ag_clean_V_preserved=100*(cv['B10']-cv['B00'])>=-.25-1e-10,joint_clean_V_preserved=100*(cv['B11']-cv['B00'])>=-.25-1e-10,both_private_timing_draws_gain=bool((delta_t>=5.-1e-10).all()),all_original_urban_draws_gain=bool((delta_u>=1.-1e-10).all()))
 if checks!=saved['checks'] or all(checks.values())!=saved['source_replication_evidence'] or not np.allclose(delta_t,saved['timing_delta_pp'],rtol=0,atol=1e-12) or not np.allclose(delta_u,saved['urban_draw_delta_pp'],rtol=0,atol=1e-12):raise ValueError('Independent predeclared source decision differs')
 answer=dict(status='SOURCE_DECIDED',checks=checks,source_replication_evidence=all(checks.values()),clean_V=cv,timing_delta_pp=delta_t.tolist(),urban_draw_delta_pp=delta_u.tolist(),reference_timing_cost=ref,source_only=True,formal_truth_read=False,NN=0,goal_complete=False)
 base.write_csv(out/'source_timing_strata.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in timing_rows]);base.write(out/'source_timing_summary.json',dict(rows=timing_rows,reference_NN=3602,new_model_readout_NN=16202,NN=0,source_only=True))
 base.write(out/'independent_source_decision.json',answer);return answer

def timing_plot(out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=base.read(out/'source_timing_summary.json')['rows'];fig,ax=plt.subplots(figsize=(11,4.7))
 for arm in ('B00','B10','B11'):
  cells=sorted([r for r in rows if r['arm']==arm and r['dimension']=='offset'],key=lambda r:int(r['stratum']));ax.plot([int(r['stratum']) for r in cells],[100*r['accuracy'] for r in cells],'o-',label=arm,color=base.COLORS[arm],ms=3)
 ax.set(xlabel='Private source-only integer offset (samples)',ylabel='Default-head accuracy (%)',title='Source timing views / same V1800 x two draws / not target accuracy',ylim=(0,100));ax.legend(frameon=False);ax.grid(alpha=.2);fig.tight_layout();files=[]
 for suffix in ('png','pdf'):
  path=out/('source_timing.'+suffix);fig.savefig(path,dpi=200);files.append(str(path))
 plt.close(fig);return files

def native_telemetry(root):
 rows=[json.loads(line) for line in (root/'source/metrics_epoch.jsonl').read_text().splitlines() if line.strip()]
 if [r['epoch'] for r in rows]!=list(range(1,31)):raise ValueError('All30 native epoch records required')
 fields=['epoch','lr','train_grad_backbone','train_grad_before_clip','train_grad_total','train_grad_clip_limit','train_grad_clip_active','train_skipped_nonfinite_grad','train_skipped_nonfinite_loss','amp','from_scratch','label_epochs','pseudo_epochs','test_eval_ran','joint_guard_test_eval_ran','joint_guard_enabled','joint_guard_phase1_v2_tail_rollback_applied','joint_guard_phase1_v2_tail_early_stop_requested']
 selected=[{k:r.get(k) for k in fields} for r in rows]
 return dict(records=30,selected=selected,missing_fields=[k for k in fields if all(r.get(k) is None for r in rows)],per_packet_fusion_gates_saved=False,per_branch_gradients_saved=False)

def wait_for_run(pid):
 if type(pid) is not int or pid<=0:raise ValueError('Positive existing run dispatcher PID required')
 root=base.PROJECT/'runs'/RUN
 while not (root/'complete.json').is_file():
  if (root/'failure.json').is_file():raise RuntimeError('Run terminal failure preserved; no analysis or restart')
  process=Path('/proc')/str(pid)
  if not process.is_dir():raise RuntimeError('Registered dispatcher no longer live; reconcile run before analysis')
  command=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if 'cvs_timing_factorial35/runner.py' not in command:raise RuntimeError('Dispatcher handle no longer belongs to this run')
  time.sleep(15)

def main():
 tic=time.perf_counter()
 p=argparse.ArgumentParser();p.add_argument('--truth',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--wait-dispatcher',type=int);a=p.parse_args();rs=roots()
 if a.wait_dispatcher is not None:wait_for_run(a.wait_dispatcher)
 # Verify every requested row before any target truth access; no target consumer controls training.
 for root in rs.values():base.completed(root)
 out=a.output;out.mkdir(exist_ok=False);results={};private={};sources=source_readout(out);source_decision=independent_source_rule(out,sources)
 for name in ORDER:results[name],private[name]=base.collect_row(name,a.truth)
 if any(not np.array_equal(private['B00']['ids'],private[n]['ids']) for n in ORDER):raise ValueError('Matched formal physical IDs required')
 comparison={};sequences={};matched_cohort=private['B00']['pred']['clean']==private['B00']['y'];fixed_retention={};matched_retention={};external={};external_pred={}
 for ref in ('B00','B01'):
  rr=base.PROJECT/'runs'/base.RUN30/(ref+'-s2026092701');base.completed(rr)
  with np.load(rr/'prediction/predictions.npz',allow_pickle=False) as p:
   if not np.array_equal(p['ids'],private['B00']['ids']):raise ValueError('Direct ordinary-secondary reference IDs differ')
   external_pred[ref]={v:p[v].copy() for v in base.VIEWS}
  external[ref]={v:base.metric(private['B00']['y'],external_pred[ref][v]) for v in base.VIEWS}
 cohort=external_pred['B00']['clean']==private['B00']['y']
 for name,ref in PAIRS:
  key=name+'-'+ref;sequences[key]=base.compare_training_sequences(name,ref);comparison[key]={};y=private[name]['y']
  for view in base.VIEWS:
   cc=private[name]['pred'][view]==y;old=private[ref]['pred'][view]==y;comparison[key][view]=dict(accuracy_gain_pp=100*(cc.mean()-old.mean()),repairs=int((cc&~old).sum()),breaks=int((~cc&old).sum()))
 for name in ORDER:
  fixed_retention[name]={v:float((private[name]['pred'][v]==private[name]['y'])[cohort].mean()) for v in base.VIEWS}
  matched_retention[name]={v:float((private[name]['pred'][v]==private[name]['y'])[matched_cohort].mean()) for v in base.VIEWS}
 interaction=factorial({n:{v:results[n]['views'][v]['accuracy'] for v in base.VIEWS} for n in ORDER});flat=[q for r in results.values() for q in r['metrics']]
 for n in ORDER:results[n]['source']['timing_trace']=local_trace(rs[n],results[n]['source']['source_result']);results[n]['source']['native_full_telemetry']=native_telemetry(rs[n])
 local_pair=same_local_stream(rs['B10'],rs['B11']);base.write_csv(out/'per_row_seed.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in flat]);base.write_csv(out/'training_curves.csv',[dict(arm=n,**q) for n,r in results.items() for q in r['source']['curve']])
 shared_smoke=base.read(base.PROJECT/'runs'/RUN/'real_checkpoint_smoke.json')
 if shared_smoke['status']!='PASS' or shared_smoke['NN']!=2 or shared_smoke['real_IQ'] or shared_smoke['target_query'] or shared_smoke['training_updates']:raise ValueError('Shared compatibility-only smoke cost differs')
 run_launch=base.read(Path(str(base.PROJECT/'runs'/RUN)+'-launch')/'launch.json');run_wall_seconds=(base.PROJECT/'runs'/RUN/'complete.json').stat().st_mtime-run_launch['read_at']
 source_prerequisites={tag:base.read(base.PROJECT/'runs'/rr/'completion.json') for tag,rr in [('practical_attribution','20261017-diagnostic-FcrossRX-practical-attribution-e30-s2026092701-m2-r01'),('delay_phase','20261017-diagnostic-FcrossRX-delay-phase-e30-s2026092701-m2-r01')]}
 source_prerequisite_costs={tag:{k:r.get(k) for k in ('NN','wall_seconds','native_calls','updates','target_access')} for tag,r in source_prerequisites.items()}
 gaps={n:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in base.VIEWS[1:4]} for n,r in results.items()};costs={n:dict(r['source']['cost'],reused_control=n in ('B00','B01'),source_qualification_NN=sources['source_qualification'].get(n,{}).get('NN'),source_qualification_wall_seconds=sources['source_qualification'].get(n,{}).get('wall_seconds'),training_parent_epochs=0) for n,r in results.items()}
 base.write(out/'analysis_summary.json',dict(code_commit=a.commit,rows=results,paired_comparisons=comparison,interaction_accuracy_pp=interaction,interaction_macro_F1_pp=factorial({n:{v:results[n]['views'][v]['macro_f1'] for v in base.VIEWS} for n in ORDER}),source_decision=source_decision,old_ordinary_secondary_reference=external,joint_vs_old_ordinary_As_pp={v:100*(results['B11']['views'][v]['accuracy']-external['B01'][v]['accuracy']) for v in base.VIEWS},matched_B00_clean_correct_count=int(matched_cohort.sum()),matched_B00_clean_correct_retention=matched_retention,urban_mean_accuracy={n:sum(r['views'][v]['accuracy'] for v in base.VIEWS[4:])/3 for n,r in results.items()},suburban_clean_gaps_pp=gaps,fixed_P25_B00_clean_correct_count=int(cohort.sum()),fixed_P25_B00_clean_correct_satellite_retention=fixed_retention,training_sequences=sequences,timing_stream_pair=local_pair,costs=costs,shared_compatibility_smoke=shared_smoke,run_wall_seconds_from_launch_to_terminal=run_wall_seconds,source_prerequisite_costs=source_prerequisite_costs,direct_reference_P25_training_epochs=60,direct_reference_P25_training_updates=13320,own_training_parent_epochs=0,new_model_epochs=60,new_updates=13320,reused_control_epochs=60,reused_control_updates=13320,new_reference_timing_NN=3602,source_epoch_records=120,formal_confusion_matrices=len(flat),source_confusion_matrices=len(sources['rows'])+len(sources['source_clean']),NN=0,IQ_reads=0,CP_reads=0,training_updates=0,model_seed_count=1,crossseed_confirmed=False,Ag_proved=False,goal_complete=False,limitations=['Single model seed; two source channel/timing draws do not establish cross-seed reproduction','Formal per-packet state metadata unavailable; source strata are not formal target states','No per-packet fusion activation trace saved; gradient or parameter movement alone is not identification gain','Two new source readouts and one old B00 timing readout use CPU2/batch90; two original raw-secondary controls fully reused','U truth hidden; pseudo acceptance is not precision','Budget exhausted does not establish convergence; prior fixed-cycle Ag FAIL remains closed; P34 delay/boundary effect was rationale only, not proof of timing Ag']))
 files=base.plots(out,results)+((source_plots(out,sources)+timing_plot(out)) if sources['status']=='SOURCE_FACTORIAL_READOUT' else []);base.write(out/'figures.json',dict(files=files));base.write(out/'completion.json',dict(status='FACTORIAL_ANALYSIS_COMPLETE',rows=list(ORDER),model_seed_count=1,source_epochs_parsed=120,source_updates_parsed=26640,formal_confusion_matrices=len(flat),figures=len(files),NN=0,IQ_reads=0,CP_reads=0,new_updates=0,analysis_wall_seconds=time.perf_counter()-tic,code_commit=a.commit,crossseed_confirmed=False,goal_complete=False));print(json.dumps(dict(status='FACTORIAL_ANALYSIS_COMPLETE',rows=list(ORDER),out=str(out),figures=len(files),NN=0)))
if __name__=='__main__':main()
