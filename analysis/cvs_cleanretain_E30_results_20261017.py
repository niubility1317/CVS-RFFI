"""Independent R25/C0/AS50 full-source E30 analysis;no IQ/CP/NN/updates."""
import argparse,gzip,json,math,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import cvs_fullsource_E30_results_20261017 as base
import cvs_timing_actual_augmentation_20261017 as telemetry
RUN='20261017-phase1-FcrossRX-cleanretain-e30-s2026092701-m1-r01';ROOT=base.PROJECT/'runs'/RUN
ORDER=('C0','AS50','R25');PAIRS=(('AS50','C0'),('R25','C0'),('R25','AS50'))
base.ROWS=dict(C0=('20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01','B00-s2026092701'),AS50=('20261017-phase1-FcrossRX-cycleorder-factorial-e30-s2026092701-m4-r01','B01-s2026092701'),R25=(RUN,'R25-s2026092701'))
base.COLORS.update(C0='#64748b',AS50='#2563eb',R25='#16a34a');base.COMPARISON_PAIRS=PAIRS
PANEL=base.PROJECT/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
def roots():return {a:base.PROJECT/'runs'/run/row for a,(run,row) in base.ROWS.items()}
read,write,write_csv=base.read,base.write,base.write_csv
def physics_summary(root,result):
 expected_per_step=64 if root.name.startswith('R25-') else 128
 if not result['config']['As']:return dict(satellite_packets=0,not_applicable=True)
 stats=defaultdict(list);counts=Counter();bins=Counter();packets=batches=0
 bounds=(-math.inf,0,5,10,15,20,math.inf);labels=('<0','0-5','5-10','10-15','15-20','>=20')
 with gzip.open(root/'source/training_physics.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   q=json.loads(line);batches+=1
   if q['step']!=batches or len(q['records'])!=expected_per_step:raise ValueError('Training physical metadata incomplete')
   if expected_per_step==64 and q['positions']!=[i for i in range(128) if (i+batches)%2==1]:raise ValueError('Exact alternating64 physical slots required')
   for r in q['records']:
    packets+=1;quality=r['quality_snr_db'];actual=r['signal_to_added_noise_db']
    bi=next(i for i in range(6) if bounds[i]<=quality<bounds[i+1])
    bins[(q['scene'],r['state_end'],r['receiver_lock_status'],labels[bi])]+=1
    for key in ['quality_source','boundary','noise_boundary','receiver_lock_status','state_end','agc_limited','true_total_snr_known','channel_equalization_applied']:
     counts[(key,str(r[key]))]+=1
    for key in ['quality_snr_db','signal_to_added_noise_db','output_frequency_hz','residual_kernel_length','residual_kernel_energy','numerical_filter_latency_samples','input_original_rms','agc_gain_db']:
     value=r[key]
     if not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Nonfinite physical metadata '+key)
     stats[key].append(value)
    stats['actual_added_noise_ratio_minus_white_proxy_db'].append(actual-quality)
    stats['input_effective_energy_samples'].append(r['input_metrics']['effective_energy_samples'])
    stats['output_effective_energy_samples'].append(r['output_metrics']['effective_energy_samples'])
 if (packets,batches)!=(6660*expected_per_step,6660) or sum(x['count'] for x in result['satellite_coverage'])!=packets:raise ValueError('Training physics budget mismatch')
 summary={}
 for key,values in stats.items():
  x=np.asarray(values,dtype=float);summary[key]=dict(count=len(x),mean=float(x.mean()),std=float(x.std()),quantiles=dict(zip(['min','q01','q10','q50','q90','q99','max'],np.quantile(x,[0,.01,.1,.5,.9,.99,1]).tolist())))
 return dict(satellite_packets=packets,batches=batches,statistics=summary,counts=[dict(field=k[0],value=k[1],count=v) for k,v in sorted(counts.items())],quality_bins=[dict(scene=k[0],state=k[1],lock=k[2],quality_proxy_db_bin=k[3],count=v) for k,v in sorted(bins.items())],TX_RX_scene_cells=len({(x['TX'],x['RX'],x['scene']) for x in result['satellite_coverage']}),TX_RX_state_cells=len({(x['TX'],x['RX'],x['state']) for x in result['satellite_coverage']}),true_total_SNR_identified=False,causal_state_effect_identified=False,boundary_accuracy_effect_identified=False,formal_state_accuracy_identified=False)
def source_readout(out):
 data={};qualified={};role=base.read(PANEL/'data_roles.json');obs=base.read(PANEL/'observations/completion.json')
 if role['role']!='V' or role['V_panel']!=1800 or role['held_RX'] is not None or role['extra_Ldev'] or role['U_truth_used'] or role['target_access'] or obs['target_access'] or obs['draws']!=2 or obs['conditions'][0]!='full':raise ValueError('Fixed original source panel required')
 for a,root in roots().items():
  folder=ROOT/'reference_source' if a=='C0' else root/'source_qualification';p=folder/'complete.json'
  if not p.is_file():qualified[a]=base.read(root/'source_qualification_failure.json');continue
  q=base.read(p)
  if q['status']!='SOURCE_QUALIFIED_READOUT' or q['NN']!=12602 or q['training_updates'] or q['head_fit_updates'] or q['target_access'] or q['formal_scores_read'] or not q['state_unchanged']:raise ValueError('Frozen full-source source qualification required')
  qualified[a]=q
  with np.load(folder/'predictions.npz',allow_pickle=False) as z:data[a]={k:z[k].copy() for k in z.files}
 if set(data)!=set(ORDER):
  base.write(out/'source_summary.json',dict(status='SOURCE_THREE_ROW_PARTIAL',source_qualification=qualified,missing=sorted(set(ORDER)-set(data)),NN=0,goal_complete=False));return dict(status='SOURCE_FACTORIAL_PARTIAL',source_qualification=qualified,rows=[],source_clean={})
 for a,x in data.items():
  q=qualified[a];acc=(x['satellite'].argmax(-1)==x['TX']).mean(-1)
  if not np.allclose(acc,q['source_accuracies'],atol=1e-12,rtol=0) or abs(float((x['clean'].argmax(1)==x['TX']).mean())-q['clean_accuracy'])>1e-12:raise ValueError('Independent source probabilities differ')
  if not np.array_equal(x['ids'],data['C0']['ids']) or not np.array_equal(x['ids'],np.asarray(role['physical_ids'])) or not np.array_equal(x['TX'],data['C0']['TX']) or x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Matched original finite source predictions required')
 with np.load(PANEL/'observations/physics.npz',allow_pickle=False) as p:physics={k:p[k].copy() for k in ('state','lock','signal_to_added_noise_db')}
 y=data['C0']['TX'];rx=data['C0']['RX'];day=data['C0']['day']
 cohort=data['C0']['clean'].argmax(1)==y
 rows=[]
 for a,x in data.items():
  for si,scene in enumerate(SCENES):
   for dd in (-1,0,1):
    draws=(0,1) if dd==-1 else (dd,);yy=np.tile(y,len(draws));rr=np.tile(rx,len(draws));days=np.tile(day,len(draws));cc=np.tile(cohort,len(draws));z=np.concatenate([x['satellite'][d,si] for d in draws]);pred=z.argmax(1);old=np.concatenate([data['AS50']['satellite'][d,si].argmax(1) for d in draws]);state=np.concatenate([physics['state'][d,si,0] for d in draws]);lock=np.concatenate([physics['lock'][d,si,0] for d in draws]);quality=np.concatenate([physics['signal_to_added_noise_db'][d,si,0] for d in draws]);others=z.copy();others[np.arange(len(yy)),yy]=-np.inf;margin=(z[np.arange(len(yy)),yy]-others.max(1))/30
    masks=[('overall','ALL',np.ones(len(yy),bool))]+[('RX',str(r),rr==r) for r in sorted(set(rx))]+[('day',str(d),days==d) for d in sorted(set(day))]+[('TX',str(t),yy==t) for t in range(6)]+[('state',s,state==k) for k,s in enumerate(('LOS','PARTIAL','BLOCKED'))]+[('lock',s,lock==k) for k,s in enumerate(('locked','degraded','unlocked'))]+[('state_lock',s+'/'+l,(state==k)&(lock==j)) for k,s in enumerate(('LOS','PARTIAL','BLOCKED')) for j,l in enumerate(('locked','degraded','unlocked'))]+[('added_noise_ratio',label,(quality>=lo)&(quality<hi)) for label,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]]
    for dim,value,mask in masks:
     if not mask.any():continue
     q=base.metric(yy[mask],pred[mask]);den=int((mask&cc).sum());rows.append(dict(arm=a,scene=scene,draw=dd,dimension=dim,stratum=value,**q,mean_margin_over30=float(margin[mask].mean()),fixed_C0_clean_correct_count=den,retention_on_C0_clean_correct=float((pred==yy)[mask&cc].mean()) if den else None,repairs_vs_AS50=int(((pred==yy)&(old!=yy)&mask).sum()),breaks_vs_AS50=int(((pred!=yy)&(old==yy)&mask).sum())))
 base.write_csv(out/'source_strata.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in rows]);summary=dict(status='SOURCE_THREE_ROW_READOUT' if set(data)==set(ORDER) else 'SOURCE_FACTORIAL_PARTIAL',rows=rows,source_clean={a:base.metric(y,x['clean'].argmax(1)) for a,x in data.items()},source_qualification=qualified,fixed_C0_clean_correct_count=int(cohort.sum()),unique_source_V=1800,draws=2,source_observation='same original full reflect; no Ag at inference',added_noise_ratio_is_total_SNR=False,NN=0,IQ_reads=0,CP_reads=0,training_updates=0,target_feedback=False,crossseed_confirmed=False)
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
  counts=[row('AS50',scene,'added_noise_ratio',b)['count'] for b in bins];ax.set(xticks=range(4),xticklabels=[b+' dB\nn='+str(n) for b,n in zip(bins,counts)],ylim=(0,100),title=title,ylabel='Source accuracy (%)',xlabel='Signal / added noise ratio; not total SNR');ax.legend(frameon=False);ax.grid(alpha=.2)
 fig.suptitle('Same fixed source V1800 / two channel draws / default head');save(fig,'source_added_noise_accuracy')
 fig,axes=plt.subplots(1,2,figsize=(13,5.1))
 for ax,scene,title in zip(axes,SCENES[1:],('Mid urban','Low urban')):
  strata=[r['stratum'] for r in rows if (r['arm'],r['scene'],r['draw'],r['dimension'])==('AS50',scene,-1,'state_lock')];matrix=np.asarray([[100*row(a,scene,'state_lock',s)['accuracy'] for s in strata] for a in arms]);im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto');ax.set(yticks=range(len(arms)),yticklabels=arms,xticks=range(len(strata)),xticklabels=[s+'\nn='+str(row('AS50',scene,'state_lock',s)['count']) for s in strata],title=title);ax.tick_params(axis='x',rotation=55)
  for j in range(len(arms)):
   for k in range(len(strata)):ax.text(k,j,f'{matrix[j,k]:.1f}',ha='center',va='center',fontsize=8,color='white' if matrix[j,k]>65 else 'black')
  fig.colorbar(im,ax=ax,fraction=.035,label='Source accuracy (%)')
 fig.suptitle('Source state/lock strata; tiny cells descriptive, not formal test states');save(fig,'source_state_lock_accuracy');return files


base.physics_summary=physics_summary

def source_rule(out,summary):
 saved=read(ROOT/'source_matrix_decision.json')
 if not saved['source_only'] or saved['formal_scores_read'] or saved['automatic_seed_extension']:raise ValueError('Source-only decision required before independent formal analysis')
 if summary['status']!='SOURCE_THREE_ROW_READOUT':raise ValueError('Full source readout required for complete analysis')
 cv={a:read(r/'source/completion.json')['final_source_metrics']['source_val_accuracy'] for a,r in roots().items() if a in ('C0','R25')};q=summary['source_qualification'];delta=100*(np.asarray(q['R25']['source_accuracies'])[:,1:]-np.asarray(q['AS50']['source_accuracies'])[:,1:]);clean_delta=100*(cv['R25']-cv['C0']);checks=dict(clean_preserved=clean_delta>=-.25-1e-10,each_urban_draw_preserved=bool((delta>=-2.-1e-10).all()))
 if checks!=saved['checks'] or all(checks.values())!=saved['source_protection_evidence'] or not np.allclose(delta,saved['urban_delta_vs_AS50_pp'],rtol=0,atol=1e-12) or abs(clean_delta-saved['clean_net_pp'])>1e-12:raise ValueError('Independent source rule mismatch')
 value=dict(status='SOURCE_DECIDED',checks=checks,source_protection_evidence=all(checks.values()),clean_V=cv,clean_net_pp=clean_delta,urban_delta_vs_AS50_pp=delta.tolist(),NN=0,source_only=True,formal_truth_read=False,automatic_seed_extension=False,goal_complete=False);write(out/'independent_source_decision.json',value);return value

def parts_curve(root):
 sums=defaultdict(lambda:defaultdict(float));counts=Counter();keys=('clean_CE','second_CE','clean_KL','second_KL','satellite_CE','secondary_ordinary_CE','satellite_KL','secondary_ordinary_KL');max_error=0.
 with gzip.open(root/'source/SupCon_steps.jsonl.gz','rt',encoding='utf8') as f:
  for i,line in enumerate(f,1):
   r=json.loads(line);p=r['classification_terms'];e=(i-1)//222+1
   if r['step']!=i or r['satellite_L']!=64 or r['ordinary_weight']!=.75 or r['satellite_weight']!=.25 or set(p)!=set(keys) or not all(math.isfinite(p[k]) for k in keys):raise ValueError('Actual all6660 clean/satellite objective components required')
   error=max(abs(p['second_CE']-.5*(p['satellite_CE']+p['secondary_ordinary_CE'])),abs(p['second_KL']-.5*(p['satellite_KL']+p['secondary_ordinary_KL'])));max_error=max(error,max_error)
   if error>2e-5:raise ValueError('Actual normalized .75ordinary/.25As objective mismatch')
   for k in keys:sums[e][k]+=p[k]
   counts[e]+=1
 if set(counts)!=set(range(1,31)) or set(counts.values())!={222}:raise ValueError('All30 complete objective curves required')
 return [dict(epoch=e,**{k:sums[e][k]/222 for k in keys}) for e in range(1,31)],max_error

def telemetry_plots(out,curves,moves,parts):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 files=[]
 def save(fig,name):
  fig.tight_layout()
  for suffix in ('png','pdf'):
   p=out/(name+'.'+suffix);fig.savefig(p,dpi=200);files.append(str(p))
  plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13.5,4))
 for k in curves['C0'][0]:
  if k.startswith('p_'):axes[0].plot(range(1,31),[r[k] for r in curves['C0']],label=k[2:])
 for arm,rr in curves.items():
  axes[1].plot(range(1,31),[r['cumulative_waveform_MSE'] for r in rr],label=arm,color=base.COLORS[arm]);axes[2].plot(range(1,31),[r['cumulative_max_abs'] for r in rr],label=arm,color=base.COLORS[arm])
 for ax,title in zip(axes,('Actual common ordinary probabilities','Cumulative ordinary-to-raw MSE','Cumulative max absolute ordinary IQ')):
  ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=7,ncol=2)
 fig.suptitle('Same original ordinary augmentation; actual full30 epochs');save(fig,'ordinary_augmentation_curves')
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for arm,rr in moves.items():
  for ax,ending in zip(axes,('cls_head.weight','sinc.low_coordinate','sinc.band_coordinate')):
   cells=[r for r in rr if r['parameter'].endswith(ending)]
   if len(cells)!=30:raise ValueError('Actual joint updates required')
   ax.plot(range(1,31),[r['L2_from_initial'] for r in cells],label=arm,color=base.COLORS[arm])
 for ax,title in zip(axes,('Default cosine head weight','Sinc low coordinate','Sinc band coordinate')):
  ax.set(title=title,xlabel='Epoch',ylabel='L2 from same-row initialization');ax.grid(alpha=.2);ax.legend(frameon=False)
 fig.suptitle('Parameter movement verifies joint training;not accuracy or per-packet gate evidence');save(fig,'joint_training_movement')
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for k in ('clean_CE','satellite_CE','secondary_ordinary_CE'):axes[0].plot(range(1,31),[r[k] for r in parts],label=k)
 for k in ('clean_KL','satellite_KL','secondary_ordinary_KL'):axes[1].plot(range(1,31),[r[k] for r in parts],label=k)
 for ax,title in zip(axes,('Actual R25 normalized CE components','Original same-IQ RDrop KL components')):
  ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
 save(fig,'R25_supervision_components');return files

def wait_for_run(pid):
 while not (ROOT/'complete.json').is_file():
  if (ROOT/'failure.json').is_file():raise RuntimeError('Run failed;preserve output,no restart')
  p=Path('/proc')/str(pid)
  if not p.is_dir() or 'cvs_cleanretain37/runner.py' not in (p/'cmdline').read_bytes().replace(b'\0',b' ').decode():raise RuntimeError('Owned dispatcher missing;reconcile first')
  time.sleep(15)

def main():
 p=argparse.ArgumentParser();p.add_argument('--truth',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--wait-dispatcher',type=int);a=p.parse_args()
 if a.wait_dispatcher is not None:wait_for_run(a.wait_dispatcher)
 tic=time.perf_counter();rs=roots()
 for r in rs.values():base.completed(r)
 if not (ROOT/'source_matrix_decision.json').is_file():raise ValueError('Source decision required before formal truth analysis')
 out=a.output;out.mkdir(exist_ok=False);sources=source_readout(out);decision=source_rule(out,sources);results={};private={};curves={};moves={}
 for arm in ORDER:
  results[arm],private[arm]=base.collect_row(arm,a.truth);curves[arm],moves[arm]=telemetry.extract(read(rs[arm]/'source/mechanism_progress.json'))
 if any(not np.array_equal(private['C0']['ids'],private[x]['ids']) for x in ORDER):raise ValueError('All same formal physical IDs required')
 ordinary={a:dict(probability_epoch_mismatches=sum(any(x[k]!=y.get(k) for k in x if k.startswith('p_')) for x,y in zip(rr,curves['C0'])),waveform_observation_epoch_mismatches=sum(any(x[k]!=y[k] for k in ('cumulative_waveform_MSE','cumulative_max_abs','calls','packets')) for x,y in zip(rr,curves['C0']))) for a,rr in curves.items()}
 sequences={};comparison={};y=private['C0']['y'];cohort=private['C0']['pred']['clean']==y;retention={a:{v:float((private[a]['pred'][v]==y)[cohort].mean()) for v in base.VIEWS} for a in ORDER}
 for arm,ref in PAIRS:
  tag=arm+'-'+ref;sequences[tag]=base.compare_training_sequences(arm,ref);comparison[tag]={}
  for v in base.VIEWS:
   cc=private[arm]['pred'][v]==y;old=private[ref]['pred'][v]==y;both_clean=(private[arm]['pred']['clean']==y)&(private[ref]['pred']['clean']==y);comparison[tag][v]=dict(accuracy_gain_pp=100*(cc.mean()-old.mean()),repairs=int((cc&~old).sum()),breaks=int((~cc&old).sum()),clean_and_view_joint_gain_pp=100*(((private[arm]['pred']['clean']==y)&cc).mean()-((private[ref]['pred']['clean']==y)&old).mean()),common_clean_correct_count=int(both_clean.sum()),common_clean_repairs=int((cc&~old&both_clean).sum()),common_clean_breaks=int((~cc&old&both_clean).sum()))
 parts,part_error=parts_curve(rs['R25']);flat=[dict(q,origin_run_id=r['run_id'],reused_control=arm!='R25') for arm,r in results.items() for q in r['metrics']];costs={a:dict(r['source']['cost'],reused_control=a!='R25') for a,r in results.items()};smoke=read(ROOT/'real_checkpoint_smoke.json');synthetic=read(ROOT/'synthetic/complete.json');reference=read(ROOT/'reference_source/complete.json');q=read(rs['R25']/'source_qualification/complete.json')
 if smoke['NN']!=2 or synthetic['NN']!=512 or reference['NN']!=12602 or q['NN']!=12602:raise ValueError('Actual separate smoke/source cost differs')
 write_csv(out/'per_row_seed.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in flat]);write_csv(out/'training_curves.csv',[dict(arm=a,origin_run_id=r['run_id'],reused_control=a!='R25',**x) for a,r in results.items() for x in r['source']['curve']]);write_csv(out/'ordinary_probability_curves.csv',[dict(arm=a,**x) for a,rr in curves.items() for x in rr]);write_csv(out/'parameter_movement.csv',[dict(arm=a,**x) for a,rr in moves.items() for x in rr]);write_csv(out/'R25_supervision_components.csv',parts)
 gaps={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in base.VIEWS[1:4]} for a,r in results.items()};run_wall=(ROOT/'complete.json').stat().st_mtime-read(Path(str(ROOT)+'-launch')/'launch.json')['read_at']
 summary=dict(code_commit=a.commit,training_code_commit=read(rs['R25']/'source/resolved_config.json')['commit'],rows=results,paired_comparisons=comparison,training_sequences=sequences,source_decision=decision,ordinary_comparison=ordinary,source_classification_decomposition_max_error=part_error,suburban_clean_gaps_pp=gaps,urban_mean_accuracy={a:sum(r['views'][v]['accuracy'] for v in base.VIEWS[4:])/3 for a,r in results.items()},fixed_C0_clean_correct_count=int(cohort.sum()),fixed_C0_clean_correct_retention=retention,costs=costs,actual_new_NN=costs['R25']['source_identity_NN']+costs['R25']['prediction_identity_NN']+q['NN']+reference['NN']+smoke['NN']+synthetic['NN'],new_model_epochs=30,new_updates=6660,training_parent_epochs=0,reused_control_epochs=60,reused_control_updates=13320,historical_control_NN=sum(costs[a]['source_identity_NN']+costs[a]['prediction_identity_NN'] for a in ('C0','AS50'))+12602,shared_compatibility_smoke=smoke,synthetic_cost=synthetic,reference_source_cost=reference,source_qualification_cost=q['NN'],run_wall_seconds=run_wall,source_epoch_records=90,formal_confusion_matrices=len(flat),source_confusion_matrices=len(sources['rows'])+len(sources['source_clean']),NN=0,IQ_reads=0,CP_reads=0,new_updates_analysis=0,model_seed_count=1,crossseed_confirmed=False,Ag_proved=False,goal_complete=False,benchmark_context='Human explicitly directed existing clean-retaining organization;not a new independent blind selection',limitations=['One model seed;source two channel draws are not model seeds','No formal state/quality sidecar;source strata cannot be called formal state accuracy','No per-packet branch gradients or fusion activations saved;parameter movement is not gain proof','C0 ordinaryrepeat/P32AS50 rawpractical controls reused,not training parents','Budget ended does not mean converged','No new physical Ag;no Ag factorial interaction claim;U TX hidden'])
 write(out/'analysis_summary.json',summary);files=base.plots(out,results)+source_plots(out,sources)+telemetry_plots(out,curves,moves,parts);write(out/'figures.json',dict(files=files));write(out/'completion.json',dict(status='CLEANRETAIN_ANALYSIS_COMPLETE',rows=list(ORDER),model_seed_count=1,source_epochs_parsed=90,formal_confusion_matrices=len(flat),source_confusion_matrices=summary['source_confusion_matrices'],figures=len(files),NN=0,IQ_reads=0,CP_reads=0,training_updates=0,analysis_wall_seconds=time.perf_counter()-tic,code_commit=a.commit,goal_complete=False));print(json.dumps(dict(status='CLEANRETAIN_ANALYSIS_COMPLETE',figures=len(files),NN=0,out=str(out))))
if __name__=='__main__':main()
