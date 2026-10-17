"""Independent completed-row analysis. Reads no IQ/checkpoint; never trains a head."""
import argparse,csv,gzip,json,math,re
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN30='20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01'
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
ROWS={'B00':(RUN30,'B00-s2026092701'),'B01':(RUN30,'B01-s2026092701')}
COUNT=168000
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
def completed(root):
 p=read(root/'prediction/complete.json');s=read(root/'scoring/complete.json');d=read(root/'source/completion.json')
 if p.get('status')!='PREDICTIONS_COMPLETE' or p.get('count')!=COUNT or p.get('views')!=list(VIEWS) or p.get('truth_read') is not False or p.get('query_fit') is not False or p.get('all_parameters_frozen') is not True or p.get('head_fit_updates')!=0 or s.get('status')!='SCORED_COMPLETE' or s.get('views')!=list(VIEWS) or not s.get('truth_last') or d.get('epoch')!=30 or d.get('optimizer_steps')!=6660 or d.get('target_access') is not False or d.get('head_fit_updates')!=0:raise ValueError('Analysis truth remains closed: missing complete/frozen seven-view/defaulthead/source budget')
 c=d.get('config',{})
 if (c.get('epochs'),c.get('label_epochs'),c.get('pseudo_epochs'),c.get('updates'))!=(30,20,10,6660) or c.get('held_RX') is not None or c.get('source_roles')!=dict(L_s=6300,U_s=56700,V=27000):raise ValueError('Analysis requires actual E30/20L10U and original full-source roles')
 return p,s,d
def metric(y,p):
 y=np.asarray(y,dtype=np.int64);p=np.asarray(p,dtype=np.int64)
 if y.shape!=p.shape or y.ndim!=1 or not len(y) or np.any((y<0)|(y>=6)|(p<0)|(p>=6)):raise ValueError('Invalid class decisions')
 cm=np.bincount(y*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1)
 f=np.divide(2*np.diag(cm),den,out=np.zeros(6,dtype=float),where=den>0)
 return dict(count=len(y),correct=int(np.trace(cm)),accuracy=float(np.trace(cm)/len(y)),macro_f1=float(f.mean()),confusion=cm.tolist())
def physics_summary(root,result):
 if not result['config']['As']:return dict(satellite_packets=0,not_applicable=True)
 stats=defaultdict(list);counts=Counter();bins=Counter();packets=batches=0
 bounds=(-math.inf,0,5,10,15,20,math.inf);labels=('<0','0-5','5-10','10-15','15-20','>=20')
 with gzip.open(root/'source/training_physics.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   q=json.loads(line);batches+=1
   if q['step']!=batches or len(q['records'])!=128:raise ValueError('Training physical metadata incomplete')
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
 if (packets,batches)!=(852480,6660) or sum(x['count'] for x in result['satellite_coverage'])!=packets:raise ValueError('Training physics budget mismatch')
 summary={}
 for key,values in stats.items():
  x=np.asarray(values,dtype=float);summary[key]=dict(count=len(x),mean=float(x.mean()),std=float(x.std()),quantiles=dict(zip(['min','q01','q10','q50','q90','q99','max'],np.quantile(x,[0,.01,.1,.5,.9,.99,1]).tolist())))
 return dict(satellite_packets=packets,batches=batches,statistics=summary,counts=[dict(field=k[0],value=k[1],count=v) for k,v in sorted(counts.items())],quality_bins=[dict(scene=k[0],state=k[1],lock=k[2],quality_proxy_db_bin=k[3],count=v) for k,v in sorted(bins.items())],TX_RX_scene_cells=len({(x['TX'],x['RX'],x['scene']) for x in result['satellite_coverage']}),TX_RX_state_cells=len({(x['TX'],x['RX'],x['state']) for x in result['satellite_coverage']}),true_total_SNR_identified=False,causal_state_effect_identified=False,boundary_accuracy_effect_identified=False,formal_state_accuracy_identified=False)
def parse_source(root,done):
 source=root/'source';steps=[];epoch_sums=defaultdict(lambda:defaultdict(float));epoch_counts=Counter()
 active=('train/loss','train/loss_tx_labeled','full/RDrop_CE','full/RDrop_KL','full/SupCon','full/SupCon_weight','full/clean_CE','full/second_CE','full/clean_KL','full/second_KL')
 with gzip.open(source/'step_metrics.jsonl.gz','rt',encoding='utf8') as f:
  for i,line in enumerate(f,1):
   r=json.loads(line);m=r['metrics'];e=(i-1)//222+1
   if r['step']!=i or r['epoch']!=e:raise ValueError('Noncontiguous full training log')
   if any(m.get(k) or 0 for k in ['train/skipped_nonfinite_loss','train/skipped_nonfinite_grad']):raise ValueError('Actual skipped update')
   if e>20 and m.get('train/pseudo_truth_available',0)!=0:raise ValueError('U truth became available')
   for k in active:
    if k not in m or not isinstance(m[k],(int,float)) or not math.isfinite(m[k]):raise ValueError('Nonfinite active objective '+k)
    epoch_sums[e][k]+=m[k]
   for k in ('train/pseudo_total','train/pseudo_selected','train/pseudo_correct','train/pseudo_truth_available'):
    epoch_sums[e][k]+=m.get(k) or 0
   epoch_counts[e]+=1;steps.append(r['step'])
 if len(steps)!=6660 or set(epoch_counts.values())!={222}:raise ValueError('Fixed E30 update evidence incomplete')
 epochs=[json.loads(l) for l in (source/'epoch_metrics.jsonl').read_text().splitlines()]
 if [r['epoch'] for r in epochs]!=list(range(1,31)):raise ValueError('All30 epoch records required')
 curve=[]
 for r in epochs:
  e=r['epoch'];curve.append(dict(epoch=e,source_val_accuracy_pct=r['val_tx_acc'],epoch_wall_seconds=r['epoch_time_s'],**{k:epoch_sums[e][k]/222 for k in active},pseudo_total=epoch_sums[e]['train/pseudo_total'],pseudo_selected=epoch_sums[e]['train/pseudo_selected']))
 supcon_count=supcon_active=anchors=valid=0
 with gzip.open(source/'SupCon_steps.jsonl.gz','rt',encoding='utf8') as f:
  for i,line in enumerate(f,1):
   r=json.loads(line);weight=.1*min(max(((i-1)//222+1-5)/10,0),1)
   if r['step']!=i or abs(r['weight']-weight)>1e-12:raise ValueError('SupCon schedule differs')
   supcon_count+=1;supcon_active+=bool(weight)
   for q in r['details']:anchors+=q['total_anchors'];valid+=q['valid_anchors']
 if (supcon_count,supcon_active)!=(6660,5550):raise ValueError('SupCon execution budget incomplete')
 opt_count=0
 with gzip.open(source/'F_optimizer_steps.jsonl.gz','rt',encoding='utf8') as f:
  for i,line in enumerate(f,1):
   r=json.loads(line)
   if r['step']!=i or r['physical_clip_calls']!=i:raise ValueError('Sinc physical update incomplete')
   opt_count+=1
 if opt_count!=6660:raise ValueError('Missing Sinc update')
 observed=Counter();physical=defaultdict(set);sequence_records=0
 with gzip.open(source/'training_sequence.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line);observed[r['role']]+=len(r['ids']);physical[r['role']].update(tuple(x) for x in r['ids']);sequence_records+=1
 if observed!=dict(L_s=852480,U_s=567000) or {k:len(v) for k,v in physical.items()}!={'L_s':6300,'U_s':56700}:raise ValueError('Full source exposure differs')
 result=read(source/'result.json');roles=read(source/'full_source_roles.json')
 if roles['held_RX'] is not None or roles['counts']!={'L_s':6300,'U_s':56700,'V':27000} or result['head_fit_updates'] or result['U_truth_read'] or result['checkpoint_sources']:raise ValueError('Full source/defaulthead boundary differs')
 stdout=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace')
 errors=[dict(line=i,text=line[:250]) for i,line in enumerate(stdout.splitlines(),1) if re.search(r'Traceback|CUDA out of memory|\bKilled\b|RuntimeError|AssertionError|\[ERROR\]',line)]
 return dict(curve=curve,source_result=result,physical_metadata=physics_summary(root,result),full_records=dict(steps=6660,epochs=30,sequence_records=sequence_records,loader_packets=dict(observed),unique_physical={k:len(v) for k,v in physical.items()},SupCon=supcon_count,SupCon_active=supcon_active,anchors=anchors,valid_anchors=valid,Sinc_updates=opt_count,stdout_lines=len(stdout.splitlines()),error_markers=errors,NaN_text_mentions=len(re.findall(r'\bnan\b',stdout,re.I))),cost=dict(source_identity_NN=sum(result['actual_counters'].values()),source_counters=result['actual_counters'],source_wall_seconds=done['elapsed_seconds'],L_identity_NN=3409920,new_epochs=30,new_updates=6660,training_parent_epochs=0))
def collect_row(name,truth_path):
 run,row=ROWS[name];root=PROJECT/'runs'/run/row
 pred_done,score_done,source_done=completed(root)
 with np.load(root/'prediction/predictions.npz',allow_pickle=False) as f:
  if set(f.files)!={'ids',*VIEWS}:raise ValueError('Fixed prediction set differs')
  ids=f['ids'].copy();pred={v:f[v].copy() for v in VIEWS}
 if len(ids)!=COUNT or len(set(ids.tolist()))!=COUNT:raise ValueError('Prediction IDs differ')
 # No truth is opened until the frozen/prediction/scoring checks above pass.
 truth=read(truth_path)
 if set(truth)!=set(ids.tolist()):raise ValueError('Cannot align frozen predictions with truth')
 y=np.asarray([truth[s]['label'] for s in ids]);rx=np.asarray([str(truth[s]['receiver']) for s in ids]);day=np.asarray([str(truth[s]['day']) for s in ids])
 saved=read(root/'scoring/scores.json')
 if not saved['truth_last'] or not saved['target_feedback_forbidden']:raise ValueError('Scorer boundary differs')
 metrics=[];view_summary={}
 for v in VIEWS:
  with np.load(root/'prediction'/(v+'_scores.npz'),allow_pickle=False) as f:
   logits=f['logits']
   if not np.array_equal(ids,f['ids']) or logits.shape!=(COUNT,6) or not np.isfinite(logits).all() or not np.array_equal(logits.argmax(1),pred[v]):raise ValueError('Frozen logits/decisions differ')
   competitors=logits.copy();competitors[np.arange(COUNT),y]=-np.inf
   margin=(logits[np.arange(COUNT),y]-competitors.max(1))/30
  for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
   for value in values:
    mask=np.ones(COUNT,dtype=bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value
    q=metric(y[mask],pred[v][mask]);q.update(row_id=row,model_seed=2026092701,arm=name,view=v,dimension=dim,stratum=str(value))
    old=next(r for r in saved['results'] if r['view']==v and r['dimension']==dim and r['stratum']==str(value))
    if q['confusion']!=old['confusion'] or q['count']!=old['query_count'] or abs(q['accuracy']-old['accuracy'])>1e-12 or abs(q['macro_f1']-old['macro_f1'])>1e-12:raise ValueError('Independent score recount mismatch')
    metrics.append(q)
  overall=next(q for q in metrics if q['view']==v and q['dimension']=='overall')
  view_summary[v]=dict(overall,mean_true_margin_over30=float(margin.mean()),margin_quantiles_over30=np.quantile(margin,[0,.1,.5,.9,1]).tolist(),worst_RX_accuracy=min(q['accuracy'] for q in metrics if q['view']==v and q['dimension']=='receiver'))
 source=parse_source(root,source_done);source['cost'].update(prediction_identity_NN=pred_done['actual_NN'],prediction_wall_seconds=pred_done['wall_seconds'])
 return dict(arm=name,row_id=row,run_id=run,views=view_summary,metrics=metrics,source=source,count=COUNT,model_seed=2026092701,model_seed_count=1,crossseed_confirmed=False,independent_recount='VERIFIED',real_IQ_reads=0,checkpoint_reads=0,NN=0,new_updates=0,formal_per_sample_physics_available=False),dict(ids=ids,y=y,pred=pred,rx=rx,day=day)
def write_csv(path,rows):
 with Path(path).open('x',encoding='utf8',newline='') as f:
  keys=list(rows[0]);w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def plots(out,results):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 labels=['Clean','High sub','Mid sub','Low sub','High urban','Mid urban','Low urban']
 colors={'B00':'#64748b','B01':'#2563eb','RAW_PRAC':'#e87918'}
 figures=[];fig,ax=plt.subplots(figsize=(10.5,4.6));xx=np.arange(7);width=.78/len(results)
 for i,(name,r) in enumerate(results.items()):
  vals=[100*r['views'][v]['accuracy'] for v in VIEWS];pos=xx-.39+width/2+i*width
  bars=ax.bar(pos,vals,width,label=name,color=colors[name]);ax.bar_label(bars,fmt='%.1f',fontsize=7,padding=2)
 ax.set(xticks=xx,xticklabels=labels,ylabel='Accuracy (%)',ylim=(0,105),title='Fixed default head / full source / E30 / seed2026092701')
 ax.legend(frameon=False,ncol=3);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);fig.tight_layout()
 for suffix in ['png','pdf']:p=out/('formal_accuracy.'+suffix);fig.savefig(p,dpi=200);figures.append(str(p))
 plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(13.5,7))
 for name,r in results.items():
  curve=r['source']['curve'];e=[c['epoch'] for c in curve];color=colors[name]
  axes[0,0].plot(e,[c['full/RDrop_CE'] for c in curve],label=name,color=color)
  axes[0,1].plot(e,[c['full/clean_CE'] for c in curve],label=name+' ordinary clean',color=color)
  axes[0,1].plot(e,[c['full/second_CE'] for c in curve],label=name+' second',color=color,ls='--')
  axes[0,2].plot(e,[c['source_val_accuracy_pct'] for c in curve],label=name,color=color)
  axes[1,0].plot(e,[c['full/RDrop_KL'] for c in curve],label=name,color=color)
  axes[1,1].plot(e,[c['pseudo_selected']/c['pseudo_total'] if c['pseudo_total'] else np.nan for c in curve],label=name,color=color)
  axes[1,2].plot(e,[c['epoch_wall_seconds'] for c in curve],label=name,color=color)
 for ax,title in zip(axes.flat,['L paired CE','Ordinary clean / second view CE','Original source V accuracy (%)','RDrop symmetric KL','Original U gate selected fraction','Epoch wall time (seconds)']):
  ax.set(title=title,xlabel='Epoch',xlim=(1,30));ax.axvline(20.5,color='gray',ls='--',alpha=.5);ax.grid(alpha=.2)
 axes[0,0].legend(frameon=False);axes[0,1].legend(frameon=False,fontsize=7);fig.suptitle('All30 epochs; V is source health, not cross-RX target performance');fig.tight_layout()
 for suffix in ['png','pdf']:p=out/('training_curves.'+suffix);fig.savefig(p,dpi=200);figures.append(str(p))
 plt.close(fig)
 fig,axes=plt.subplots(1,len(results),figsize=(5*len(results),4.8),squeeze=False)
 for ax,(name,r) in zip(axes[0],results.items()):
  rxs=sorted({q['stratum'] for q in r['metrics'] if q['dimension']=='receiver'})
  matrix=np.asarray([[100*next(q['accuracy'] for q in r['metrics'] if q['view']==v and q['dimension']=='receiver' and q['stratum']==rx) for v in VIEWS] for rx in rxs])
  im=ax.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
  ax.set(xticks=np.arange(7),xticklabels=labels,yticks=np.arange(len(rxs)),yticklabels=['RX'+rx for rx in rxs],title=name)
  ax.tick_params(axis='x',rotation=60)
  for j in range(len(rxs)):
   for k in range(7):ax.text(k,j,f'{matrix[j,k]:.1f}',ha='center',va='center',fontsize=7,color='white' if matrix[j,k]>65 else 'black')
  fig.colorbar(im,ax=ax,fraction=.045,label='Accuracy (%)')
 fig.suptitle('Fixed test RX accuracy (state labels unavailable)',fontsize=11);fig.tight_layout()
 for suffix in ['png','pdf']:p=out/('receiver_accuracy.'+suffix);fig.savefig(p,dpi=200);figures.append(str(p))
 plt.close(fig)
 for name,r in results.items():
  physics=r['source']['physical_metadata']
  if physics.get('not_applicable'):continue
  scenes=['practical_low_suburban','practical_mid_urban','practical_low_urban'];scene_labels=['Low suburban','Mid urban','Low urban'];bins=physics['quality_bins'];fig,axes=plt.subplots(2,2,figsize=(10.5,7.3))
  panels=[('state',['LOS','PARTIAL','BLOCKED'],axes[0,0],'Training propagation state (%)'),('lock',['locked','degraded','unlocked'],axes[0,1],'Training receiver lock (%)'),('quality_proxy_db_bin',['<0','0-5','5-10','10-15','15-20','>=20'],axes[1,0],'Training white-power quality proxy (%)')]
  for key,labels,ax,title in panels:
   bottom=np.zeros(3);totals=np.asarray([sum(b['count'] for b in bins if b['scene']==s) for s in scenes])
   for label in labels:
    vals=np.asarray([sum(b['count'] for b in bins if b['scene']==s and b[key]==label) for s in scenes],dtype=float);vals=100*vals/totals
    ax.bar(range(3),vals,bottom=bottom,label=label);bottom+=vals
   ax.set(xticks=range(3),xticklabels=scene_labels,ylim=(0,102),title=title);ax.legend(frameon=False,fontsize=7,ncol=3);ax.grid(axis='y',alpha=.15)
  coverage=r['source']['source_result']['satellite_coverage'];rxs=[1,3,4,6,8]
  matrix=np.asarray([[sum(c['count'] for c in coverage if c['RX']==rx and c['scene']==s) for s in scenes] for rx in rxs])
  im=axes[1,1].imshow(matrix,cmap='Blues',aspect='auto');axes[1,1].set(xticks=range(3),xticklabels=scene_labels,yticks=range(5),yticklabels=['RX'+str(rx) for rx in rxs],title='Training observations by source RX/scene')
  for j in range(5):
   for k in range(3):axes[1,1].text(k,j,str(matrix[j,k]),ha='center',va='center',fontsize=8,color='white' if matrix[j,k]>matrix.min()+.6*(matrix.max()-matrix.min()) else 'black')
  fig.colorbar(im,ax=axes[1,1],fraction=.045,label='Observation count');fig.suptitle(name+' full source E30 training support; counts are not accuracy or total SNR');fig.tight_layout()
  for suffix in ['png','pdf']:p=out/('training_support_'+name+'.'+suffix);fig.savefig(p,dpi=200);figures.append(str(p))
  plt.close(fig)
 return figures
def compare_training_sequences(name,ref):
 from itertools import zip_longest
 roots=[PROJECT/'runs'/ROWS[x][0]/ROWS[x][1]/'source/training_sequence.jsonl.gz' for x in (name,ref)]
 count=mismatch=0
 with gzip.open(roots[0],'rt',encoding='utf8') as a,gzip.open(roots[1],'rt',encoding='utf8') as b:
  for x,y in zip_longest(a,b):
   count+=1
   mismatch+=x is None or y is None or json.loads(x)!=json.loads(y)
 return dict(records_compared=count,mismatched_records=mismatch,identical=mismatch==0,no_hash_gate=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--row',choices=[*ROWS,'all'],required=True);p.add_argument('--truth',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 names=list(ROWS) if a.row=='all' else [a.row]
 # Guard all requested rows before any truth access or output creation.
 for name in names:
  run,row=ROWS[name];completed(PROJECT/'runs'/run/row)
 out=a.output;out.mkdir(parents=True,exist_ok=False);results={};private={}
 for name in names:results[name],private[name]=collect_row(name,a.truth)
 comparisons={};sequence_comparisons={}
 if len(names)>1:
  if any(not np.array_equal(private[names[0]]['ids'],private[n]['ids']) for n in names):raise ValueError('Paired physical IDs differ')
  for name,ref in [('B01','B00')]:
   if name not in results or ref not in results:continue
   sequence_comparisons[name+'-'+ref]=compare_training_sequences(name,ref)
   y=private[name]['y'];cohort=private[ref]['pred']['clean']==y;views={}
   for v in VIEWS:
    correct=private[name]['pred'][v]==y;old=private[ref]['pred'][v]==y
    views[v]=dict(accuracy_gain_pp=100*(correct.mean()-old.mean()),repairs=int((correct&~old).sum()),breaks=int((~correct&old).sum()),reference_clean_correct_count=int(cohort.sum()),satellite_retention_on_reference_clean_correct=float(correct[cohort].mean()))
   comparisons[name+'-'+ref]=views
 flat=[]
 for r in results.values():flat.extend(r['metrics'])
 write_csv(out/'per_row_seed.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in flat])
 curves=[dict(arm=n,**q) for n,r in results.items() for q in r['source']['curve']]
 write_csv(out/'training_curves.csv',curves)
 write(out/'analysis_summary.json',dict(code_commit=a.commit,rows=results,paired_comparisons=comparisons,training_sequences=sequence_comparisons,source_epoch_records=sum(len(r['source']['curve']) for r in results.values()),formal_confusion_matrices=len(flat),NN=0,checkpoint_reads=0,real_IQ_reads=0,training_updates=0,crossseed_confirmed=False,Ag_proved=False,goal_complete=False,limitations=['One model seed only','No fixed formal per-sample state/quality files; no invented LOS/lock accuracy','Original U truth hidden; gate selected fraction is not precision','No per-packet fusion-gate activations saved']))
 figures=plots(out,results);write(out/'figures.json',dict(files=figures))
 write(out/'completion.json',dict(status='ANALYSIS_COMPLETE',rows=names,formal_confusion_matrices=len(flat),source_updates=len(names)*6660,source_epochs=len(names)*30,NN=0,real_IQ_reads=0,checkpoint_reads=0,training_updates=0,code_commit=a.commit))
 print(json.dumps(dict(status='ANALYSIS_COMPLETE',rows=names,out=str(out),figures=len(figures),confusion_matrices=len(flat),NN=0)))
if __name__=='__main__':main()
