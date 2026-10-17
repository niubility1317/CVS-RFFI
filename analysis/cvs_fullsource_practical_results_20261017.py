"""Independent completed-row analysis. Reads no IQ/checkpoint; never trains a head."""
import argparse,csv,gzip,json,math,re
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN22='20261017-phase1-FcrossRX-fullsource-concat-e20-s2026092701-m2-r01'
RUN23='20261017-phase1-FcrossRX-fullsource-rawpractical-e20-s2026092701-m1-r01'
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
ROWS={'B00':(RUN22,'B00-s2026092701'),'B01':(RUN22,'B01-s2026092701'),'RAW_PRAC':(RUN23,'RAW_PRAC-s2026092701')}
COUNT=168000
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
def completed(root):
 p=read(root/'prediction/complete.json');s=read(root/'scoring/complete.json');d=read(root/'source/completion.json')
 if p.get('status')!='PREDICTIONS_COMPLETE' or p.get('count')!=COUNT or p.get('views')!=list(VIEWS) or p.get('truth_read') is not False or p.get('query_fit') is not False or p.get('all_parameters_frozen') is not True or p.get('head_fit_updates')!=0 or s.get('status')!='SCORED_COMPLETE' or s.get('views')!=list(VIEWS) or not s.get('truth_last') or d.get('epoch')!=20 or d.get('optimizer_steps')!=4440 or d.get('target_access') is not False or d.get('head_fit_updates')!=0:raise ValueError('Analysis truth remains closed: missing complete/frozen seven-view/defaulthead/source budget')
 return p,s,d
def metric(y,p):
 y=np.asarray(y,dtype=np.int64);p=np.asarray(p,dtype=np.int64)
 if y.shape!=p.shape or y.ndim!=1 or not len(y) or np.any((y<0)|(y>=6)|(p<0)|(p>=6)):raise ValueError('Invalid class decisions')
 cm=np.bincount(y*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1)
 f=np.divide(2*np.diag(cm),den,out=np.zeros(6,dtype=float),where=den>0)
 return dict(count=len(y),correct=int(np.trace(cm)),accuracy=float(np.trace(cm)/len(y)),macro_f1=float(f.mean()),confusion=cm.tolist())
def parse_source(root,done):
 source=root/'source';steps=[];epoch_sums=defaultdict(lambda:defaultdict(float));epoch_counts=Counter()
 active=('train/loss','train/loss_tx_labeled','full/RDrop_CE','full/RDrop_KL','full/SupCon','full/SupCon_weight','full/clean_CE','full/second_CE','full/clean_KL','full/second_KL')
 with gzip.open(source/'step_metrics.jsonl.gz','rt',encoding='utf8') as f:
  for i,line in enumerate(f,1):
   r=json.loads(line);m=r['metrics'];e=(i-1)//222+1
   if r['step']!=i or r['epoch']!=e:raise ValueError('Noncontiguous full training log')
   if any(m.get(k) or 0 for k in ['train/skipped_nonfinite_loss','train/skipped_nonfinite_grad']):raise ValueError('Actual skipped update')
   if e>13 and m.get('train/pseudo_truth_available',0)!=0:raise ValueError('U truth became available')
   for k in active:
    if k not in m or not isinstance(m[k],(int,float)) or not math.isfinite(m[k]):raise ValueError('Nonfinite active objective '+k)
    epoch_sums[e][k]+=m[k]
   for k in ('train/pseudo_total','train/pseudo_selected','train/pseudo_correct','train/pseudo_truth_available'):
    epoch_sums[e][k]+=m.get(k) or 0
   epoch_counts[e]+=1;steps.append(r['step'])
 if len(steps)!=4440 or set(epoch_counts.values())!={222}:raise ValueError('Fixed E20 update evidence incomplete')
 epochs=[json.loads(l) for l in (source/'epoch_metrics.jsonl').read_text().splitlines()]
 if [r['epoch'] for r in epochs]!=list(range(1,21)):raise ValueError('All20 epoch records required')
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
 if (supcon_count,supcon_active)!=(4440,3330):raise ValueError('SupCon execution budget incomplete')
 opt_count=0
 with gzip.open(source/'F_optimizer_steps.jsonl.gz','rt',encoding='utf8') as f:
  for i,line in enumerate(f,1):
   r=json.loads(line)
   if r['step']!=i or r['physical_clip_calls']!=i:raise ValueError('Sinc physical update incomplete')
   opt_count+=1
 if opt_count!=4440:raise ValueError('Missing Sinc update')
 observed=Counter();physical=defaultdict(set);sequence_records=0
 with gzip.open(source/'training_sequence.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   r=json.loads(line);observed[r['role']]+=len(r['ids']);physical[r['role']].update(tuple(x) for x in r['ids']);sequence_records+=1
 if observed!=dict(L_s=568320,U_s=396900) or {k:len(v) for k,v in physical.items()}!={'L_s':6300,'U_s':56700}:raise ValueError('Full source exposure differs')
 result=read(source/'result.json');roles=read(source/'full_source_roles.json')
 if roles['held_RX'] is not None or roles['counts']!={'L_s':6300,'U_s':56700,'V':27000} or result['head_fit_updates'] or result['U_truth_read'] or result['checkpoint_sources']:raise ValueError('Full source/defaulthead boundary differs')
 stdout=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace')
 errors=[dict(line=i,text=line[:250]) for i,line in enumerate(stdout.splitlines(),1) if re.search(r'Traceback|CUDA out of memory|\bKilled\b|RuntimeError|AssertionError|\[ERROR\]',line)]
 return dict(curve=curve,source_result=result,full_records=dict(steps=4440,epochs=20,sequence_records=sequence_records,loader_packets=dict(observed),unique_physical={k:len(v) for k,v in physical.items()},SupCon=supcon_count,SupCon_active=supcon_active,anchors=anchors,valid_anchors=valid,Sinc_updates=opt_count,stdout_lines=len(stdout.splitlines()),error_markers=errors,NaN_text_mentions=len(re.findall(r'\bnan\b',stdout,re.I))),cost=dict(source_identity_NN=sum(result['actual_counters'].values()),source_counters=result['actual_counters'],source_wall_seconds=done['elapsed_seconds'],L_identity_NN=2273280,new_epochs=20,new_updates=4440,training_parent_epochs=0))
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
 ax.set(xticks=xx,xticklabels=labels,ylabel='Accuracy (%)',ylim=(0,105),title='Fixed default head / full source / E20 / seed2026092701')
 ax.legend(frameon=False,ncol=3);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);fig.tight_layout()
 for suffix in ['png','pdf']:p=out/('formal_accuracy.'+suffix);fig.savefig(p,dpi=200);figures.append(str(p))
 plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(10.5,6.7))
 for name,r in results.items():
  curve=r['source']['curve'];e=[c['epoch'] for c in curve];color=colors[name]
  axes[0,0].plot(e,[c['full/RDrop_CE'] for c in curve],label=name,color=color)
  axes[0,1].plot(e,[c['source_val_accuracy_pct'] for c in curve],label=name,color=color)
  axes[1,0].plot(e,[c['full/RDrop_KL'] for c in curve],label=name,color=color)
  axes[1,1].plot(e,[c['pseudo_selected']/c['pseudo_total'] if c['pseudo_total'] else np.nan for c in curve],label=name,color=color)
 for ax,title in zip(axes.flat,['L paired CE','Original source V accuracy (%)','RDrop symmetric KL','Original U gate selected fraction']):
  ax.set(title=title,xlabel='Epoch',xlim=(1,20));ax.axvline(13.5,color='gray',ls='--',alpha=.5);ax.grid(alpha=.2)
 axes[0,0].legend(frameon=False);fig.suptitle('All20 epochs; V is source health, not cross-RX target performance');fig.tight_layout()
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
 fig.suptitle('All fixed test RX; no per-sample receiver-state labels inferred');fig.tight_layout()
 for suffix in ['png','pdf']:p=out/('receiver_accuracy.'+suffix);fig.savefig(p,dpi=200);figures.append(str(p))
 plt.close(fig)
 return figures
def main():
 p=argparse.ArgumentParser();p.add_argument('--row',choices=[*ROWS,'all'],required=True);p.add_argument('--truth',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
 names=list(ROWS) if a.row=='all' else [a.row]
 # Guard all requested rows before any truth access or output creation.
 for name in names:
  run,row=ROWS[name];completed(PROJECT/'runs'/run/row)
 out=a.output;out.mkdir(parents=True,exist_ok=False);results={};private={}
 for name in names:results[name],private[name]=collect_row(name,a.truth)
 comparisons={}
 if len(names)>1:
  if any(not np.array_equal(private[names[0]]['ids'],private[n]['ids']) for n in names):raise ValueError('Paired physical IDs differ')
  for name,ref in [('B01','B00'),('RAW_PRAC','B01'),('RAW_PRAC','B00')]:
   if name not in results or ref not in results:continue
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
 write(out/'analysis_summary.json',dict(code_commit=a.commit,rows=results,paired_comparisons=comparisons,source_epoch_records=sum(len(r['source']['curve']) for r in results.values()),formal_confusion_matrices=len(flat),NN=0,checkpoint_reads=0,real_IQ_reads=0,training_updates=0,crossseed_confirmed=False,Ag_proved=False,goal_complete=False,limitations=['One model seed only','No fixed formal per-sample state/quality files; no invented LOS/lock accuracy','Original U truth hidden; gate selected fraction is not precision','No per-packet fusion-gate activations saved']))
 figures=plots(out,results);write(out/'figures.json',dict(files=figures))
 write(out/'completion.json',dict(status='ANALYSIS_COMPLETE',rows=names,formal_confusion_matrices=len(flat),source_updates=len(names)*4440,source_epochs=len(names)*20,NN=0,real_IQ_reads=0,checkpoint_reads=0,training_updates=0,code_commit=a.commit))
 print(json.dumps(dict(status='ANALYSIS_COMPLETE',rows=names,out=str(out),figures=len(figures),confusion_matrices=len(flat),NN=0)))
if __name__=='__main__':main()
