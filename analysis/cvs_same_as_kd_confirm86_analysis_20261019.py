"""Matched3 studentseed confirmation from complete frozen artifacts only."""
import json,statistics
from pathlib import Path
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def collect(project):
 import numpy as np
 from experiments.cvs_same_as_kd_confirm86 import design as d
 from experiments.cvs_early_geometry.evaluate import truth_arrays
 if not (d.BASE/'source_matrix_decision.json').is_file():raise ValueError('Source confirmation must finish before final readout')
 predictions={};metrics={};strata={};cost={}
 matrix=[dict(r,root=d.row_root(r['row_id'])) for r in d.rows()]
 for seed in d.SEEDS:
  ref=Path(project)/'runs'/('20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01' if seed==2026092701 else '20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02')
  for arm,old in [('AS30','B01'),('NOAS30','B00')]:matrix.append(dict(arm=arm,model_seed=seed,row_id=arm+'-s'+str(seed),root=ref/(old+'-s'+str(seed))))
 expected_roles=read(d.SOURCE)['role_ids']
 for row in matrix:
  key=row['row_id'];b=row['root']
  if read(b/'source/source_contract.json')['role_ids']!=expected_roles:raise ValueError('Comparison source roles differ')
  done=read(b/'prediction/complete.json');scored=read(b/'scoring/complete.json')
  if done['status']!='PREDICTIONS_COMPLETE' or done['views']!=list(VIEWS) or done['truth_read'] or scored['status']!='SCORED_COMPLETE':raise ValueError('Incomplete fixed all7: '+key)
  with np.load(b/'prediction/predictions.npz',allow_pickle=False) as f:predictions[key]={k:f[k].copy() for k in f.files}
  rr=read(b/'scoring/scores.json')['results'];strata[key]=rr;metrics[key]={}
  for view in VIEWS:
   x=next(v for v in rr if v['view']==view and v['dimension']=='overall');rx=[v for v in rr if v['view']==view and v['dimension']=='receiver'];metrics[key][view]=dict(accuracy=x['accuracy'],macro_F1=x['macro_f1'],worst_RX=min(v['accuracy'] for v in rx),count=len(predictions[key][view]))
  metrics[key]['urban_mean_accuracy']=sum(metrics[key][v]['accuracy'] for v in VIEWS[-3:])/3
  cost[key]={}
  for seg in ('stage20','source'):
   q=read(b/seg/'result.json');cost[key][seg]={k:v for k,v in q.items() if v is None or isinstance(v,(str,int,float,bool))};cost[key][seg]['sameAsKD']=q.get('sameAsKD');cost[key][seg]['actual_U_student_NN']=q['student_train_NN']-q['L_NN']
 ids=predictions[d.rows()[0]['row_id']]['ids']
 if any(not np.array_equal(x['ids'],ids) for x in predictions.values()):raise ValueError('Fixed physical ordering differs')
 y,rx,day=truth_arrays(read(d.TRUTH),ids);paired={}
 for seed in d.SEEDS:
  a='KD-s'+str(seed);b='T1-s'+str(seed);paired[str(seed)]={}
  for view in VIEWS:
   ac=predictions[a][view]==y;bc=predictions[b][view]==y;paired[str(seed)][view]=dict(repaired=int((ac&~bc).sum()),new_errors=int((~ac&bc).sum()),net=int(ac.sum()-bc.sum()),accuracy_pp=100*float(ac.mean()-bc.mean()),macro_F1_pp=100*(metrics[a][view]['macro_F1']-metrics[b][view]['macro_F1']),worst_RX_pp=100*(metrics[a][view]['worst_RX']-metrics[b][view]['worst_RX']))
  paired[str(seed)]['urban_mean_accuracy_pp']=100*(metrics[a]['urban_mean_accuracy']-metrics[b]['urban_mean_accuracy'])
 summary={};delta={}
 for arm in ('T1','KD','AS30','NOAS30'):
  summary[arm]={}
  for view in VIEWS:
   summary[arm][view]={}
   for metric in ('accuracy','macro_F1','worst_RX'):
    values=[100*metrics[arm+'-s'+str(s)][view][metric] for s in d.SEEDS];summary[arm][view][metric]=dict(mean_percent=statistics.mean(values),sample_std_pp=statistics.stdev(values),per_seed_percent=values)
  vals=[100*metrics[arm+'-s'+str(s)]['urban_mean_accuracy'] for s in d.SEEDS];summary[arm]['urban_mean_accuracy']=dict(mean_percent=statistics.mean(vals),sample_std_pp=statistics.stdev(vals),per_seed_percent=vals)
 for view in VIEWS:
  values=[paired[str(s)][view]['accuracy_pp'] for s in d.SEEDS];delta[view]=dict(mean_pp=statistics.mean(values),sample_std_pp=statistics.stdev(values),per_seed_pp=values,positive_seed_count=sum(v>0 for v in values))
 vals=[paired[str(s)]['urban_mean_accuracy_pp'] for s in d.SEEDS];delta['urban_mean_accuracy']=dict(mean_pp=statistics.mean(vals),sample_std_pp=statistics.stdev(vals),per_seed_pp=vals)
 comparisons={}
 for arm,refarm in [('KD','AS30'),('KD','NOAS30'),('T1','AS30'),('T1','NOAS30'),('AS30','NOAS30')]:
  cells={}
  for view in VIEWS:
   vals=[];pairs=[]
   for seed in d.SEEDS:
    a=arm+'-s'+str(seed);b=refarm+'-s'+str(seed);ac=predictions[a][view]==y;bc=predictions[b][view]==y;v=100*float(ac.mean()-bc.mean());vals.append(v);pairs.append(dict(model_seed=seed,repaired=int((ac&~bc).sum()),new_errors=int((~ac&bc).sum()),accuracy_pp=v))
   cells[view]=dict(mean_pp=statistics.mean(vals),sample_std_pp=statistics.stdev(vals),per_seed_pp=vals,pairs=pairs)
  comparisons[arm+'_minus_'+refarm]=cells
 return dict(status='ALL6_ROWS_ALL7_SCORED',run_id=d.RUN,model_seed_count=3,teacher_seed_count=1,metrics=metrics,summary=summary,paired=paired,paired_summary=delta,baseline_comparisons=comparisons,strata=strata,cost=cost,source_decision=read(d.BASE/'source_matrix_decision.json'),source_decision_context='Source function reads no target metrics. Seed01 formal scores were read after the entire3seed matrix lock;newseed target scores remain unread until all source readouts. This is descriptive confirmation,not new blind selection.',head_fit_updates=0,new_NN=0,raw_IQ_reads=0,goal_complete=False,baseline_initialization_caveat='Historical seed02/03 native seeds/source roles/budgets match,202/206 initializationtensors match;4 sequence_pool projectionweights differ from current fixedseed01 subinitializer. T1/KD current pair all206 tensors match. Historical As/NoAs comparisons descriptive,not pure mechanism attribution.',uncertainty='Sample std across3 studentseeds with fixed sequence-pool subinitialization;fixed test observations,one sharedteacher;not independent blind test or inorbit deployment')


def render(result,training,out):
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=Path(out);out.mkdir(parents=True,exist_ok=False);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180})
 labels=['Clean','High','Mid','Low suburban','High urban','Mid urban','Low urban'];colors={'T1':'#295A83','KD':'#6B4E9B'}
 fig,axs=plt.subplots(1,2,figsize=(12,4.2))
 for arm in ('T1','KD'):
  q=result['summary'][arm];vals=[q[v]['accuracy']['mean_percent'] for v in VIEWS];std=[q[v]['accuracy']['sample_std_pp'] for v in VIEWS]
  axs[0].errorbar(range(7),vals,yerr=std,color=colors[arm],marker='o',capsize=3,label=arm)
 axs[0].set_ylim(0,100);axs[0].set_ylabel('Accuracy (%)');axs[0].set_title('FINAL30 default-head evaluation, 3 student seeds',loc='left');axs[0].legend(frameon=False)
 means=[result['paired_summary'][v]['mean_pp'] for v in VIEWS];std=[result['paired_summary'][v]['sample_std_pp'] for v in VIEWS]
 axs[1].bar(range(7),means,yerr=std,capsize=3,color='#8E78AC',alpha=.7)
 for i,v in enumerate(VIEWS):axs[1].scatter(np.full(3,i)+np.array([-.12,0,.12]),result['paired_summary'][v]['per_seed_pp'],color='#423453',s=20,zorder=4)
 axs[1].axhline(0,color='#777',lw=.8);axs[1].set_ylabel('KD minus T1 (percentage points)');axs[1].set_title('Matched differences; points = individual seeds',loc='left')
 for ax in axs:ax.set_xticks(range(7),labels,rotation=24,ha='right')
 fig.text(.08,.015,'Error bars: sample standard deviation across 3 paired student seeds. One shared CSIL teacher; fixed received observations.',fontsize=8,color='#555');fig.tight_layout(rect=(0,.06,1,1));fig.savefig(out/'confirmation_seven_views.png',bbox_inches='tight');fig.savefig(out/'confirmation_seven_views.pdf',bbox_inches='tight');plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(12,3.8))
 for arm in ('T1','KD'):
  for ax,key in zip(axs,('As_native_NLL','As_per_example_cosine_gradient_proxy','teacher_correct_fraction')):
   vals=np.asarray([[e[key] if e.get(key) is not None else np.nan for e in training[str(s)]['rows'][arm]['epochs']] for s in (2026092701,2026092702,2026092703)],float)
   if np.isnan(vals).all():continue
   for y in vals:ax.plot(range(1,31),y,color=colors[arm],alpha=.20,lw=.7)
   ax.plot(range(1,31),np.nanmean(vals,axis=0),color=colors[arm],label=arm,lw=2)
 for ax,title in zip(axs,('As native-scale smoothed CE (0.01)','As per-example cosine gradient proxy','Frozen teacher correct fraction on As')):
  ax.set_title(title,loc='left',fontsize=9);ax.set_xlabel('Cumulative epoch');ax.axvline(13.5,color='#aaa',lw=.7,ls=':');ax.axvline(20.5,color='#aaa',lw=.7,ls='--');ax.legend(frameon=False)
 axs[2].set_ylim(0,1);fig.text(.07,.01,'Thin: each student seed; bold: mean. All complete records parsed. Dotted: U starts; dashed: optimizer reset. Gradient proxy excludes outer weights.',fontsize=8,color='#555');fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'confirmation_training.png',bbox_inches='tight');fig.savefig(out/'confirmation_training.pdf',bbox_inches='tight');plt.close(fig)
 return dict(status='FIGURES_EXPORTED',files=[p.name for p in out.iterdir()])
