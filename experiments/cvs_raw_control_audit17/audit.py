"""Full historical source records;descriptive differences,not candidate promotion."""
import argparse,csv,gzip,json,math,re,statistics
from collections import Counter
from itertools import combinations
from pathlib import Path
from . import design as d

def records(p):
 opener=gzip.open if str(p).endswith('.gz') else open
 with opener(p,'rt',encoding='utf8') as f:
  for row in f:yield json.loads(row)
def near(x,y):
 if not math.isfinite(x) or not math.isfinite(y) or abs(x-y)>2e-5*max(1,abs(x),abs(y)):raise ValueError('Actual decomposed raw8L objective differs')
def metric(labels,preds):
 if len(labels)!=1260 or len(preds)!=1260 or Counter(labels)!=Counter({i:210 for i in range(6)}) or any(type(i) is not int or i not in range(6) for i in labels+preds):raise ValueError('Exact1260heldsource/6classes source readout')
 cm=[[0]*6 for _ in range(6)]
 for a,b in zip(labels,preds):cm[a][b]+=1
 fs=[]
 for i in range(6):
  den=sum(cm[i])+sum(r[i] for r in cm);fs.append(2*cm[i][i]/den if den else 0.)
 return dict(count=1260,accuracy=sum(cm[i][i] for i in range(6))/1260,macro_F1=sum(fs)/6,confusion=cm)
def recipe(c,a):
 keys=('F_base','CROSS_RX','Ag','As','ordinary_L','alpha','label_smoothing','use_aug','sinc_coordinates','raw_weight','ordinary_weight','s_slot_weight','L_forwards_per_packet','expected_L_packets','same_IQ_within_each_pair','base_raw_total_weight','extra_pair_weight','no_new_U_augmentation')
 native=('seed','epochs','label_epochs','pseudo_epochs','lr','weight_decay','batch_size','amp','use_aug','strong_noise_std','label_smoothing')
 return dict(feature_extractor=c['feature_extractor'],supcon=c['supcon'],precision=c['numerical_policy'],mechanism={k:c['training_mechanism'][k] for k in keys},native={k:a[k] for k in native},ordinary_args={k:v for k,v in a.items() if k.startswith('aug_')})
def prefix(a,b,value=lambda x:x,threshold=0):
 return next((i for i,(x,y) in enumerate(zip(a,b),1) if (abs(value(x)-value(y))>threshold if isinstance(value(x),(float,int)) else value(x)!=value(y))),None)
def analyze(inp,out):
 inp,out=Path(inp),Path(out);init=d.read(out/'initial_states.json');data={};trajectories=[];rowresults=[]
 if init['initial_checkpoint_loads']!=5 or init['trained_checkpoint_loads'] or init['IQ_reads'] or init['network_forwards'] or init['optimizer_updates'] or init['target_access']:raise ValueError('Only initial source tensor comparison,not new model execution')
 for ref in d.sources():
  k=ref['key'];b=inp/k;z=d.verify_source(b,ref);steps=list(records(b/'satmix_steps.jsonl.gz'));seq=list(records(b/'training_sequence.jsonl.gz'));native=list(records(b/'step_metrics.jsonl.gz'));units=list(records(b/'unit_steps.jsonl.gz'));epochs=list(records(b/'epoch_metrics.jsonl'));more=list(records(b/'metrics_epoch.jsonl'));stdout=(b/'stdout.log').read_text(encoding='utf8',errors='replace')
  if any(len(v)!=4440 for v in (steps,native,units)) or len(seq)!=5994 or Counter(v['role'] for v in seq)!=Counter(L=4440,U=1554) or len(epochs)!=20 or len(more)!=20 or re.search(r'Traceback|RuntimeError:|FloatingPointError:|CUDA out of memory',stdout):raise ValueError('Full4440updates/5994LU/20epochs/stdout required')
  for name in ('epoch_metrics.csv','metrics_epoch.csv'):
   with (b/name).open(encoding='utf-8-sig',newline='') as f:
    if len(list(csv.DictReader(f)))!=20:raise ValueError('Full saved CSV epochs required')
  grad=[]
  for i,(a,n,u) in enumerate(zip(steps,native,units),1):
   e=(i-1)//222+1;m=n['metrics']
   if (a['step'],a['epoch'],n['step'],n['epoch'],u['step'],u['physical_clip_calls'])!=(i,e,i,e,i,i) or a['As_packets'] or a['sat_raw_MSE'] or a['extra_raw_MSE'] or a['extra_pair_weight']!=.25 or a['base_raw_total_weight']!=.75 or a['base_ordinary_weight']!=.25 or a['total_pair_weight']!=1.25:raise ValueError('Actual all-epoch8L raw1/ordinary.25 control required')
   near(a['base_total_L'],.5*(a['raw_CE']+a['raw_KL'])+.25*(a['ordinary_CE']+a['ordinary_KL'])+.25*(a['s_slot_CE']+a['s_slot_KL'])+a['supcon_weight']*a['supcon']);near(a['extra_pair_loss'],a['extra_pair_CE']+a['extra_pair_KL']);near(a['total_L'],a['base_total_L']+.25*a['extra_pair_loss'])
   if m['train/pseudo_truth_available'] or m['train/optimizer_step_applied']!=1 or m.get('train/skipped_nonfinite_grad',0) or m.get('train/skipped_nonfinite_loss',0) or not math.isfinite(m['train/grad_before_clip']):raise ValueError('HiddenU/no skips/finite actual update scope required')
   grad.append(m['train/grad_before_clip'])
  scores={};views={}
  for v in d.VIEWS:
   q=d.read(b/'source_readouts'/(v+'.json'));sc=metric(q['labels'],q['predictions']);near(sc['accuracy'],q['accuracy']);near(sc['macro_F1'],q['macro_F1']);near(sc['accuracy'],z['result']['views'][v]['accuracy'])
   ids=q['physical_ids']
   if len(ids)!=1260 or len({tuple(p) for p in ids})!=1260 or any(len(p)!=5 or p[0]!=1 or p[1]!=1 or p[2] not in (1,2,3) or p[3] not in range(70) for p in ids):raise ValueError('Exact saved heldRX1L physical readout only')
   scores[v]=sc;views[v]=q
  for e,epoch in enumerate(epochs,1):
   if epoch['epoch']!=e or epoch['epochs']!=20 or epoch['label_epochs']!=13 or epoch['pseudo_epochs']!=7:raise ValueError('FixedE20/13L7U epoch chronology required')
   ss=steps[(e-1)*222:e*222];g=grad[(e-1)*222:e*222];trajectories.append(dict(control=k,epoch=e,total_L=statistics.mean(a['total_L'] for a in ss),ordinary_MSE=statistics.mean(a['ordinary_raw_MSE'] for a in ss),gradient_preclip=statistics.mean(g),seenV_percent=epoch['val_tx_acc'],lr=epoch['lr'],pseudo_selected_epoch_SUM=epoch['train_pseudo_selected']))
  NN=sum(z['result'][n] for n in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets'))+z['result']['held_forward_packets'];rowresults.append(dict(key=k,source_run=ref['run_id'],arm=ref['arm'],scores=scores,historical_model_epochs=20,historical_optimizer_updates=4440,historical_NN=NN,stored_steps=4440,stored_LU_records=5994,full_stdout_lines=len(stdout.splitlines()),gradient_zero=sum(g==0 for g in grad),seenV=z['result']['seen_V']))
  data[k]=dict(**z,steps=steps,sequences=seq,grad=grad,views=views,recipe=recipe(z['config'],z['args']))
 pairs=[]
 for ka,kb in combinations(data,2):
  a,b=data[ka],data[kb];p=dict(first=ka,second=kb,initial_states_equal=init['pairwise'][ka+'|'+kb]['tensors_equal'],source_role_IDs_exact=a['contract']['role_ids']==b['contract']['role_ids'],normalized_recipe_exact=a['recipe']==b['recipe'],all_native_arg_changed_keys=[k for k in sorted(set(a['args'])|set(b['args'])) if a['args'].get(k)!=b['args'].get(k)],L_U_sequence_exact=a['sequences']==b['sequences'],first_sequence_difference=prefix(a['sequences'],b['sequences']),stored_ordinary_MSE_exact=all(x['ordinary_raw_MSE']==y['ordinary_raw_MSE'] for x,y in zip(a['steps'],b['steps'])),ordinary_MSE_max_absolute_difference=max(abs(x['ordinary_raw_MSE']-y['ordinary_raw_MSE']) for x,y in zip(a['steps'],b['steps'])),first_ordinary_MSE_difference=prefix(a['steps'],b['steps'],lambda x:x['ordinary_raw_MSE']),first_total_L_difference=prefix(a['steps'],b['steps'],lambda x:x['total_L']),first_total_L_difference_above1e_minus5=prefix(a['steps'],b['steps'],lambda x:x['total_L'],1e-5),first_gradient_difference=prefix(a['grad'],b['grad']),first_gradient_difference_above1e_minus5=prefix(a['grad'],b['grad'],threshold=1e-5),first_step_gradient_absolute_difference=abs(a['grad'][0]-b['grad'][0]),actual_six_channel_configs_equal=d.read(inp/ka/'actual_channel.json')['actual_configs']==d.read(inp/kb/'actual_channel.json')['actual_configs'],eval_views={},all_eval_physics_exact=True)
  for v in d.VIEWS:
   aa,bb=a['views'][v],b['views'][v]
   if aa['physical_ids']!=bb['physical_ids'] or aa['labels']!=bb['labels']:raise ValueError('Source evaluation must pair the same held physical observations/labels')
   if v!='clean':p['all_eval_physics_exact']&=aa.get('physics')==bb.get('physics') and aa.get('physics') is not None
   y=aa['labels'];pa=aa['predictions'];pb=bb['predictions'];rep=sum(t!=x and t==z for t,x,z in zip(y,pa,pb));br=sum(t==x and t!=z for t,x,z in zip(y,pa,pb));p['eval_views'][v]=dict(second_minus_first_accuracy_pp=100*(rep-br)/1260,repairs=rep,breaks=br,hard_disagreement_count=sum(x!=z for x,z in zip(pa,pb)),hard_disagreement_fraction=sum(x!=z for x,z in zip(pa,pb))/1260)
  pairs.append(p)
 dispersion={}
 for v in d.VIEWS:
  vals=[r['scores'][v]['accuracy']*100 for r in rowresults];dispersion[v]=dict(count=5,mean_percent=statistics.mean(vals),sample_SD_pp=statistics.stdev(vals),min_percent=min(vals),max_percent=max(vals),range_pp=max(vals)-min(vals),not_independent_modelseeds=True,not_confidence_interval=True)
 common={k:all(p[k] for p in pairs) for k in ('initial_states_equal','source_role_IDs_exact','normalized_recipe_exact','L_U_sequence_exact','stored_ordinary_MSE_exact','actual_six_channel_configs_equal','all_eval_physics_exact')}
 summary=dict(status='ANALYZED',execution='VERIFIED',run_id=d.RUN,source_only=True,target_access=False,cohort_fixed_before_collection=True,scores_previously_exposed=True,all_pair_compatibility=common,rows=rowresults,pairs=pairs,dispersion=dispersion,historical_model_epochs=100,historical_updates=22200,historical_NN=sum(r['historical_NN'] for r in rowresults),historical_NN_definition='Direct sum of four saved student/teacher counters plus held readout counter;no arbitrary smoke subtraction;not a count of distinct physical IDs',new_model_epochs=0,new_updates=0,new_NN=0,new_IQ_reads=0,initial_CP_reads=5,trained_CP_reads=0,source_saved_predictions_recounted=44100,complete_native_view_unit_records=66600,complete_LU_records=29970,complete_epoch_rows=100,head_fit_updates=0,model_seeds=[d.SEED],current_running_scientificcriteria_changed=False,candidate_promotion=False,closed_routes_reopened=False,goal_complete=False,limits=['Same nominal recipe descriptive cohort,not fiveindependent seeds or a new controlled replicate experiment','Stored ordinary scalar MSE equality doesnotprove exactIQ equality;no originalIQ replay','This audit cannot uniquely attribute variability to CUDA/numerics/loader/dropout or implementation','No retrospective candidate promotion,thresholdchange or closedroute rescue','All source-only results;no newAg/As/joint/target/crossseed performance proof'])
 d.write(out/'analysis_summary.json',summary);d.write(out/'training_curves.json',trajectories);print(json.dumps(dict(status='ANALYZED',compatibility=common,dispersion=dispersion,new_NN=0)));return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();analyze(a.input,a.output)
