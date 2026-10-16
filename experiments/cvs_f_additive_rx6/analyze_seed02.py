"""Independent recount of completed RX1 seed02 source run; no model, IQ, target or job mutation."""
import argparse,csv,gzip,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def score(labels,predictions):
    if len(labels)!=len(predictions) or any(type(v) is not int or v not in range(6) for v in labels+predictions):raise ValueError('Legal complete source decisions required')
    cm=[[0]*6 for _ in range(6)]
    for y,p in zip(labels,predictions):cm[y][p]+=1
    f1=[]
    for c in range(6):
        tp=cm[c][c];fp=sum(row[c] for row in cm)-tp;fn=sum(cm[c])-tp
        f1.append(2*tp/max(1,2*tp+fp+fn))
    return dict(count=len(labels),accuracy=sum(cm[i][i] for i in range(6))/len(labels),macro_F1=sum(f1)/6,confusion=cm)


OUT=None
RUN='20261016-phase1-F-crossRX-additiveAs-confirm-sourceRX1-e20-s2026092702-m3-r01'
SEED=2026092702
ARMS=('D00','D_raw','D_As')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def records(p):
 with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def finite(v):
 if isinstance(v,float):assert math.isfinite(v)
 elif isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,list):
  for x in v:finite(x)
def analyze(arms=ARMS):
 scores={};predictions={};execution={};curves=[];seqs=[];args_all=[]
 for arm in arms:
  row=arm+'-RX1-s'+str(SEED);src=OUT/row/'source';r=read(src/'satmix_result.json');init=read(src/'initialization.json');done=read(src/'completion.json');frozen=read(src/'source_frozen.json');audit=read(src/'additive_audit.json')
  contract=read(src/'source_contract.json');assert contract['source_receivers']==[3,4,6,8] and {k:len(v) for k,v in contract['role_ids'].items()}=={'L_s':5040,'U_s':45360,'V':21600}
  assert init['scratch_only'] and not init['checkpoint_sources'] and not init['ancestors'] and not init['target_contact']
  assert done['epoch']==frozen['epoch']==20 and done['optimizer_steps']==4440 and not done['target_access'] and done['head_fit_updates']==0 and frozen['selection']=='final_only'
  assert r['actual_steps']==r['additive_steps']==4440 and r['L_forward_packets']==4546560 and r['extra_L_forward_packets']==1136640 and r['supcon_applied']==3330 and r['supcon_layer_forward_terms']==13320
  assert r['Ag'] is False and r['As']==(arm=='D_As') and not r['U_truth_read'] and not r['target_access'] and not r['head_fit_updates'] and r['teacher_additive_calls']==0 and not r['checkpoint_sources']
  expected_As=568320 if arm=='D_As' else 0;weight=0. if arm=='D00' else .25
  assert r['As_training_observations']==expected_As and r['total_pair_weight']==1+weight and audit['calls']==4440 and audit['As_generated_packets']==expected_As and audit['no_As_U_or_teacher']
  scores[arm]={};predictions[arm]={}
  for view in VIEWS:
   p=read(src/'source_readouts'/(view+'.json'));s=score(p['labels'],p['predictions'])
   assert s['count']==1260 and len({tuple(x) for x in p['physical_ids']})==1260 and {x[1] for x in p['physical_ids']}=={1}
   assert math.isclose(s['accuracy'],r['views'][view]['accuracy'],abs_tol=1e-12) and math.isclose(s['macro_F1'],r['views'][view]['macro_F1'],abs_tol=1e-12)
   if view!='clean':assert p['labels']==predictions[arm]['clean']['labels'] and p['physical_ids']==predictions[arm]['clean']['physical_ids']
   scores[arm][view]=s;predictions[arm][view]=p
  native=records(src/'metrics_epoch.jsonl');compact=records(src/'epoch_metrics.jsonl')
  with (src/'metrics_epoch.csv').open(encoding='utf8',newline='') as f:table=list(csv.DictReader(f))
  assert [v['epoch'] for v in native]==list(range(1,21)) and len(compact)==len(table)==20
  for a,b,c in zip(native,compact,table):
   for k in ('train_loss_tx_labeled','train_loss_unlabeled','val_tx_acc','lr'):assert math.isclose(float(a[k]),float(b[k]),abs_tol=1e-10) and math.isclose(float(a[k]),float(c[k]),abs_tol=1e-10)
  steps=records(src/'step_metrics.jsonl.gz');mix=records(src/'satmix_steps.jsonl.gz');unit=records(src/'unit_steps.jsonl.gz')
  assert all([v['step'] for v in values]==list(range(1,4441)) for values in (steps,mix,unit))
  terms=As=truth=skips=bn=0;formula_error=base_error=0.;aggregate={}
  for n,v,u in zip(steps,mix,unit):
   finite(n);finite(v);finite(u)
   epoch=(v['step']-1)//222+1;sw=.1*min(max((epoch-5)/10,0.),1.)
   assert v['epoch']==epoch and v['same_IQ_pairs'] and v['all_L_supervised'] and v['extra_view_once'] and v['extra_same_IQ_pair']
   assert v['As_packets']==0 and v['extra_As_packets']==(128 if arm=='D_As' else 0) and v['extra_input']==('As(raw)' if arm=='D_As' else 'raw')
   assert v['extra_pair_weight']==weight and v['base_raw_total_weight']==.75 and v['base_ordinary_weight']==.25 and v['total_pair_weight']==1+weight
   assert math.isclose(v['supcon_weight'],sw,abs_tol=1e-12) and len(v['supcon_details'])==(4 if epoch>5 else 0)
   assert sum(v['scene_counts'].values())==sum(v['state_counts'].values())==sum(v['lock_counts'].values())==0
   terms+=len(v['supcon_details']);As+=v['extra_As_packets'];bn+=v['extra_BN_batch_updates']
   if arm!='D_As':assert v['extra_raw_MSE']==0
   assert v['sat_raw_MSE']==0
   base=.5*(v['raw_CE']+v['raw_KL'])+.25*(v['ordinary_CE']+v['ordinary_KL'])+.25*(v['s_slot_CE']+v['s_slot_KL'])+sw*v['supcon']
   extra=v['extra_pair_CE']+v['extra_pair_KL'];total=base+weight*extra
   base_error=max(base_error,abs(base-v['base_total_L']))
   formula_error=max(formula_error,abs(total-v['total_L']),abs(extra-v['extra_pair_loss']),abs(weight*extra-v['extra_pair_weighted']))
   nm=n['metrics']
   for k in ('total_L','base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_weight','extra_As_packets'):assert math.isclose(float(nm['satmix/'+k]),v[k],abs_tol=1e-6)
   truth+=int(nm.get('train/pseudo_truth_available',0) or 0);skips+=int(nm.get('train/skipped_nonfinite_loss',0) or nm.get('train/skipped_nonfinite_grad',0) or 0)
   assert u['physical_clip_calls']==v['step']
   sums=aggregate.setdefault(epoch,{})
   for k in ('total_L','base_total_L','raw_CE','ordinary_CE','s_slot_CE','raw_KL','ordinary_KL','s_slot_KL','supcon','ordinary_raw_MSE','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair_weighted','extra_raw_MSE'):sums[k]=sums.get(k,0.)+v[k]/222
  assert terms==13320 and As==expected_As and truth==skips==0 and formula_error<1e-5 and base_error<1e-5 and bn==audit['extra_BN_updates']
  for k in ('base_total_L','extra_pair_loss'):assert math.isclose(aggregate[20][k],r['final_base_L_objective' if k=='base_total_L' else 'final_extra_pair_objective'],abs_tol=1e-8)
  assert r['final_native_logged_L_CE']==r['final_L_CE'] and math.isclose(compact[-1]['train_loss_tx_labeled'],r['final_L_CE'],abs_tol=1e-10)
  for v in compact:curves.append(dict(row=row,arm=arm,**v,**aggregate[v['epoch']]))
  coverage=read(src/'satellite_supervision_coverage.json');assert coverage['total']==As and sum(v['count'] for v in coverage['cells'])==As and not coverage['confidence_gate']
  if arm=='D_As':
   assert {v['scene'] for v in coverage['cells']}==set(VIEWS[1:]) and {v['RX'] for v in coverage['cells']}=={3,4,6,8} and {v['TX'] for v in coverage['cells']}==set(range(6)) and {v['day'] for v in coverage['cells']}=={1,2,3}
  seq=records(src/'training_sequence.jsonl.gz');seqs.append(seq);assert all(p[1] in (3,4,6,8) for v in seq for p in v['ids'])
  observed={role:sum(len(v['ids']) for v in seq if v['role']==role) for role in ('L','U')}
  unique={role:len({tuple(p) for v in seq if v['role']==role for p in v['ids']}) for role in ('L','U')}
  assert observed=={'L':568320,'U':396368} and unique=={'L':5040,'U':45360}
  args=read(src/'actual_native_args.json');args_all.append(args)
  assert args['seed']==SEED and args['use_aug'] and args['source_fold_held_rx']==1 and args['label_epochs']==13 and args['pseudo_epochs']==7 and args['aug_enable_pa_normal'] and args['aug_p_dac']==0 and not args['aug_enable_class_signature']
  ordinary=read(src/'ordinary_augmentation_actual.json');assert ordinary['no_second_pair_redraw'] and ordinary['practical_input_is_raw']
  stdout=(OUT/'logs'/(row+'.stdout.log')).read_text(encoding='utf8',errors='replace')
  assert not any(s in stdout for s in ('Traceback (most recent call last)','CUDA out of memory','FloatingPointError','SIGKILL')) and 'SMOKE PASS own scratch checkpoint; query_read=false' in stdout
  counters={k:r[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets')}
  NN=sum(counters.values())-2+r['held_forward_packets']
  execution[arm]=dict(full_steps=4440,full_epochs=20,full_stdout_lines=len(stdout.splitlines()),stdout_warnings=[v for v in stdout.splitlines() if 'warning' in v.lower()],native_nonfinite_placeholder_fields=sorted({k for v in native for k,x in v.items() if isinstance(x,float) and not math.isfinite(x)}),all_step_values_finite_or_explicit_null=True,max_loss_recount_error=formula_error,max_base_loss_recount_error=base_error,supcon_layer_forward_terms=terms,L_forward_packets=r['L_forward_packets'],extra_L_forward_packets=r['extra_L_forward_packets'],As_training_observations=As,As_source_eval_observations=7560,ordinary_generated_packets=568320,physical_observations=observed,unique_physical_training=unique,teacher_additive_calls=0,teacher_inherited_counter_offset_removed=2,instrumented_counters=counters,actual_network_forward_packets=NN,held_forward_packets=r['held_forward_packets'],elapsed_training_seconds=r['elapsed_seconds'],final_native_logged_first_ordinary_CE=r['final_L_CE'],final_base_L_objective=r['final_base_L_objective'],final_extra_pair_objective=r['final_extra_pair_objective'],total_pair_weight=r['total_pair_weight'],seen_V=r['seen_V'],tracked_BN_module_names=audit['tracked_BN_module_names'],extra_BN_updates=bn,coverage=coverage,ordinary_augmentation=ordinary)
 if len(arms)==1:
  value=dict(status='COMPLETED_SOURCE_ROW_ANALYZED',run_id=RUN,arm=arms[0],scores=scores,execution=execution,source_only=True,target_access=False,comparison_available=False);dump(OUT/('analysis_'+arms[0]+'.json'),value);return value
 assert arms==ARMS and all(s==seqs[0] for s in seqs)
 differences=[sorted(k for k,v in args_all[0].items() if v!=a.get(k)) for a in args_all[1:]]
 assert all(set(x)=={'candidate_id','output_dir'} for x in differences)
 complete=read(OUT/'completion.json');result=read(OUT/'result.json');assert complete['status']=='ANALYZED' and complete['rows']==3 and complete['new_epochs']==60 and complete['new_optimizer_steps']==13320
 paired={};strata=[];comparisons={};conditions={}
 for ref in ARMS[:2]:
  paired[ref]={}
  for view in VIEWS:
   b,s=predictions[ref][view],predictions['D_As'][view];assert b['labels']==s['labels'] and b['physical_ids']==s['physical_ids']
   repair=sum(p!=y and q==y for p,q,y in zip(b['predictions'],s['predictions'],b['labels']));broken=sum(p==y and q!=y for p,q,y in zip(b['predictions'],s['predictions'],b['labels']))
   delta=100*(scores['D_As'][view]['accuracy']-scores[ref][view]['accuracy']);assert math.isclose(delta,100*(repair-broken)/1260,abs_tol=1e-10)
   paired[ref][view]=dict(accuracy_gain_pp=delta,macro_F1_gain_pp=100*(scores['D_As'][view]['macro_F1']-scores[ref][view]['macro_F1']),repaired=repair,broken=broken)
   for dimension in ('TX','day'):
    for value in (range(6) if dimension=='TX' else (1,2,3)):
     take=[i for i,(y,p) in enumerate(zip(b['labels'],b['physical_ids'])) if (y if dimension=='TX' else p[2])==value]
     acc={a:sum(predictions[a][view]['predictions'][i]==b['labels'][i] for i in take)/len(take) for a in (ref,'D_As')}
     strata.append(dict(reference=ref,view=view,dimension=dimension,value=value,count=len(take),reference_accuracy=acc[ref],D_As_accuracy=acc['D_As'],gain_pp=100*(acc['D_As']-acc[ref])))
  delta={v:paired[ref][v]['accuracy_gain_pp'] for v in VIEWS};mf1=paired[ref]['clean']['macro_F1_gain_pp'];dv=100*(execution['D_As']['seen_V']['source_val_accuracy']-execution[ref]['seen_V']['source_val_accuracy']);urban=sum(delta[v] for v in VIEWS[4:])/3;protect=0. if ref=='D00' else -.5
  part=dict(clean=delta['clean']>=protect,clean_macroF1=mf1>=protect,low_urban=delta[VIEWS[-1]]>=5.,urban_mean=urban>=3.,practical_protection=all(delta[v]>=-.5 for v in VIEWS[1:]),seen_V=dv>=-.5)
  conditions.update({ref+'_'+k:v for k,v in part.items()});comparisons[ref]=dict(conditions=part,all_view_gain_pp=delta,clean_macroF1_gain_pp=mf1,seen_V_gain_pp=dv,urban_mean_gain_pp=urban)
 conditions['healthy']=all(v['final_base_L_objective']<=.5 for v in execution.values())
 qualified=all(conditions.values());assert qualified==complete['qualified'] and conditions==result['qualification']['conditions']
 for ref in ARMS[:2]:
  for v in VIEWS:assert math.isclose(comparisons[ref]['all_view_gain_pp'][v],result['qualification']['comparisons'][ref]['all_view_gain_pp'][v],abs_tol=1e-12)
 gaps={a:{v:dict(clean_minus_suburban_pp=100*(s['clean']['accuracy']-s[v]['accuracy']),absolute_gap_pp=100*abs(s['clean']['accuracy']-s[v]['accuracy'])) for v in VIEWS[1:4]} for a,s in scores.items()}
 value=dict(status='ANALYZED',actual_execution='VERIFIED',run_id=RUN,source_only=True,model_seed=SEED,qualification=dict(qualified=qualified,conditions=conditions,comparisons=comparisons),scores=scores,paired=paired,suburban_gaps=gaps,execution=execution,all_L_U_sequences_identical=True,native_argument_differences=differences,CE_scope='Native final_L_CE is first ordinary smoothed CE;mean readout CE is runtime defaulthead report,hard predictions lack logits for independent CE recount;common base objective includes CE/RDrop/SupCon',new_model_epochs=60,new_optimizer_updates=13320,dispatcher_smoke_NN_packets=2,total_actual_NN_forward_packets=sum(v['actual_network_forward_packets'] for v in execution.values())+2,target_access=False,head_fit_updates=0,Ag_efficacy_proved=False,joint_goal_complete=False,missing=['Independent source confirmation','Validated Ag','Same-contract B00/B10/B01/B11 and interaction','Formal target clean+sixpractical','Cross-seed joint verification'])
 dump(OUT/'analysis_summary.json',value)
 for name,values in (('training_curves.csv',curves),('source_strata.csv',strata)):
  fields=sorted({k for v in values for k in v})
  with (OUT/name).open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)
 return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();OUT=Path(a.input);v=analyze();print(json.dumps({k:v[k] for k in ('status','qualification','total_actual_NN_forward_packets')}))
