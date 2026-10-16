"""Full independent source scoring and telemetry recount;no IQ/Torch/weights/target."""
import argparse,csv,gzip,json,math,statistics
from collections import Counter
from itertools import zip_longest
from pathlib import Path
RUN='20261017-phase1-FcrossRX-noisepair-sourceRX1-e20-s2026092701-m3-r01'
ARMS=('RAW10','INDEP','ANTI');NEW=ARMS;FIT=(3,4,6,8);SEED=2026092701
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def stream(p):
 with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def finite(v):
 if isinstance(v,float):assert math.isfinite(v)
 elif isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,list):
  for x in v:finite(x)
def score(y,p):
 assert len(y)==len(p)==1260 and all(type(v) is int and v in range(6) for v in y+p)
 cm=[[0]*6 for _ in range(6)]
 for a,b in zip(y,p):cm[a][b]+=1
 f1=[2*cm[i][i]/max(1,sum(cm[i])+sum(r[i] for r in cm)) for i in range(6)]
 return dict(count=len(y),accuracy=sum(cm[i][i] for i in range(6))/len(y),macro_F1=sum(f1)/6,confusion=cm)
def describe(v):
 v=sorted(v);n=len(v)
 def q(p):return v[min(n-1,int(p*(n-1)))] if n else None
 return dict(count=n,min=q(0),p10=q(.1),median=q(.5),p90=q(.9),max=q(1),mean=statistics.fmean(v) if n else None)

COMMON=('physical_id','TX','RX','day','step','epoch','scene','session','state','lock','quality_snr_db','output_frequency_hz','residual_scale','residual_kernel_energy','residual_kernel_length','numerical_filter_latency_samples','boundary','phase_tracking_applied','iq_compensation_applied','EQ','G','nonnoise_latents_shared','AGC_recomputed_per_view','added_noise_variance_reference')
def pair_valid(a,b):
 assert a['view']==0 and b['view']==1 and all(a[k]==b[k] for k in COMMON)
 assert a['RX'] in FIT and a['day'] in (1,2,3) and a['TX'] in range(6) and a['G']==1 and a['EQ'] is False and a['nonnoise_latents_shared'] and a['AGC_recomputed_per_view'] and a['added_noise_variance_reference']==.001 and a['boundary']=='synthetic_reflect' and a['numerical_filter_latency_samples']==24
 finite(a);finite(b)
def independent_qualify(results,scores):
 conditions={};comparisons={};s=results['ANTI'];eps=1e-10
 for ref in ('RAW10','INDEP'):
  delta={v:100*(scores['ANTI'][v]['accuracy']-scores[ref][v]['accuracy']) for v in VIEWS};f1=100*(scores['ANTI']['clean']['macro_F1']-scores[ref]['clean']['macro_F1']);dv=100*(s['seen_V']['source_val_accuracy']-results[ref]['seen_V']['source_val_accuracy']);urban=sum(delta[v] for v in VIEWS[4:])/3;clean=0. if ref=='RAW10' else -.5
  checks=dict(clean=delta['clean']>=clean-eps,clean_macroF1=f1>=clean-eps,urban=urban>=(3. if ref=='RAW10' else 1.)-eps,low=delta[VIEWS[-1]]>=(5. if ref=='RAW10' else 1.)-eps,all_practical=all(delta[v]>=-.5-eps for v in VIEWS[1:]),seenV=dv>=-.5-eps)
  conditions.update({ref+'_'+k:v for k,v in checks.items()});comparisons[ref]=dict(all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,seen_V_gain_pp=dv,urban_mean_gain_pp=urban,conditions=checks)
 conditions['healthy']=all(r['actual_steps']==4440 and r['L_forward_packets']==5683200 and r['extra_L_forward_packets']==2273280 and r['noise_pair_steps']==4440 and r['supcon_applied']==3330 and r['supcon_layer_forward_terms']==13320 and r['final_base_L_objective']<=.5 and r['noise_pair_physical_presentations']==568320 and r['As_training_observations']==(0 if arm=='RAW10' else 1136640) and not r['teacher_noise_pair_calls'] and not r['head_fit_updates'] and not r['U_truth_read'] and not r['target_access'] and not r['checkpoint_sources'] and r['held_RX']==1 and r['fit_RX']==list(FIT) for arm,r in results.items())
 return dict(qualified=all(conditions.values()),conditions=conditions,comparisons=comparisons)

def analyze(root):
 root=Path(root);request=read(root/'request.json');configs=request['rows'];completion=read(root/'completion.json')
 assert request['source_only'] and len(configs)==3 and {c['arm'] for c in configs}==set(ARMS) and all(c['run_id']==RUN and c['model_seed']==SEED and c['epochs']==20 and not c['target_access'] and not c['baseline_reuse'] for c in configs)
 assert completion['status']=='ANALYZED' and completion['rows']==completion['new_rows']==3 and completion['reused_rows']==0 and completion['new_epochs']==60 and completion['new_optimizer_steps']==13320 and not (root/'failure.json').exists()
 sources={a:root/(a+'-RX1-s'+str(SEED))/'source' for a in ARMS};results={};scores={};pred={};execution={};contracts={};arguments={};curves=[];rawL=None
 for arm,src in sources.items():
  r=read(src/'satmix_result.json');results[arm]=r;init=read(src/'initialization.json');done=read(src/'completion.json');freeze=read(src/'source_frozen.json');contract=read(src/'source_contract.json');audit=read(src/'noise_pair_audit.json');args=read(src/'actual_native_args.json');arguments[arm]=args
  assert init['scratch_only'] and not any(init[k] for k in ('checkpoint_sources','ancestors','target_contact')) and done['epoch']==freeze['epoch']==20 and done['optimizer_steps']==4440 and freeze['selection']=='final_only' and not any(done[k] for k in ('target_access','head_fit_updates','checkpoint_sources','ancestors'))
  assert contract['source_receivers']==list(FIT) and contract['held_RX']==1 and {k:len(v) for k,v in contract['role_ids'].items()}=={'L_s':5040,'U_s':45360,'V':21600};contracts[arm]=contract['role_ids']
  assert r['arm']==arm and r['actual_steps']==r['noise_pair_steps']==r['additive_steps']==4440 and r['L_forward_packets']==5683200 and r['extra_L_forward_packets']==2273280 and r['supcon_applied']==3330 and r['supcon_layer_forward_terms']==13320 and not r['Ag'] and r['As']==(arm!='RAW10')
  assert not any(r[k] for k in ('teacher_noise_pair_calls','head_fit_updates','U_truth_read','target_access','checkpoint_sources')) and r['As_training_observations']==(0 if arm=='RAW10' else 1136640) and r['noise_pair_physical_presentations']==568320 and r['total_pair_weight']==1.25
  assert audit['calls']==4440 and audit['physical_presentations']==568320 and audit['normal_pair_count']==5 and audit['base_raw_total_weight']==.75 and audit['base_ordinary_weight']==.25 and audit['extra_BN_updates']==0 and audit['no_noise_U_or_teacher']
  assert args['seed']==SEED and args['epochs']==20 and args['label_epochs']==13 and args['pseudo_epochs']==7 and args['use_aug'] and args['source_fold_held_rx']==1 and not args['baseline_ckpt'] and not args['teacher_ckpt'] and args['aug_enable_pa_normal'] and args['aug_p_dac']==0 and not args['aug_enable_class_signature']
  scores[arm]={};pred[arm]={}
  for view in VIEWS:
   p=read(src/'source_readouts'/(view+'.json'));ss=score(p['labels'],p['predictions']);assert len(p['physical_ids'])==len({tuple(v) for v in p['physical_ids']})==1260 and {v[1] for v in p['physical_ids']}=={1}
   assert all(math.isclose(ss[k],r['views'][view][k],abs_tol=1e-12) for k in ('accuracy','macro_F1')) and len(p['margin_over30'])==1260 and all(math.isfinite(x) for x in p['margin_over30'])
   assert all((m<=1e-12 or y==q) and (m>=-1e-12 or y!=q) for y,q,m in zip(p['labels'],p['predictions'],p['margin_over30']))
   if view!='clean':
    assert len(p['physics'])==1260
    for m in p['physics']:assert m['processing_route']=='residual' and m['mode']=='post_sync' and m['RX']==1 and m['fs_hz']==25000000 and m['fc_hz']==2462000000 and not m['channel_equalization_applied'] and not m['true_total_snr_known'] and m['quality_source']=='white_input_expected_power_proxy' and m['boundary']==m['noise_boundary']=='synthetic_reflect' and m['numerical_filter_latency_samples']==24 and m['added_noise_variance_reference']==.001
   scores[arm][view]=ss;pred[arm][view]=p
  native=list(stream(src/'metrics_epoch.jsonl'));compact=list(stream(src/'epoch_metrics.jsonl'))
  with (src/'metrics_epoch.csv').open(encoding='utf8',newline='') as f:table=list(csv.DictReader(f))
  assert [x['epoch'] for x in native]==list(range(1,21)) and len(compact)==len(table)==20
  for a,b,c in zip(native,compact,table):
   for k in ('train_loss_tx_labeled','train_loss_unlabeled','val_tx_acc','lr'):assert math.isclose(float(a[k]),float(b[k]),abs_tol=1e-10) and math.isclose(float(a[k]),float(c[k]),abs_tol=1e-10)
  counts=terms=truth=skips=0;formula_error=base_error=0.;agg={}
  for n,v,u in zip_longest(stream(src/'step_metrics.jsonl.gz'),stream(src/'satmix_steps.jsonl.gz'),stream(src/'unit_steps.jsonl.gz')):
   assert n is not None and v is not None and u is not None;counts+=1;finite(n);finite(v);finite(u);assert n['step']==v['step']==u['step']==counts and u['physical_clip_calls']==counts
   epoch=(counts-1)//222+1;weight=.1*min(max((epoch-5)/10,0.),1.);assert v['epoch']==epoch and v['same_IQ_pairs'] and v['all_L_supervised'] and v['extra_same_IQ_pair'] and not v['cross_noise_KL'] and v['L_forwards_per_sample']==10 and v['extra_pair_count']==2
   assert v['As_packets']==0 and v['extra_As_packets']==(0 if arm=='RAW10' else 256) and v['extra_input']==arm and v['extra_pair_weight']==.25 and v['base_raw_total_weight']==.75 and v['base_ordinary_weight']==.25 and v['total_pair_weight']==1.25 and v['extra_BN_batch_updates']==0
   assert math.isclose(v['supcon_weight'],weight,abs_tol=1e-12) and len(v['supcon_details'])==(4 if epoch>5 else 0) and v['sat_raw_MSE']==0 and sum(v['scene_counts'].values())==sum(v['state_counts'].values())==sum(v['lock_counts'].values())==0
   if arm=='RAW10':assert v['extra_raw_MSE']==v['noise_view_pair_MSE']==0
   terms+=len(v['supcon_details']);base=.5*(v['raw_CE']+v['raw_KL'])+.25*(v['ordinary_CE']+v['ordinary_KL'])+.25*(v['s_slot_CE']+v['s_slot_KL'])+weight*v['supcon'];extra=.5*(v['extra_pair0_CE']+v['extra_pair0_KL']+v['extra_pair1_CE']+v['extra_pair1_KL']);total=base+.25*extra
   base_error=max(base_error,abs(base-v['base_total_L']));formula_error=max(formula_error,abs(extra-v['extra_pair_loss']),abs(total-v['total_L']),abs(.25*extra-v['extra_pair_weighted']),abs(v['extra_pair_CE']-.5*(v['extra_pair0_CE']+v['extra_pair1_CE'])),abs(v['extra_pair_KL']-.5*(v['extra_pair0_KL']+v['extra_pair1_KL'])))
   for k in ('base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair_weighted','total_L','extra_As_packets','extra_raw_MSE','noise_view_pair_MSE'):assert math.isclose(float(n['metrics']['satmix/'+k]),v[k],abs_tol=1e-6)
   truth+=int(n['metrics'].get('train/pseudo_truth_available',0) or 0);skips+=int(n['metrics'].get('train/skipped_nonfinite_loss',0) or n['metrics'].get('train/skipped_nonfinite_grad',0) or 0)
   sums=agg.setdefault(epoch,{})
   for k in ('total_L','base_total_L','raw_CE','ordinary_CE','s_slot_CE','supcon','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair0_CE','extra_pair1_CE','extra_pair0_KL','extra_pair1_KL','extra_raw_MSE','noise_view_pair_MSE'):sums[k]=sums.get(k,0.)+v[k]/222
  assert counts==4440 and terms==13320 and truth==skips==0 and formula_error<1e-5 and base_error<1e-5 and math.isclose(agg[20]['base_total_L'],r['final_base_L_objective'],abs_tol=1e-8)
  for v in compact:curves.append(dict(arm=arm,**v,**agg[v['epoch']]))
  obs=Counter();unique={'L':set(),'U':set()};L=[];seqcount=0
  for item in stream(src/'training_sequence.jsonl.gz'):
   role=item['role'];assert role in unique and all(p[1] in FIT for p in item['ids']);obs[role]+=len(item['ids']);unique[role].update(tuple(p) for p in item['ids']);seqcount+=1
   if role=='L':assert len(item['ids'])==128;L.append(item['ids'])
  assert dict(obs)=={'L':568320,'U':396368} and {k:len(v) for k,v in unique.items()}=={'L':5040,'U':45360} and len(L)==4440
  if arm=='RAW10':rawL=L
  else:
   assert L==rawL
   for x,y in zip_longest(stream(sources['RAW10']/'training_sequence.jsonl.gz'),stream(src/'training_sequence.jsonl.gz')):assert x==y
  coverage=read(src/'satellite_supervision_coverage.json');assert coverage['total']==r['As_training_observations'] and not coverage['confidence_gate'] and coverage['all_legal_L_supervised']
  counters={k:r[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets')};NN=sum(counters.values())-2+r['held_forward_packets'];assert counters['student_train_forward_packets']==5683200+obs['U'] and NN==6959958
  stdout=(root/'logs'/(arm+'-RX1-s'+str(SEED)+'.stdout.log')).read_text(encoding='utf8',errors='replace');assert 'SMOKE PASS own scratch checkpoint; query_read=false' in stdout and not any(v in stdout for v in ('Traceback (most recent call last)','CUDA out of memory','FloatingPointError','SIGKILL'))
  execution[arm]=dict(epochs=20,optimizer_updates=4440,actual_network_forward_packets=NN,L_network_packets=r['L_forward_packets'],extra_L_network_packets=r['extra_L_forward_packets'],held_forward_packets=r['held_forward_packets'],native_elapsed_seconds=r['elapsed_seconds'],teacher_counter_offset_removed=2,instrumented_counters=counters,physical_observations=dict(obs),unique_physical_packets={k:len(v) for k,v in unique.items()},seen_V=r['seen_V'],final_base_L_objective=r['final_base_L_objective'],full_epochs_parsed=20,full_steps_parsed=4440,stdout_lines=len(stdout.splitlines()),formula_error=formula_error,base_formula_error=base_error,teacher_noise_pair_calls=0)
 assert all(contracts[a]==contracts['RAW10'] for a in ARMS)
 argdiff={a:sorted(k for k in set(arguments[a])|set(arguments['RAW10']) if arguments[a].get(k)!=arguments['RAW10'].get(k)) for a in ARMS[1:]};assert all(set(v)<=set(('candidate_id','output_dir')) for v in argdiff.values())
 for view in VIEWS:
  for arm in ARMS[1:]:
   assert pred[arm][view]['physical_ids']==pred['RAW10'][view]['physical_ids'] and pred[arm][view]['labels']==pred['RAW10'][view]['labels']
   if view!='clean':assert pred[arm][view]['physics']==pred['RAW10'][view]['physics']
 # Recount all actual training noise records,streaming both As arms in lockstep.
 traces={a:iter(stream(sources[a]/'noise_pairs.jsonl.gz')) for a in ('INDEP','ANTI')};coverage={a:Counter() for a in traces};noise_stats={a:dict(count=0,added_snr_sum=0.,added_snr_min=math.inf,added_snr_max=-math.inf,AGC_limited=0) for a in traces};records=0
 assert not list(stream(sources['RAW10']/'noise_pairs.jsonl.gz'))
 for step in range(1,4441):
  for slot in range(128):
   pair={a:[next(traces[a]),next(traces[a])] for a in traces}
   for arm,(p,q) in pair.items():
    pair_valid(p,q);assert p['physical_id']==rawL[step-1][slot] and p['step']==step and p['epoch']==(step-1)//222+1
    for item in (p,q):
     coverage[arm][(item['TX'],item['RX'],item['day'],item['scene'],item['state'],item['lock'])]+=1;ns=noise_stats[arm];db=item['signal_to_added_noise_db'];assert db is not None and math.isfinite(db);ns['count']+=1;ns['added_snr_sum']+=db;ns['added_snr_min']=min(ns['added_snr_min'],db);ns['added_snr_max']=max(ns['added_snr_max'],db);ns['AGC_limited']+=int(item['agc_limited'])
   assert pair['INDEP'][0]==pair['ANTI'][0] and all(pair['INDEP'][1][k]==pair['ANTI'][1][k] for k in COMMON)
   assert pair['ANTI'][0]['measured_added_noise_power_pre_agc']==pair['ANTI'][1]['measured_added_noise_power_pre_agc'] and pair['ANTI'][0]['signal_to_added_noise_db']==pair['ANTI'][1]['signal_to_added_noise_db'];records+=4
 for a,t in traces.items():
  assert next(t,None) is None and noise_stats[a]['count']==1136640
  actual=read(sources[a]/'satellite_supervision_coverage.json')['cells'];assert coverage[a]==Counter({(v['TX'],v['RX'],v['day'],v['scene'],v['state'],v['lock']):v['count'] for v in actual}) and {k[0] for k in coverage[a]}==set(range(6)) and {k[1] for k in coverage[a]}==set(FIT) and {k[3] for k in coverage[a]}==set(VIEWS[1:])
  noise_stats[a]['mean_added_noise_SNR_db']=noise_stats[a].pop('added_snr_sum')/noise_stats[a]['count']
 qualification=independent_qualify(results,scores);online=read(root/'result.json')['qualification'];assert qualification['conditions']==online['conditions'] and qualification['qualified']==online['qualified']==completion['qualified']
 paired={};strata=[]
 for ref in ('RAW10','INDEP'):
  paired[ref]={}
  for view in VIEWS:
   b=pred[ref][view];s=pred['ANTI'][view];repair=sum(a!=y and q==y for a,q,y in zip(b['predictions'],s['predictions'],b['labels']));broken=sum(a==y and q!=y for a,q,y in zip(b['predictions'],s['predictions'],b['labels']));gain=100*(scores['ANTI'][view]['accuracy']-scores[ref][view]['accuracy']);assert math.isclose(gain,100*(repair-broken)/1260,abs_tol=1e-10);paired[ref][view]=dict(accuracy_gain_pp=gain,macro_F1_gain_pp=100*(scores['ANTI'][view]['macro_F1']-scores[ref][view]['macro_F1']),repaired=repair,broken=broken)
   for dim,values in (('TX',range(6)),('day',(1,2,3))):
    for value in values:
     take=[i for i,(y,p) in enumerate(zip(b['labels'],b['physical_ids'])) if (y if dim=='TX' else p[2])==value];acc={a:sum(pred[a][view]['predictions'][i]==b['labels'][i] for i in take)/len(take) for a in ARMS};strata.append(dict(reference=ref,view=view,dimension=dim,value=value,count=len(take),reference_accuracy=acc[ref],ANTI_accuracy=acc['ANTI'],gain_pp=100*(acc['ANTI']-acc[ref])))
 summary=dict(status='ANALYZED',run_id=RUN,source_only=True,target_access=False,model_seed=SEED,actual_execution='VERIFIED',scores=scores,qualification=qualification,paired=paired,suburban_clean_gaps_pp={a:{v:100*(scores[a]['clean']['accuracy']-scores[a][v]['accuracy']) for v in VIEWS[1:4]} for a in ARMS},execution=execution,training_noise_audit=dict(full_records_recounted=records,noise0_records_exact_across_As_arms=True,nonnoise_states_exact_across_As_arms=True,stats=noise_stats,coverage_cells={a:len(v) for a,v in coverage.items()},same_IQ_RDpairs=True,no_cross_noise_KL=True),all_L_U_sequences_identical=True,all_training_role_IDs_identical=True,argument_differences=argdiff,new_epochs=60,new_updates=13320,new_NN_packets=sum(v['actual_network_forward_packets'] for v in execution.values())+2,dispatcher_smoke_NN=2,analysis_NN=0,analysis_IQ_reads=0,head_fit_updates=0,Ag_efficacy_proved=False,joint_goal_complete=False,limitations=['One source heldRX1/modelseed only;fixedFINAL20,no target','No Ag/factorial/interaction/crossseed orlow60 efficacy proof','Trainingnoisecorrelation isnot newidentityinformation;frozenvariance doesnotprove trainingbenefit','CE readout recount uses stored harddecisions/F1;full-logit CE andrawIQ not recomputed'])
 dump(root/'analysis_summary.json',summary)
 for name,rows in (('training_curves.csv',curves),('source_strata.csv',strata)):
  with (root/name).open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row}));w.writeheader();w.writerows(rows)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();r=analyze(a.input);print(json.dumps(dict(status=r['status'],qualification=r['qualification'],new_NN_packets=r['new_NN_packets'])))
