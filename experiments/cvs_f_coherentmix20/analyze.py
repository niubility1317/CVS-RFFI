"""Independent full source readback; no torch, IQ, checkpoint or target reads."""
import argparse,csv,gzip,json,math,statistics
from collections import Counter
from pathlib import Path
RUN='20261016-phase1-FcrossRX-coherentmix-source-RX1-e20-s2026092702-m3-r02'
ARMS=('D_raw','SAME_RX','CROSS_RX');NEW=ARMS[1:];FIT=(3,4,6,8);SEED=2026092702
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
def pairing_audit(src,arm,L):
 bank=read(src/'coherent_bank.json');index={tuple(e['physical_id']):e for e in bank['entries']};assert len(index)==5040
 assert {p[1] for p in index}==set(FIT) and all(p[0]==1 for p in index)
 assert set(Counter((e['TX'],p[1],p[2]) for p,e in index.items()).values())=={70}
 coverage=Counter();unique=set();relative=[];coherence=[];scalars=[];counts=Counter();ratio_error=0.;hist=Counter()
 for i,p in enumerate(stream(src/'coherent_pairs.jsonl.gz')):
  finite(p);step=i//128+1;slot=i%128;primary=tuple(p['primary']);partner=tuple(p['partner']);assert p['step']==step and p['epoch']==(step-1)//222+1 and p['mode']==arm
  assert primary==tuple(L[step-1]['ids'][slot]) and primary in index and partner in index and partner!=primary
  assert p['TX']==index[primary]['TX']==index[partner]['TX'] and primary[2]==partner[2] and ((primary[1]==partner[1])==(arm=='SAME_RX'))
  delta=p['relative_L2'];assert 0<=delta<=.10001 and abs(p['norm_ratio']-1)<=1e-5 and 0<=p['reference_coherence']<=1
  relative.append(delta);coherence.append(p['reference_coherence']);scalars.append(math.hypot(p['scalar_real'],p['scalar_imag']));ratio_error=max(ratio_error,abs(p['norm_ratio']-1))
  counts['packets']+=1;counts['capped']+=int(p['arc_capped']);counts['fallback']+=int(p['fallback']);counts['nontrivial_005']+=int(delta>=.005)
  hist[min(19,int(delta/.005))]+=1;unique.add(partner);coverage[(p['TX'],primary[1],partner[1],primary[2])]+=1
 assert counts['packets']==568320 and unique==set(index)
 expected={(t,r,s,day) for t in range(6) for r in FIT for s in FIT for day in (1,2,3) if (r==s)==(arm=='SAME_RX')};assert set(coverage)==expected
 result=read(src/'satmix_result.json')['mix'];actual={(c['TX'],c['primary_RX'],c['partner_RX'],c['day']):c['count'] for c in result['strata']};assert actual==dict(coverage)
 assert result['calls']==4440 and result['packets']==568320 and result['unique_partner_packets']==5040 and not any(result[k] for k in ('U_used','V_used','held_RX_used','target_access'))
 assert math.isclose(max(relative),result['max_relative_L2_change'],abs_tol=1e-12) and math.isclose(counts['nontrivial_005']/568320,result['nontrivial_fraction_005'],abs_tol=1e-12)
 return dict(counts=dict(counts),unique_partner_packets=5040,relative_L2=describe(relative),reference_coherence=describe(coherence),scalar_magnitude=describe(scalars),max_norm_ratio_error=ratio_error,relative_L2_histogram=[dict(lower=i*.005,upper=(i+1)*.005,count=hist[i]) for i in range(20)],coverage=[dict(TX=k[0],primary_RX=k[1],partner_RX=k[2],day=k[3],count=v) for k,v in sorted(coverage.items())],bank_IQ_reads=5040,bank_read_seconds=bank['bank_read_seconds'],no_U_V_held_or_teacher=True)
def analyze(root,reference):
 global RUN,SEED
 root=Path(root);reference=Path(reference);request=read(root/'request.json');cfg=request['rows'];assert request['source_only'] and len(cfg)==2 and {c['arm'] for c in cfg}==set(NEW)
 active={(c['run_id'],c['model_seed']) for c in cfg};assert len(active)==1 and next(iter(active)) in {('20261016-phase1-FcrossRX-coherentmix-source-RX1-e20-s2026092702-m3-r02',2026092702),('20261017-phase1-FcrossRX-coherentmix-confirm-source-RX1-e20-s2026092701-m3-r01',2026092701)}
 RUN,SEED=next(iter(active));assert all(not c['target_access'] and c['epochs']==20 and not c['baseline_reuse'] for c in cfg)
 sources={a:(reference if a=='D_raw' else root)/(a+'-RX1-s'+str(SEED)+'/source') for a in ARMS};scores={};pred={};cost={};sequences={};contracts={};arguments={};curves=[];mix_audits={};result={}
 for arm,src in sources.items():
  r=read(src/'satmix_result.json');result[arm]=r;init=read(src/'initialization.json');done=read(src/'completion.json');frozen=read(src/'source_frozen.json');contract=read(src/'source_contract.json');audit=read(src/'additive_audit.json')
  assert init['scratch_only'] and not init['checkpoint_sources'] and not init['ancestors'] and not init['target_contact']
  assert done['epoch']==frozen['epoch']==20 and done['optimizer_steps']==4440 and frozen['selection']=='final_only' and not done['target_access'] and done['head_fit_updates']==0
  assert contract['source_receivers']==list(FIT) and {k:len(v) for k,v in contract['role_ids'].items()}=={'L_s':5040,'U_s':45360,'V':21600};contracts[arm]=contract['role_ids']
  assert r['actual_steps']==r['additive_steps']==4440 and r['L_forward_packets']==4546560 and r['extra_L_forward_packets']==1136640 and r['supcon_applied']==3330 and r['supcon_layer_forward_terms']==13320
  assert r['Ag']==(arm!='D_raw') and r['As'] is False and not any(r[k] for k in ('As_training_observations','U_truth_read','target_access','head_fit_updates','teacher_additive_calls','checkpoint_sources'))
  assert r['total_pair_weight']==1.25 and audit['calls']==4440 and audit['As_generated_packets']==0 and audit['no_As_U_or_teacher']
  scores[arm]={};pred[arm]={}
  for view in VIEWS:
   p=read(src/'source_readouts'/(view+'.json'));s=score(p['labels'],p['predictions']);assert len(p['physical_ids'])==len({tuple(v) for v in p['physical_ids']})==1260 and {v[1] for v in p['physical_ids']}=={1}
   assert math.isclose(s['accuracy'],r['views'][view]['accuracy'],abs_tol=1e-12) and math.isclose(s['macro_F1'],r['views'][view]['macro_F1'],abs_tol=1e-12)
   if arm!='D_raw':
    assert len(p['margin_over30'])==1260 and all(math.isfinite(x) for x in p['margin_over30'])
    assert all((z<=1e-12 or a==b) and (z>=-1e-12 or a!=b) for a,b,z in zip(p['labels'],p['predictions'],p['margin_over30']))
    if view!='clean':
     assert len(p['physics'])==1260
     for m in p['physics']:
      assert m['processing_route']=='residual' and m['mode']=='post_sync' and m['RX']==1 and not m['channel_equalization_applied'] and not m['true_total_snr_known'] and m['fs_hz']==25000000 and m['fc_hz']==2462000000
      assert m['quality_source']=='white_input_expected_power_proxy' and m['boundary']=='synthetic_reflect' and m['noise_boundary']=='synthetic_reflect' and m['numerical_filter_latency_samples']==24 and m['added_noise_variance_reference']==.001
   scores[arm][view]=s;pred[arm][view]=p
  native=list(stream(src/'metrics_epoch.jsonl'));compact=list(stream(src/'epoch_metrics.jsonl'))
  with (src/'metrics_epoch.csv').open(encoding='utf8',newline='') as f:table=list(csv.DictReader(f))
  assert [v['epoch'] for v in native]==list(range(1,21)) and len(compact)==len(table)==20
  for a,b,c in zip(native,compact,table):
   for k in ('train_loss_tx_labeled','train_loss_unlabeled','val_tx_acc','lr'):assert math.isclose(float(a[k]),float(b[k]),abs_tol=1e-10) and math.isclose(float(a[k]),float(c[k]),abs_tol=1e-10)
  steps=list(stream(src/'step_metrics.jsonl.gz'));views=list(stream(src/'satmix_steps.jsonl.gz'));units=list(stream(src/'unit_steps.jsonl.gz'));assert all([v['step'] for v in x]==list(range(1,4441)) for x in (steps,views,units))
  terms=truth=skips=bn=0;formula_error=base_error=0.;agg={}
  for n,v,u in zip(steps,views,units):
   finite(n);finite(v);finite(u);epoch=(v['step']-1)//222+1;sw=.1*min(max((epoch-5)/10,0.),1.)
   assert v['epoch']==epoch and v['same_IQ_pairs'] and v['all_L_supervised'] and v['extra_view_once'] and v['extra_same_IQ_pair'] and v['As_packets']==v['extra_As_packets']==0 and v['extra_pair_weight']==.25 and v['base_raw_total_weight']==.75 and v['base_ordinary_weight']==.25 and v['total_pair_weight']==1.25
   assert v['extra_input']==('raw' if arm=='D_raw' else 'coherent_'+arm) and math.isclose(v['supcon_weight'],sw,abs_tol=1e-12) and len(v['supcon_details'])==(4 if epoch>5 else 0) and v['sat_raw_MSE']==0
   if arm=='D_raw':assert v['extra_raw_MSE']==0
   else:assert v['coherent_packets']==128
   assert sum(v['scene_counts'].values())==sum(v['state_counts'].values())==sum(v['lock_counts'].values())==0
   terms+=len(v['supcon_details']);bn+=v['extra_BN_batch_updates'];base=.5*(v['raw_CE']+v['raw_KL'])+.25*(v['ordinary_CE']+v['ordinary_KL'])+.25*(v['s_slot_CE']+v['s_slot_KL'])+sw*v['supcon'];extra=v['extra_pair_CE']+v['extra_pair_KL'];total=base+.25*extra
   base_error=max(base_error,abs(base-v['base_total_L']));formula_error=max(formula_error,abs(total-v['total_L']),abs(extra-v['extra_pair_loss']),abs(.25*extra-v['extra_pair_weighted']))
   nm=n['metrics'];truth+=int(nm.get('train/pseudo_truth_available',0) or 0);skips+=int(nm.get('train/skipped_nonfinite_loss',0) or nm.get('train/skipped_nonfinite_grad',0) or 0);assert u['physical_clip_calls']==v['step']
   for k in ('total_L','base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_weight','extra_As_packets'):assert math.isclose(float(nm['satmix/'+k]),v[k],abs_tol=1e-6)
   z=agg.setdefault(epoch,{})
   for k in ('total_L','base_total_L','raw_CE','ordinary_CE','s_slot_CE','supcon','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_raw_MSE'):z[k]=z.get(k,0.)+v[k]/222
  assert terms==13320 and truth==skips==0 and formula_error<1e-5 and base_error<1e-5 and bn==audit['extra_BN_updates']
  assert math.isclose(agg[20]['base_total_L'],r['final_base_L_objective'],abs_tol=1e-8)
  for v in compact:curves.append(dict(arm=arm,**v,**agg[v['epoch']]))
  seq=list(stream(src/'training_sequence.jsonl.gz'));sequences[arm]=seq;assert all(p[1] in FIT for v in seq for p in v['ids'])
  observations={role:sum(len(v['ids']) for v in seq if v['role']==role) for role in ('L','U')};unique={role:len({tuple(p) for v in seq if v['role']==role for p in v['ids']}) for role in ('L','U')};assert observations=={'L':568320,'U':396368} and unique=={'L':5040,'U':45360}
  arguments[arm]=read(src/'actual_native_args.json');a=arguments[arm];assert a['seed']==SEED and a['use_aug'] and a['source_fold_held_rx']==1 and a['label_epochs']==13 and a['pseudo_epochs']==7 and a['aug_enable_pa_normal'] and a['aug_p_dac']==0 and not a['aug_enable_class_signature']
  cov=read(src/'satellite_supervision_coverage.json');assert cov['total']==0 and not cov['cells'] and not cov['confidence_gate']
  counters={k:r[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets')};NN=sum(counters.values())-2+r['held_forward_packets']
  stdout=((reference if arm=='D_raw' else root)/'logs'/(arm+'-RX1-s'+str(SEED)+'.stdout.log')).read_text(encoding='utf8',errors='replace');assert 'SMOKE PASS own scratch checkpoint; query_read=false' in stdout and not any(x in stdout for x in ('Traceback (most recent call last)','CUDA out of memory','FloatingPointError','SIGKILL'))
  cost[arm]=dict(epochs=20,optimizer_updates=4440,actual_network_forward_packets=NN,L_network_packets=r['L_forward_packets'],extra_L_network_packets=r['extra_L_forward_packets'],held_forward_packets=r['held_forward_packets'],native_elapsed_seconds=r['elapsed_seconds'],teacher_counter_offset_removed=2,instrumented_counters=counters,physical_observations=observations,unique_physical_packets=unique,seen_V=r['seen_V'],final_base_L_objective=r['final_base_L_objective'],full_epochs_parsed=20,full_steps_parsed=4440,stdout_lines=len(stdout.splitlines()),formula_error=formula_error,base_formula_error=base_error,extra_BN_updates=bn,teacher_mix_calls=r.get('teacher_mix_calls',0))
  if arm in NEW:mix_audits[arm]=pairing_audit(src,arm,[v for v in seq if v['role']=='L'])
 assert all(sequences[a]==sequences['D_raw'] and contracts[a]==contracts['D_raw'] for a in NEW)
 args_diff={a:sorted(k for k in set(arguments[a])|set(arguments['D_raw']) if arguments[a].get(k)!=arguments['D_raw'].get(k)) for a in NEW}
 assert all(set(v)<=set(('candidate_id','output_dir','run_id')) for v in args_diff.values())
 for view in VIEWS:
  for a in NEW:assert pred[a][view]['labels']==pred['D_raw'][view]['labels'] and pred[a][view]['physical_ids']==pred['D_raw'][view]['physical_ids']
  if view!='clean':assert pred['SAME_RX'][view]['physics']==pred['CROSS_RX'][view]['physics']
 comparisons={};paired={};conditions={};strata=[];eps=1e-10
 for ref in ('D_raw','SAME_RX'):
  paired[ref]={};deltas={}
  for view in VIEWS:
   b=pred[ref][view];s=pred['CROSS_RX'][view];repair=sum(p!=y and q==y for p,q,y in zip(b['predictions'],s['predictions'],b['labels']));broken=sum(p==y and q!=y for p,q,y in zip(b['predictions'],s['predictions'],b['labels']));delta=100*(scores['CROSS_RX'][view]['accuracy']-scores[ref][view]['accuracy']);assert math.isclose(delta,100*(repair-broken)/1260,abs_tol=1e-10);deltas[view]=delta
   paired[ref][view]=dict(accuracy_gain_pp=delta,macro_F1_gain_pp=100*(scores['CROSS_RX'][view]['macro_F1']-scores[ref][view]['macro_F1']),repaired=repair,broken=broken)
   for dimension,values in (('TX',range(6)),('day',(1,2,3))):
    for value in values:
     take=[i for i,(y,p) in enumerate(zip(b['labels'],b['physical_ids'])) if (y if dimension=='TX' else p[2])==value];acc={a:sum(pred[a][view]['predictions'][i]==b['labels'][i] for i in take)/len(take) for a in ARMS};strata.append(dict(reference=ref,view=view,dimension=dimension,value=value,count=len(take),reference_accuracy=acc[ref],CROSS_RX_accuracy=acc['CROSS_RX'],gain_pp=100*(acc['CROSS_RX']-acc[ref])))
  f1=paired[ref]['clean']['macro_F1_gain_pp'];dv=100*(result['CROSS_RX']['seen_V']['source_val_accuracy']-result[ref]['seen_V']['source_val_accuracy']);checks=dict(clean=deltas['clean']>=(1. if ref=='D_raw' else .5)-eps,clean_macroF1=f1>=-eps,seen_V=dv>=-.5-eps);conditions.update({('reused_D_raw' if ref=='D_raw' else ref)+'_'+k:v for k,v in checks.items()});comparisons[ref]=dict(all_view_gain_pp=deltas,clean_macroF1_gain_pp=f1,seen_V_gain_pp=dv,urban_mean_gain_pp=sum(deltas[v] for v in VIEWS[4:])/3,standalone_practical_protection=all(deltas[v]>=-.5-eps for v in VIEWS[1:]),conditions=checks)
 conditions['healthy']=all(result[a]['actual_steps']==4440 and result[a]['L_forward_packets']==4546560 and result[a]['supcon_applied']==3330 and result[a]['final_base_L_objective']<=.5 and result[a]['mix']['calls']==4440 and result[a]['mix']['packets']==568320 and result[a]['mix']['unique_partner_packets']==5040 and result[a]['mix']['max_relative_L2_change']<=.10001 and not any(result[a][k] for k in ('As_training_observations','teacher_mix_calls','head_fit_updates','target_access','U_truth_read','checkpoint_sources')) and result[a]['held_RX']==1 and result[a]['fit_RX']==list(FIT) for a in NEW)
 completion=read(root/'completion.json');online=read(root/'result.json')['qualification'];assert completion['status']=='ANALYZED' and completion['rows']==3 and completion['new_rows']==2 and completion['reused_rows']==1 and completion['new_epochs']==40 and completion['new_optimizer_steps']==8880;assert conditions==online['conditions'] and all(conditions.values())==online['qualified']==completion['qualified']
 for ref in comparisons:
  key='reused_D_raw' if ref=='D_raw' else ref
  for v in VIEWS:assert math.isclose(comparisons[ref]['all_view_gain_pp'][v],online['comparisons'][key]['all_view_gain_pp'][v],abs_tol=1e-12)
 gaps={a:{v:100*(scores[a]['clean']['accuracy']-scores[a][v]['accuracy']) for v in VIEWS[1:4]} for a in ARMS}
 summary=dict(status='ANALYZED',run_id=RUN,source_only=True,target_access=False,model_seed=SEED,actual_execution='VERIFIED',scores=scores,qualification=dict(qualified=all(conditions.values()),conditions=conditions,comparisons=comparisons),paired=paired,suburban_clean_gaps_pp=gaps,execution=cost,mix_audits=mix_audits,all_L_U_sequences_identical=True,all_training_role_IDs_identical=True,argument_differences=args_diff,new_epochs=40,new_updates=8880,new_NN_packets=sum(cost[a]['actual_network_forward_packets'] for a in NEW)+2,reused_reference_epochs=20,reused_reference_updates=4440,reused_reference_NN=cost['D_raw']['actual_network_forward_packets'],dispatcher_smoke_NN=2,failed_r01_NN=0,failed_r01_updates=0,analysis_NN=0,analysis_IQ_reads=0,head_fit_updates=0,Ag_efficacy_proved=False,joint_goal_complete=False,limitations=['One source RX / seed only','Reference has no stored per-packet margins/physics;new-arm fullphysics identical and code/config/namedstream/IDs matchreference','No target data,Ag+As factorial or crossseed Ag evidence','Readout mean CE needs full logits for independent recount;hard decisions only independently score accuracy/F1'])
 dump(root/'analysis_summary.json',summary)
 for name,rows in (('training_curves.csv',curves),('source_strata.csv',strata)):
  with (root/name).open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row}));w.writeheader();w.writerows(rows)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--reference',required=True);a=p.parse_args();r=analyze(a.input,a.reference);print(json.dumps(dict(status=r['status'],qualification=r['qualification'],new_NN_packets=r['new_NN_packets'])))
