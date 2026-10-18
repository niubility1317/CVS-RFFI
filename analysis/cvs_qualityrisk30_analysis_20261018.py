"""Zero-model source/log/formal recount for the fixed finite-bank quality-stratified risk E30 pair.

Reads saved predictions, metadata and complete structured logs only. No model,
checkpoint or IQ loading; no training mutation or automatic technical retry.
"""
import argparse,json,os,re,sys,time,traceback
from collections import Counter,defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_practical_budget50_analysis_20261018 as old
RUN='20261018-phase1-FcrossRX-qualityRisk-e30-s2026092701-m2-r01'
TRAINING_RELEASE='cvs_FcrossRX_qualityRisk_E30_20261018_r02'
REFERENCE_RUN=old.REFERENCE_RUN
PRIOR_E50_RUN=old.RUN
CONDITIONS=('NOAS30','AS30','BANK_IID30','BANK_STRAT30')
PAIRS=(('BANK_STRAT30','BANK_IID30'),('BANK_IID30','AS30'),('BANK_STRAT30','AS30'),('BANK_IID30','NOAS30'),('BANK_STRAT30','NOAS30'),('AS30','NOAS30'))
VIEWS=old.VIEWS;SOURCE_SCENES=old.SOURCE_SCENES;SEED=2026092701;COUNT=168000
STEPS=2220;EPOCHS=10;STEP_OFFSET=4440
read=old.read;write=old.write;metric=old.metric;paired=old.paired;write_csv=old.write_csv
AwaitingArtifacts=old.AwaitingArtifacts;logs=old.logs;compare_values=old.compare_values

def rows(project):
 p=Path(project)/'runs'
 return [('NOAS30',p/REFERENCE_RUN/'B00-s2026092701'),('AS30',p/REFERENCE_RUN/'B01-s2026092701'),('BANK_IID30',p/RUN/'I01-s2026092701'),('BANK_STRAT30',p/RUN/'S01-s2026092701')]

def ensure_metadata(project,stage):
 result={}
 for arm,root in rows(project):
  names=['source/completion.json','source/source_frozen.json','source_qualification/complete.json']
  if stage=='final':names+=['prediction/complete.json','scoring/complete.json','complete.json']
  found={}
  for name in names:
   try:found[name]=read(root/name)
   except (FileNotFoundError,json.JSONDecodeError) as exc:raise AwaitingArtifacts(arm+'/'+name) from exc
  c=found['source/completion.json'];f=found['source/source_frozen.json'];q=found['source_qualification/complete.json']
  if (c['epoch'],c['optimizer_steps'])!=(10,2220) or c['target_access'] or c.get('head_fit_updates',0) or c['config']['model_seed']!=SEED or c['config']['row_id']!=root.name or c['config']['run_id']!=root.parent.name:raise ValueError('Actual fixed row/E30/defaulthead differs')
  if c.get('cumulative_epochs',30)!=30 or c.get('cumulative_optimizer_steps',6660)!=6660:raise ValueError('Cumulative E30 differs')
  if f['selection']!='final_only' or f['target_access'] or f['config']!=c['config'] or q['target_access'] or q['NN']!=12600 or q.get('training_updates',0) or q.get('head_fit_updates',0):raise ValueError('Frozen final/source readout differs')
  if stage=='final':
   p=found['prediction/complete.json'];s=found['scoring/complete.json']
   if p['status']!='PREDICTIONS_COMPLETE' or s['status']!='SCORED_COMPLETE' or found['complete.json']['status']!='TRAINED_AND_SCORED' or p['views']!=list(VIEWS) or s['views']!=list(VIEWS) or p['count']!=COUNT or p['truth_read'] or p['query_fit'] or not p['all_parameters_frozen'] or p['head_fit_updates'] or p['actual_NN']!=COUNT*7+2:raise ValueError('Truth closed until all4 logical rows have all7 frozen predictions and independent scores')
  result[arm]=found
 return result

def source_conditions(summary,quality):
 s=summary['BANK_STRAT30'];oldas=summary['AS30'];clean=float(100*(s['full_V_clean']-oldas['full_V_clean']))
 differences={group:float(np.mean([100*(quality['BANK_STRAT30'][v][group]['accuracy']-quality['BANK_IID30'][v][group]['accuracy']) for v in SOURCE_SCENES[1:]])) for group in ('hard_lt8','easy_ge8')}
 for label,value in [('fullV clean S-originalAs',clean),*differences.items()]:logs.finite(value,label)
 checks=dict(hard_lt8_equal_scene_accuracy_gain_at_least_0p5pp=bool(differences['hard_lt8']>=.5-1e-12),easy_ge8_equal_scene_accuracy_drop_at_most_0p2pp=bool(differences['easy_ge8']>=-.2-1e-12),fullV_clean_vs_originalAs_within_0p1pp=bool(clean>=-.1-1e-12));passed=bool(all(checks.values()))
 return dict(checks=checks,source_criteria_pass=passed,M3_source_eligible=passed,E50_source_eligible=False,fixed_stage1_closed=bool(not passed),fullV_clean_S_minus_originalAs_pp=clean,hard_lt8_S_minus_I_equal_scene_accuracy_pp=differences['hard_lt8'],easy_ge8_S_minus_I_equal_scene_accuracy_pp=differences['easy_ge8'],only_three_fixed_source_criteria=True,scientific_negative_skips_formal=False,target_scores_read=False,automatic_extension=False,channel_draws_are_not_model_seeds=True)


def source_analysis(project):
 started=time.perf_counter();meta=ensure_metadata(project,'source');data={};summary={};metrics=[];quality={}
 panel=Path(project)/old.stable.PANEL;panel_roles=read(panel/'data_roles.json')
 with np.load(panel/'observations/physics.npz',allow_pickle=False) as z:physics={k:z[k].copy() for k in ('state','lock','signal_to_added_noise_db')}
 for key_,array in physics.items():
  if array.shape!=(2,3,11,1800):raise ValueError('Existing source physics must retain two draws/three scenes/full index0/1800 IDs')
 if not np.isfinite(physics['signal_to_added_noise_db'][:,:,0]).all():raise ValueError('Source actual added-noise ratio finite full observations required')
 for arm,root in rows(project):
  with np.load(root/'source_qualification/predictions.npz',allow_pickle=False) as z:x={k:z[k].copy() for k in z.files}
  if not np.array_equal(x['ids'],np.asarray(panel_roles['physical_ids'])):raise ValueError('Existing source physics physical ordering differs')
  if x['clean'].shape!=(1800,6) or x['satellite'].shape!=(2,3,1800,6) or not np.isfinite(x['clean']).all() or not np.isfinite(x['satellite']).all():raise ValueError('Finite fixed source logits absent')
  if data and any(not np.array_equal(x[k],data['NOAS30']['x'][k]) for k in ('ids','TX','RX','day')):raise ValueError('Same source physical panel required')
  if len({tuple(np.atleast_1d(v).tolist()) for v in x['ids']})!=1800:raise ValueError('Source1800 physical IDs must be unique')
  q=meta[arm]['source_qualification/complete.json'];acc=(x['satellite'].argmax(-1)==x['TX']).mean(-1)
  if not np.allclose(acc,q['source_accuracies'],atol=1e-12,rtol=0) or abs(float((x['clean'].argmax(1)==x['TX']).mean())-q['clean_accuracy'])>1e-12:raise ValueError('Independent source readout differs')
  native=meta[arm]['source/completion.json']['final_source_metrics']
  if native['source_val_count']!=27000:raise ValueError('Native fullV count differs')
  pred=dict(clean=x['clean'].argmax(-1));quality[arm]={}
  for i,v in enumerate(SOURCE_SCENES):pred[v]=x['satellite'][:,i].argmax(-1).reshape(-1)
  data[arm]=dict(x=x,pred=pred);panel_summary={}
  observations=[('clean',-1,x['clean'],x['TX'],x['RX'],x['day'],None,None,None)]
  for si,v in enumerate(SOURCE_SCENES):
   for draw in (-1,0,1):
    draws=(0,1) if draw==-1 else (draw,)
    observations.append((v,draw,np.concatenate([x['satellite'][j,si] for j in draws]),np.tile(x['TX'],len(draws)),np.tile(x['RX'],len(draws)),np.tile(x['day'],len(draws)),np.concatenate([physics['state'][j,si,0] for j in draws]),np.concatenate([physics['lock'][j,si,0] for j in draws]),np.concatenate([physics['signal_to_added_noise_db'][j,si,0] for j in draws])))
  for view,draw,z,y,rx,day,state,lock,ratio in observations:
   p=z.argmax(1);other=z.copy();other[np.arange(len(y)),y]=-np.inf;margin=(z[np.arange(len(y)),y]-other.max(1))/30
   masks=[('overall','ALL',np.ones(len(y),bool))]+[('RX',str(v),rx==v) for v in sorted(set(rx))]+[('day',str(v),day==v) for v in sorted(set(day))]+[('TX',str(v),y==v) for v in range(6)]
   if ratio is not None:
    masks += [('added_noise_ratio',label,(ratio>=lo)&(ratio<hi)) for label,lo,hi in [('<3',-np.inf,3),('3-8',3,8),('8-15',8,15),('>=15',15,np.inf)]]
    masks += [('quality_group','hard_lt8',ratio<8),('quality_group','easy_ge8',ratio>=8)]
    masks += [('state_lock',s+'/'+l,(state==i)&(lock==j)) for i,s in enumerate(('LOS','PARTIAL','BLOCKED')) for j,l in enumerate(('locked','degraded','unlocked'))]
   for dim,stratum,mask in masks:
    if not mask.any():continue
    m=metric(y[mask],p[mask]);metrics.append(dict(arm=arm,scene=view,draw=draw,dimension=dim,stratum=stratum,mean_trueclass_margin_over30=float(margin[mask].mean()),**m))
    if dim=='overall' and draw==-1:panel_summary[view]=dict(accuracy=m['accuracy'],macro_F1=m['macro_f1'],worst_RX=min(float((p[rx==r]==y[rx==r]).mean()) for r in np.unique(rx)),mean_trueclass_margin_over30=float(margin.mean()))
    if dim=='quality_group' and draw==-1:quality[arm].setdefault(view,{})[stratum]=dict(count=m['count'],accuracy=m['accuracy'],macro_F1=m['macro_f1'],mean_trueclass_margin_over30=float(margin[mask].mean()))
   if ratio is not None and draw==-1 and set(quality[arm].get(view,{}))!={'hard_lt8','easy_ge8'}:raise ValueError('Fixed source hard/easy populations required for the declared risk comparison')
  summary[arm]=dict(full_V_clean=native['source_val_accuracy'],full_V_RX_accuracy=native['source_val_rx_accuracy'],panel=panel_summary,mid_low_mean=float(np.mean([panel_summary[v]['accuracy'] for v in SOURCE_SCENES[1:]])),mid_low_by_draw_accuracy=acc[:,1:].tolist(),source_readout_NN=q['NN'])
 pairs={};quality_pairs={};cohort=data['NOAS30']['pred']['clean']==data['NOAS30']['x']['TX']
 for a,b in PAIRS:
  key_=a+'__'+b;pairs[key_]={};quality_pairs[key_]={}
  for v in ('clean',*SOURCE_SCENES):
   y=data[a]['x']['TX'] if v=='clean' else np.tile(data[a]['x']['TX'],2);co=cohort if v=='clean' else np.tile(cohort,2);pairs[key_][v]=paired(y,data[b]['pred'][v],data[a]['pred'][v],co)
   if v!='clean':
    ratio=physics['signal_to_added_noise_db'][:,SOURCE_SCENES.index(v),0].reshape(-1);quality_pairs[key_][v]={}
    for group,mask in [('hard_lt8',ratio<8),('easy_ge8',ratio>=8)]:quality_pairs[key_][v][group]=paired(y[mask],data[b]['pred'][v][mask],data[a]['pred'][v][mask],co[mask])
 return dict(status='SOURCE_ANALYZED',run_id=RUN,summary=summary,metrics=metrics,quality_groups=quality,quality_paired=quality_pairs,paired=pairs,decision=source_conditions(summary,quality),source_unique_V=1800,channel_draws=2,model_seed_count=1,model_seed=SEED,fullV_metrics_from_native_not_individual_prediction_recount=True,source_high_urban_available=False,source_panel_fc_hz=2450000000,training_formal_fc_hz=2462000000,added_noise_ratio_is_total_SNR=False,hard_easy_accuracy_is_equal_scene_not_pooled_count_weighted=True,margin_uses_existing_logits_only=True,source_physics_full_condition_index=0,formal_physics_tags_available=False,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False,wall_seconds=time.perf_counter()-started)


def actual_cost(value):
 keys=('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN')
 if any(type(value.get(k)) is not int or value[k]<0 for k in keys):raise ValueError('Actual native four counters required')
 counters={k:value[k] for k in keys};total=sum(counters.values())
 if value.get('native_NN',total)!=total:raise ValueError('Saved native sum differs from actual counters')
 return dict(**counters,native_training_and_validation_NN=total)
def exposure_summary(counts,all_ids):
 values=np.asarray([counts[x] for x in all_ids],int)
 return dict(unique_physical_packets=len(counts),all_L_physical_packets=len(all_ids),unexposed_L_packets=int((values==0).sum()),minimum=int(values.min()),maximum=int(values.max()),mean=float(values.mean()),exposure_count_histogram={str(k):int(v) for k,v in sorted(Counter(values.tolist()).items())},per_ID_matched_to_original_not_assumed=True)


BANK_SIZE=252000;BANK_L=6300;BANK_VIEWS=40
BANK_KEYS=('bank_weight_sum','bin0','bin1','bin2','bin3','sampling_cpu_wall_seconds','weighted_loss_cpu_wall_seconds','extra_NN')
TIMERS=('sampling_cpu_wall_seconds','cache_gather_cpu_wall_seconds','cache_transfer_cpu_wall_seconds','weighted_loss_cpu_wall_seconds')

def hist_summary(values):
 values=np.asarray(values);return dict(length=len(values),minimum=float(values.min()),maximum=float(values.max()),mean=float(values.mean()),zero_count=int((values==0).sum()),histogram={str(k):int(v) for k,v in sorted(Counter(values.tolist()).items())})

def audit_bank(project):
 b=Path(project)/'runs'/RUN/'proposal_bank';done=read(b/'complete.json')
 if (done['status'],done['run_id'],done['role'],done['count'],done['source_L'],done['each_L_views'],done['bin_field'],done['bin_edges'])!=('SOURCE_L_BANK_COMPLETE',RUN,'L',BANK_SIZE,BANK_L,BANK_VIEWS,'signal_to_added_noise_db',[3.,8.,15.]) or any(done[k] for k in ('NN','checkpoint_reads','U_truth_used','U_iterations','target_access','quality_rejection','retries','bank_online_risk_equivalence_claim')):raise ValueError('Own complete sourceL-only shared physical bank required')
 with np.load(b/'metadata.npz',allow_pickle=False) as z:meta={k:z[k].copy() for k in z.files}
 keys=('TX','RX','day','domain','bin','scene','state','lock','signal_to_added_noise_db','quality_snr_db','cumulative_epoch','replica','context_step','source_L_index')
 if set(meta)!={*keys,'sample_ids'} or any(meta[k].shape!=(BANK_SIZE,) for k in keys) or meta['sample_ids'].shape!=(BANK_SIZE,5) or not np.isfinite(meta['signal_to_added_noise_db']).all() or not np.isfinite(meta['quality_snr_db']).all():raise ValueError('Full bank metadata/index shape and actualquality fields required')
 if not np.isin(meta['TX'],np.arange(6)).all() or not np.isin(meta['RX'],[1,3,4,6,8]).all() or not np.isin(meta['day'],[1,2,3]).all() or not np.isin(meta['domain'],np.arange(15)).all() or not np.all(meta['sample_ids'][:,0]==1) or not np.array_equal(meta['sample_ids'][:,1],meta['RX']) or not np.array_equal(meta['sample_ids'][:,2],meta['day']) or not np.isin(meta['scene'],SOURCE_SCENES).all() or not np.isin(meta['state'],['LOS','PARTIAL','BLOCKED']).all() or not np.isin(meta['lock'],['locked','degraded','unlocked']).all():raise ValueError('Actual bank legal L/source metadata differs')
 calculated=np.searchsorted([3.,8.,15.],meta['signal_to_added_noise_db'],side='right');counts=np.bincount(calculated,minlength=4);mass=counts/BANK_SIZE
 if not np.array_equal(calculated,meta['bin']) or counts.tolist()!=done['bin_counts'] or np.any(counts<=0) or not np.allclose(mass,done['nominal_bin_mass'],rtol=0,atol=1e-15):raise ValueError('Bank bins and nominal finite-bank mass differ')
 source_index=meta['source_L_index']
 if source_index.dtype.kind not in 'iu' or np.any((source_index<0)|(source_index>=BANK_L)) or np.bincount(source_index,minlength=BANK_L).tolist()!=[BANK_VIEWS]*BANK_L or done['source_L_index_counts']!=[BANK_VIEWS]*BANK_L:raise ValueError('Every physical L must have exactly40 bank observations')
 first=np.unique(source_index,return_index=True)[1];ids=meta['sample_ids'][first];label=meta['TX'][first];domain=meta['domain'][first]
 if len(set(map(tuple,ids.tolist())))!=BANK_L or any(not np.array_equal(meta[k],meta[k][first][source_index]) for k in ('TX','RX','day','domain','sample_ids')):raise ValueError('Bank true label/domain/PID must remain associated with original sourceL index')
 contexts=[(e,r,(e-1)*222+1+r) for e in range(21,31) for r in range(4)];observed=Counter(zip(meta['cumulative_epoch'].tolist(),meta['replica'].tolist(),meta['context_step'].tolist()))
 if observed!=Counter({v:BANK_L for v in contexts}) or done['contexts']!=[dict(cumulative_epoch=e,replica=r,cumulative_step=s) for e,r,s in contexts]:raise ValueError('Fixed40 original channel namespace contexts differ')
 domain_map={}
 for r,d,dom in zip(meta['RX'][first],meta['day'][first],domain):
  pair=(int(r),int(d))
  if pair in domain_map and domain_map[pair]!=int(dom):raise ValueError('Original RX/day domain metadata inconsistent')
  domain_map[pair]=int(dom)
 if len(domain_map)!=15 or len(set(domain_map.values()))!=15:raise ValueError('All15 original RX/day domains required')
 strata=Counter(zip(*(meta[k].tolist() for k in ('TX','RX','day','bin','scene','state','lock'))));saved=Counter({tuple(z[k] for k in ('TX','RX','day','bin','scene','state','lock')):z['count'] for z in done['strata']})
 if strata!=saved or done['propagation_simulations']!=BANK_SIZE or done['shape']!=[BANK_SIZE,2,256] or done['dtype']!='float32' or done['IQ_bytes']!=BANK_SIZE*2*256*4:raise ValueError('Actual bank coverage/simulation/storage declaration differs')
 storage={name:(b/name).stat().st_size for name in ('iq.npy','metadata.npz','physics.jsonl.gz','complete.json')}
 public=dict(status='VERIFIED_COMPLETE_BANK_METADATA',observations=BANK_SIZE,unique_L=BANK_L,each_L_views=BANK_VIEWS,contexts=done['contexts'],bin_edges=done['bin_edges'],bin_labels=['<3','[3,8)','[8,15)','>=15'],bin_counts=counts.tolist(),nominal_bin_mass=mass.tolist(),strata=done['strata'],propagation_simulations=done['propagation_simulations'],IQ_payload_bytes_declared=done['IQ_bytes'],storage_bytes=storage,wall_seconds=done['wall_seconds'],timing=done['timing'],CPUwall={k:done[k] for k in ('raw_dataset_cache_cpu_wall_seconds','simulation_adapter_cpu_wall_seconds','IQ_cache_write_cpu_wall_seconds','metadata_write_cpu_wall_seconds')},NN=0,checkpoint_reads=0,analysis_IQ_reads=0,source_role='L only',bank_cost_counted_once=True,finite_bank_not_online_risk=True,full_physics_metadata_not_rescanned=True)
 return public,dict(meta=meta,domain_map=domain_map,source_ids=ids,nominal_mass=mass,bin_counts=counts)


def check_native_L(r,s,m):
 errors=[]
 for prefix,record in [('common',r),('extra',r),('shape',s)]:errors.append(logs._near(record[prefix+'_RD'],record[prefix+'_CE']+record[prefix+'_KL'],'Original RDrop CE/KL decomposition differs'))
 base=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];total=base+.25*s['shape_RD']
 errors += [logs._near(r['base_L_before_shape'],base,'Original .5/.5 base differs'),logs._near(r['total_L'],total,'Original full1.25L differs')]
 for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight'):errors.append(logs._near(m['concat/'+k],r[k],'Actual full6L native scalar differs'))
 return max(errors)


def check_bank_step(r,s,c,m,bank,mode):
 meta=bank['meta'];positions=[j for j in range(128) if (j+c['step'])%2!=0];ordinary=[j for j in range(128) if j not in positions]
 if c['mode']!=mode or c['selected_L_positions']!=positions or c['ordinary_L_positions']!=ordinary or any(len(c[k])!=128 for k in ('current_L_TX','current_L_RX','current_L_day','current_L_sample_ids','mixed_labels','mixed_domains','per_packet_CE_a','per_packet_CE_b','per_packet_KL')):raise ValueError('Original128/alternating64 and complete loss vectors required')
 indices=np.asarray(c['bank_indices']);bb=meta['bin'][indices] if indices.shape==(64,) and indices.dtype.kind in 'iu' and np.all((indices>=0)&(indices<BANK_SIZE)) else None
 if bb is None:raise ValueError('64 legal finite bank indices;replacement is allowed')
 mappings={'bank_TX':'TX','bank_RX':'RX','bank_day':'day','bank_domain':'domain','bank_sample_ids':'sample_ids','bank_source_L_index':'source_L_index','bin':'bin','scene':'scene','state':'state','lock':'lock','signal_to_added_noise_db':'signal_to_added_noise_db','quality_snr_db':'quality_snr_db','bank_context_epoch':'cumulative_epoch','bank_replica':'replica','bank_context_step':'context_step'}
 if any(not np.array_equal(np.asarray(c[k]),meta[v][indices]) for k,v in mappings.items()):raise ValueError('CachedAs actual true label/domain/PID/physical-quality record differs from bank index')
 expected_y=np.asarray(c['current_L_TX']).copy();expected_y[positions]=meta['TX'][indices];expected_domain=np.asarray([bank['domain_map'][(int(rx),int(day))] for rx,day in zip(c['current_L_RX'],c['current_L_day'])]);expected_domain[positions]=meta['domain'][indices]
 if not np.array_equal(expected_y,c['mixed_labels']) or not np.array_equal(expected_domain,c['mixed_domains']) or any((z[1],z[2])!=(rx,day) for z,rx,day in zip(c['current_L_sample_ids'],c['current_L_RX'],c['current_L_day'])):raise ValueError('Mixed As true labels/domains must replace only actual64 original positions')
 counts=np.bincount(bb,minlength=4);weights=np.ones(64) if mode=='IID' else 4*bank['nominal_mass'][bb]
 if c['bin_counts']!=counts.tolist() or (mode=='STRATIFIED' and counts.tolist()!=[16]*4) or c['bank_bin_counts']!=bank['bin_counts'].tolist() or not np.allclose(c['nominal_bin_mass'],bank['nominal_mass'],rtol=0,atol=1e-15) or not np.allclose(c['weights'],weights,rtol=0,atol=1e-14) or not c['with_replacement'] or c['global_RNG_used'] or c['extra_NN']:raise ValueError('IID versus16/bin with4p finite-bank importance weights differs')
 e=[logs._near(c['bank_weight_sum'],64,'Nominal bank weight mass must stay64',1e-10),logs._near(c['Torch_bank_weight_sum'],64,'Actual Torch bank mass differs')]
 if c['ordinary_weight_sum']!=64 or (c['As_packets'],c['bank_cache_reads'],c['propagation_simulations'],c['L_forwards'],c['L_NN'],c['total_L_scale'],c['CE_logit_packets'],c['new_weight_multiplication_packets'])!=(64,64,0,6,768,1.25,256,128) or c['CE_operator_widths']!=[128,128] or c['baseline_CE_operator_widths']!=[128,128] or any(c[k] for k in ('extra_CE_operator_calls','ordinary_redraws','bank_redraws_in_second_forward')) or not c['same_IQ_each_pair']:raise ValueError('Original sixforward/sameIQ/CEscope and cachedAs budget differs')
 applied=np.asarray(c['applied_bank_weights'],float)
 if applied.shape!=(64,) or not np.array_equal(applied,weights.astype(np.float32).astype(float)):raise ValueError('Actual float32 bank weights differ from nominal importance weights')
 ca=np.asarray(c['per_packet_CE_a'],float);cb=np.asarray(c['per_packet_CE_b'],float);kl=np.asarray(c['per_packet_KL'],float);ce=.5*(ca+cb);w=np.ones(128);w[positions]=applied
 expected=dict(ordinary_CE=ce[ordinary].mean(),ordinary_KL=kl[ordinary].mean(),bank_unweighted_CE=ce[positions].mean(),bank_unweighted_KL=kl[positions].mean(),bank_weighted_CE=(ce[positions]*applied).mean(),bank_weighted_KL=(kl[positions]*applied).mean(),mixed_weighted_CE=(ce*w).mean(),mixed_weighted_KL=(kl*w).mean(),mixed_weighted_RD=((ce+kl)*w).mean(),total_L=r['total_L'])
 for k,value in expected.items():e.append(logs._near(c[k],value,'Actual weighted CE/KL fixed128 reduction differs: '+k))
 for k,other in [('extra_CE','mixed_weighted_CE'),('extra_KL','mixed_weighted_KL'),('extra_RD','mixed_weighted_RD')]:e.append(logs._near(r[k],c[other],'Actual reference weighted mixed loss differs'))
 for k in BANK_KEYS:e.append(logs._near(m['bank/'+k],c[k],'Actual native bank telemetry differs'))
 if c['scene_counts']!=r['scene_counts'] or dict(Counter(c['scene']))!=r['scene_counts'] or dict(Counter(c['state']))!=r['state_counts'] or c['state_counts']!=r['state_counts'] or c['sampling_seed']!=20261016421 or c['sampling_namespace']!='FcrossRX/qualityRisk/bank_sampling/v1' or any(c[k]<0 for k in TIMERS):raise ValueError('Original source scenes/states/private sampling/timers differ')
 if c['mixed_label_replacements']!=sum(c['current_L_TX'][p]!=v for p,v in zip(positions,c['bank_TX'])) or c['currentL_bank_PID_matches']!=sum(tuple(c['current_L_sample_ids'][p])==tuple(v) for p,v in zip(positions,c['bank_sample_ids'])):raise ValueError('Actual bank replacement diagnostic counts differ')
 return max(e),logs.relation(r,True)


def parse_new_logs(root,arm,bank):
 mode={'BANK_IID30':'IID','BANK_STRAT30':'STRATIFIED'}.get(arm)
 if mode is None:raise ValueError('Only two new qualityRisk tails')
 b=root/'source';done=read(b/'completion.json');result=read(b/'result.json');cfg=done['config'];p=cfg['quality_risk']
 if cfg['crossview']['mode']!='ORD128' or p['mode']!=mode or (p['bank_size'],p['source_L'],p['each_L_bank_views'],p['bank_packets_per_step'],p['mixed_denominator'],p['mixed_outer_weight'],p['As_expected_outer_weight'],p['all_loss_scale'],p['CE_label_smoothing'])!=(BANK_SIZE,BANK_L,BANK_VIEWS,64,128,.5,.25,1.25,.01) or p['bin_field']!='signal_to_added_noise_db' or p['bin_edges']!=[3.,8.,15.] or p['per_bin_packets']!=(None if mode=='IID' else 16) or p['bank_weight']!=('1' if mode=='IID' else '4*N_bin/N_bank') or not all(p[k] for k in ('sample_with_replacement','KL_same_weights_as_CE','private_RNG','original_alternating_mask','mixed_labels_replaced','mixed_domain_metadata_replaced','ORD128_currentL_only')) or any(p[k] for k in ('global_RNG_used','online_bank_risk_equivalence_claim','common_U_trajectory_equality_claim','extra_NN','extra_CE_operator_calls','formal_test_changed')):raise ValueError('Fixed finite-bank expected-risk recipe differs')
 if (done['epoch'],done['optimizer_steps'],cfg['updates'],result['optimizer_steps'])!=(10,2220,2220,2220) or (result['parent_epochs'],result['parent_optimizer_steps'],result['cumulative_epochs'],result['cumulative_optimizer_steps'])!=(20,4440,30,6660) or result['parent_reference_row_id']!='B01-s2026092701':raise ValueError('Both originalAs20+10 tails required')
 if any(result[k] for k in ('U_truth_read','target_access','head_fit_updates','extra_U_NN','U_As','Ag_packets')) or (result['L_NN'],result['L_As'],result['total_L_loss_scale'],result['EMA_local_successful_updates'],result['EMA_cumulative_updates'],result['U_steps'],result['U_packets'],result['U_student_NN'])!=(1704960,142080,1.25,2220,6660,2220,567000,567000):raise ValueError('Original sixforward/U/EMA budget differs')
 ema=read(b/'EMA_initialization.json')
 if ema['origin']!='strict own stage20 EMA' or ema['extra_teacher'] or not ema['optimizer_segment_counters_reset'] or not ema['temporal_cache_reset']:raise ValueError('Same strict originalAs20 EMA required')
 cost=actual_cost(result);epochs,telemetry=logs.native_epochs(b,EPOCHS);counts=Counter();sums=defaultdict(Counter);timers=Counter();steps=[];maxerr=0.;ordinary_mse=[];index_count=np.zeros(BANK_SIZE,np.int64);physical_count=np.zeros(BANK_L,np.int64);physical_mass=np.zeros(BANK_L,float);bins_count=np.zeros(4,np.int64);bins_mass=np.zeros(4,float);strata=Counter();weighted_strata=Counter()
 streams=[logs.records(b/name) for name in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz','bank_steps.jsonl.gz')]
 for i,items in enumerate(zip_longest(*streams),1):
  if any(z is None for z in items):raise logs.IncompleteTrainingLogs('New qualityRisk tail stream prefix')
  if i>STEPS:raise ValueError('Extra update outside E30 tail')
  r,s,n,u,c=items;ep=(i-1)//222+1;m=n['metrics'];logs.scan_numbers(items)
  if any(z['step']!=i for z in items) or any(z['cumulative_step']!=4440+i for z in (r,s,u,c)) or n['epoch']!=ep or c['epoch']!=ep or c['cumulative_epoch']!=20+ep or u['physical_clip_calls']!=i:raise ValueError('Contiguous local/cumulative log counters required')
  if (r['L_forwards'],r['As_packets'],r['Ag_packets'],r['extra_control_packets'],r['Ag_mode'])!=(6,64,0,128,'NONE') or r['ordinary_redraws'] or not r['same_IQ_each_pair'] or not r['all_satellite_L_supervised']:raise ValueError('Original sixforward labeled supervision differs')
  if (s['mode'],s['NN'],s['packets'],s['applied'],s['weight'])!=('NONE',256,128,0,.25) or s['new_boundary_samples'] or not s['same_IQ_each_pair'] or s['relative_noise_RMS'] or s['noise_peak_fraction_mean'] or s['SNR_added_db'] is not None:raise ValueError('Original raw shape slot differs')
  logs._near(r['SupCon_weight'],.1,'Original cumulative ORD128 SupCon weight differs',1e-12);e,anchors=check_bank_step(r,s,c,m,bank,mode);maxerr=max(maxerr,e,check_native_L(r,s,m))
  if m['concat/L_forwards']!=6 or m['concat/As_packets']!=64 or any(m.get(k) or 0 for k in ('train/pseudo_truth_available','train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')):raise ValueError('Actual native forwards/Uhidden/numerics differ')
  number=m['train/pseudo_total'];picked=m['train/pseudo_selected'];logs.finite(m['train/loss'],'Actual native total loss')
  if int(number)!=number or number!=(124 if i%222==0 else 256) or not 0<=picked<=number:raise ValueError('Original U consumption differs')
  ix=np.asarray(c['bank_indices'],int);source_ix=np.asarray(c['bank_source_L_index'],int);bb=np.asarray(c['bin'],int);weights=np.asarray(c['weights'],float)
  np.add.at(index_count,ix,1);np.add.at(physical_count,source_ix,1);np.add.at(physical_mass,source_ix,weights);bins_count+=np.bincount(bb,minlength=4);bins_mass+=np.bincount(bb,weights=weights,minlength=4)
  for tx,rx,day,g,scene,state,lock,weight in zip(c['bank_TX'],c['bank_RX'],c['bank_day'],bb,c['scene'],c['state'],c['lock'],weights):key_=(tx,rx,day,int(g),scene,state,lock);strata[key_]+=1;weighted_strata[key_]+=float(weight)
  timers.update({k:c[k] for k in TIMERS});counts.update(steps=1,L_NN=768,As_packets=64,bank_cache_reads=64,U_forward_packets=int(number),U_selected=int(picked),SupCon_valid_anchor_terms=anchors,CE_operator_calls=2,CE_logit_packets=256,new_weight_multiplication_packets=128);steps.append(c);ordinary_mse.append(r['ordinary_raw_MSE'])
  sums[ep].update({k:r[k] for k in ('total_L','common_CE','common_KL','extra_CE','extra_KL','SupCon','ordinary_raw_MSE')});sums[ep].update({k:s[k] for k in ('shape_CE','shape_KL','shape_RD')});sums[ep].update({k:c[k] for k in ('ordinary_CE','ordinary_KL','bank_unweighted_CE','bank_unweighted_KL','bank_weighted_CE','bank_weighted_KL','mixed_weighted_CE','mixed_weighted_KL','bank_weight_sum')});sums[ep]['native_loss']+=m['train/loss']
 if counts['steps']!=STEPS:raise logs.IncompleteTrainingLogs('New qualityRisk tail final update absent')
 unique={'L':set(),'U':set()};packets=Counter();batches=Counter();repeats=Counter();bank_ids=set(map(tuple,bank['source_ids'].tolist()));source_labels={tuple(z):int(t) for z,t in zip(bank['source_ids'],bank['meta']['TX'][np.unique(bank['meta']['source_L_index'],return_index=True)[1]])}
 for q in logs.records(b/'training_sequence.jsonl.gz'):
  role=q['role'];ids=q['ids']
  if role not in unique or len(ids)!=(128 if role=='L' else 124 if batches['U']%222==221 else 256):raise ValueError('Original physical L/U batch sizes differ')
  for z in ids:
   if len(z)!=5 or z[1] not in (1,3,4,6,8) or z[2] not in (1,2,3):raise ValueError('Original source RX/day permission differs')
   unique[role].add(tuple(z))
  if role=='L':
   if batches['L']>=STEPS:raise ValueError('Extra primary L stream outside tail')
   c=steps[batches['L']]
   if ids!=c['current_L_sample_ids'] or any(source_labels.get(tuple(z))!=t for z,t in zip(ids,c['current_L_TX'])):raise ValueError('Original currentL true labels/PID sequence differs from sharedbank source')
   nrepeat=128-len(set(map(tuple,ids)));repeats['batches_with_repeated_physical_ID']+=bool(nrepeat);repeats['repeated_positions']+=nrepeat
  packets[role]+=len(ids);batches[role]+=1
 if batches!=dict(L=2220,U=2220) or packets!=dict(L=284160,U=567000) or unique['L']!=bank_ids or len(unique['U'])!=56700 or unique['L']&unique['U'] or counts['U_forward_packets']!=packets['U']:raise ValueError('Full currentL6300/U56700 hidden stream required')
 if cost['student_train_NN']!=2271960 or cost['student_eval_NN']!=324002 or cost['teacher_train_NN'] or cost['teacher_eval_NN']!=567000:raise ValueError('Actual four counters/source loops differ')
 a=result['quality_risk']
 if (a['mode'],a['bank_size'],a['bank_sampling_calls'],a['bank_cache_reads'],a['selected_packets'],a['ordinary_complement_packets'],a['training_propagation_simulations'],a['shared_propagation_simulations'],a['per_step_L_NN'],a['pair_weights'],a['total_L_scale'],a['As_expected_weight'])!=(mode,BANK_SIZE,2220,142080,142080,142080,0,BANK_SIZE,768,[.5,.5,.25],1.25,.25) or a['bank_bin_counts']!=bank['bin_counts'].tolist() or not np.allclose(a['nominal_bin_mass'],bank['nominal_mass'],rtol=0,atol=1e-15) or not all(a[k] for k in ('shared_bank_cost_counted_once_at_run_level','selected_true_label_metadata','old_currentL_labels_not_reused_for_bank','finite_bank_risk_equivalence_only')) or any(a[k] for k in ('extra_NN','extra_CE_operator_calls','online_risk_equivalence_claim','common_U_trajectory_equality_claim','global_RNG_used','sampler_current_L_changed')):raise ValueError('Actual finite-bank scope differs')
 if (a['CE_operator_calls'],a['baseline_CE_operator_calls'],a['CE_logit_packets'],a['baseline_CE_logit_packets'],a['new_weight_multiplication_packets'])!=(4440,4440,568320,568320,284160) or a['bin_counts']!=bins_count.tolist() or not np.allclose(a['bin_weighted_mass'],bins_mass,rtol=0,atol=1e-8):raise ValueError('Actual bin exposure or weighted-loss operator cost differs')
 for k in TIMERS:logs._near(a[k],timers[k],'Actual disjoint CPUwall sum differs: '+k,1e-12)
 def counter_rows(values):return {tuple(z[k] for k in ('TX','RX','day','bin','scene','state','lock')):z['mass'] for z in values}
 for k,expected in [('selected_TX_RX_day_bin_scene_state_lock_exposure',strata),('weighted_TX_RX_day_bin_scene_state_lock_mass',weighted_strata)]:
  saved=counter_rows(a[k])
  if set(saved)!=set(expected):raise ValueError('Complete bank exposure strata differ')
  for key_,value in expected.items():logs._near(saved[key_],value,'Actual bank exposure/weighted mass differs')
 with np.load(b/'bank_exposure.npz',allow_pickle=False) as z:exposure={k:z[k].copy() for k in z.files}
 for k,expected in [('bank_index_counts',index_count),('source_L_counts',physical_count),('source_L_weighted_mass',physical_mass),('bin_counts',bins_count),('bin_weighted_mass',bins_mass)]:
  if exposure[k].shape!=expected.shape or not np.allclose(exposure[k],expected,rtol=0,atol=1e-8):raise ValueError('Saved complete exposure NPZ differs from full2220 stream: '+k)
 coverage=Counter()
 for z in result['coverage']:coverage[(z['TX'],z['RX'],z['scene'],z['state'],z['lock'])]+=z['count']
 expected_coverage=Counter()
 for (tx,rx,day,g,scene,state,lock),n in strata.items():expected_coverage[(tx,rx,scene,state,lock)]+=n
 if coverage!=expected_coverage:raise ValueError('Actual labeled bankAs channel coverage differs')
 ordinary=read(b/'ordinary_augmentation_actual.json');args=read(b/'actual_native_args.json');args=args.get('args',args)
 if ordinary['schedule_frozen_at']!=13 or ordinary['schedule_args_epochs']!=20 or not ordinary['no_ordinary_before_practical'] or not ordinary['RDrop_same_IQ'] or args['label_smoothing']!=.01 or not args['use_aug'] or not args['use_unlabeled'] or not args['use_ema_teacher']:raise ValueError('Original ordinary parent schedule/globalLS/U/EMA flags differ')
 text=(root.parent/'logs'/(root.name+'.log')).read_text(encoding='utf8',errors='replace');errors={k:len(re.findall(k,text)) for k in ('Traceback','CUDA out of memory',r'\bKilled\b','RuntimeError','AssertionError')}
 if any(errors.values()):raise ValueError('Technical error marker in completed stdout')
 curves=[dict(epoch=z['epoch']+20,local_epoch=z['epoch'],**{k:v/222 for k,v in sums[z['epoch']].items()},**{k:z[k] for k in ('lr','val_tx_acc','epoch_time_s','train_grad_backbone','train_grad_total')}) for z in epochs]
 risk={k:v for k,v in a.items() if k not in ('bank','bank_complete')};mass_summary=dict(length=len(physical_mass),minimum=float(physical_mass.min()),maximum=float(physical_mass.max()),mean=float(physical_mass.mean()),quantiles=np.quantile(physical_mass,[0,.05,.5,.95,1]).tolist())
 return dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',cost=dict(cost,wall_seconds=result['wall_seconds'],local_epochs=10,local_updates=2220,parent_epochs=20,parent_updates=4440,cumulative_epochs=30,cumulative_updates=6660,L_NN=1704960,As_packets=142080),counts=dict(counts),physical_loader_packets=dict(packets),unique_physical={k:len(v) for k,v in unique.items()},quality_risk=risk,exposure_summary=dict(bank_index_counts=hist_summary(index_count),source_L_counts=hist_summary(physical_count),source_L_weighted_mass=mass_summary,bin_counts=bins_count.tolist(),bin_weighted_mass=bins_mass.tolist(),primary_sampler_repetitions=dict(repeats),no_raw_physical_IDs=True),risk_accounting=dict(**dict(timers),CE_operator_calls=4440,baseline_CE_operator_calls=4440,CE_logit_packets=568320,baseline_CE_logit_packets=568320,extra_CE_operator_calls=0,new_weight_multiplication_packets=284160,extra_NN=0,timing=a['timing'],CPU_scopes_disjoint=True,not_CUDA_event=True,loss_recount_weight_basis='actual logged float32 applied_bank_weights',exposure_mass_basis='nominal float64 weights;not claimed equal to float32 mass'),EMA=dict(initialization=ema,local_updates=2220,cumulative_updates=6660),native_telemetry=telemetry,epoch_curves=curves,maximum_L_decomposition_error=maxerr,stdout=dict(lines=len(text.splitlines()),error_markers=errors,all_scanned=True)),dict(args=args,ordinary_mse=ordinary_mse)


def training_analysis(project):
 base=Path(project)/'runs';prior=read(base/PRIOR_E50_RUN/'analysis_auto_r01/training_logs.json')
 if prior['status']!='VERIFIED_COMPLETE_TRAINING_LOGS':raise ValueError('Existing complete original30 audit required')
 result={};lookup=dict(rows(project));publicbank,privatebank=audit_bank(project)
 for name in CONDITIONS[:2]:
  r=prior['rows'][name]
  if r['cost']['new_updates']!=6660 or r['cost']['new_epochs']!=30 or len(r['epoch_curves'])!=30:raise ValueError('Original30 complete saved audit schema differs')
  result[name]=dict(reused=True,cost=r['cost'],epoch_curves=r['epoch_curves'],evidence='Existing original30 complete audit reused;old stepstream notrescanned')
 parent_result=read(lookup['AS30']/'stage20/result.json');parentcost=actual_cost(parent_result);parent={};comparability={};args=read(lookup['AS30']/'source/actual_native_args.json');args=args.get('args',args)
 for name in CONDITIONS[2:]:
  candidate,private=parse_new_logs(lookup[name],name,privatebank);candidate['reused']=False;result[name]=candidate;parent[name]=dict(cost=parentcost,epochs=20,updates=4440,wall_seconds=parent_result['wall_seconds'],reused=True,shared_originalAs20=True)
  oldcost=result['AS30']['cost'];oldtail={k:oldcost[k]-parentcost[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')};comparability[name]=dict(reference_condition='AS30',old_step_logs_rescanned=False,old_tail_four_counters=oldtail,new_tail_four_counters=candidate['cost'],four_counters_equal=all(oldtail[k]==candidate['cost'][k] for k in oldtail),different_actual_arg_keys=sorted(k for k in set(args)|set(private['args']) if args.get(k)!=private['args'].get(k)),U_recipe_unchanged=True,common_U_or_full_IQ_trajectory_not_claimed=True)
 return dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',run_id=RUN,rows=result,parent20=parent,bank=publicbank,new_model_rows=2,new_model_epochs=20,new_updates=4440,cumulative_candidate_epochs_each=30,cumulative_candidate_updates_each=6660,reference_comparability=comparability,shared_parent_count=1,analysis_NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_scores_read=False)


def formal_analysis(project,truth_path,source):
 started=time.perf_counter();metadata=ensure_metadata(project,'final')
 if source['status']!='SOURCE_ANALYZED' or source['run_id']!=RUN or source['target_scores_read']:raise ValueError('Independent source analysis must finish first')
 truth=read(truth_path);results={};private={};all_metrics=[];partitions=0
 for arm,root in rows(project):
  with np.load(root/'prediction/predictions.npz',allow_pickle=False) as f:
   if set(f.files)!={'ids',*VIEWS}:raise ValueError('Exact seven fixed decisions required')
   ids=f['ids'].copy();pred={v:f[v].copy() for v in VIEWS}
  if len(ids)!=COUNT or len(set(ids.tolist()))!=COUNT or set(truth)!=set(ids.tolist()):raise ValueError('Exact frozen physical ID/truth alignment required')
  y=np.asarray([truth[z]['label'] for z in ids]);rx=np.asarray([str(truth[z]['receiver']) for z in ids]);day=np.asarray([str(truth[z]['day']) for z in ids]);saved=read(root/'scoring/scores.json')
  if not saved['truth_last'] or not saved['target_feedback_forbidden']:raise ValueError('Independent original scorer boundary differs')
  if private and (not np.array_equal(ids,private['NOAS30']['ids']) or not np.array_equal(y,private['NOAS30']['y'])):raise ValueError('All rows must share physical pairing')
  metrics=[];views={}
  for v in VIEWS:
   with np.load(root/'prediction'/(v+'_scores.npz'),allow_pickle=False) as f:
    z=f['logits']
    if not np.array_equal(ids,f['ids']) or z.shape!=(COUNT,6) or not np.isfinite(z).all() or not np.array_equal(z.argmax(1),pred[v]):raise ValueError('Default logits/fixed decisions differ')
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    subtotal=np.zeros((6,6),dtype=np.int64)
    for value in values:
     mask=np.ones(COUNT,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=metric(y[mask],pred[v][mask]);original=next(r for r in saved['results'] if r['view']==v and r['dimension']==dim and r['stratum']==str(value))
     if q['confusion']!=original['confusion'] or q['count']!=original['query_count'] or abs(q['accuracy']-original['accuracy'])>1e-12 or abs(q['macro_f1']-original['macro_f1'])>1e-12:raise ValueError('Independent CM/F1 recount differs')
     metrics.append(dict(arm=arm,row_id=root.name,model_seed=SEED,view=v,dimension=dim,stratum=str(value),**q));subtotal+=np.asarray(q['confusion'])
    if dim!='overall':
     if not np.array_equal(subtotal,metric(y,pred[v])['confusion']):raise ValueError('Formal RX/day/TX partitions differ')
     partitions+=1
   q=next(r for r in metrics if r['view']==v and r['dimension']=='overall');views[v]=dict(q,worst_RX_accuracy=min(r['accuracy'] for r in metrics if r['view']==v and r['dimension']=='receiver'))
  results[arm]=dict(views=views,urban_mean_accuracy=float(np.mean([views[v]['accuracy'] for v in VIEWS[4:]])),prediction_NN=metadata[arm]['prediction/complete.json']['actual_NN'],prediction_wall_seconds=metadata[arm]['prediction/complete.json']['wall_seconds'],reused=arm in CONDITIONS[:2]);private[arm]=dict(ids=ids,y=y,rx=rx,day=day,pred=pred);all_metrics.extend(metrics)
 paired_strata=[];pairs={};comparisons={};cohort=private['NOAS30']['pred']['clean']==private['NOAS30']['y']
 for a,b in PAIRS:
  key=a+'__'+b;pairs[key]={};comparisons[key]={}
  for v in VIEWS:
   p=paired(private[a]['y'],private[b]['pred'][v],private[a]['pred'][v],cohort);pairs[key][v]=p;comparisons[key][v]=dict(accuracy_delta_pp=p['accuracy_delta_pp'],macro_F1_delta_pp=100*(results[a]['views'][v]['macro_f1']-results[b]['views'][v]['macro_f1']),worst_RX_delta_pp=100*(results[a]['views'][v]['worst_RX_accuracy']-results[b]['views'][v]['worst_RX_accuracy']))
  for v in VIEWS:
   for dim,values in [('receiver',sorted(set(private[a]['rx']))),('day',sorted(set(private[a]['day']))),('transmitter',range(6))]:
    for value in values:
     mask=private[a]['rx']==value if dim=='receiver' else private[a]['day']==value if dim=='day' else private[a]['y']==value
     paired_strata.append(dict(comparison=key,view=v,dimension=dim,stratum=str(value),**paired(private[a]['y'][mask],private[b]['pred'][v][mask],private[a]['pred'][v][mask],cohort[mask])))
  comparisons[key]['urban_mean_accuracy_delta_pp']=float(np.mean([comparisons[key][v]['accuracy_delta_pp'] for v in VIEWS[4:]]))
 risk_increment={v:comparisons['BANK_STRAT30__BANK_IID30'][v]['accuracy_delta_pp'] for v in VIEWS};risk_increment['urban_mean']=comparisons['BANK_STRAT30__BANK_IID30']['urban_mean_accuracy_delta_pp']
 key='BANK_STRAT30__NOAS30';clean=pairs[key]['clean'];d=clean['accuracy_delta_pp'];stage=dict(clean_delta_pp=d,clean_net_repairs=clean['net_repairs'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(clean['net_repairs'],COUNT),clean_positive_observed=d>0,positive_below_0p1pp_is_micro_observation=0<d<.1,midurban_gain_pp=comparisons[key]['practical_mid_urban']['accuracy_delta_pp'],lowurban_gain_pp=comparisons[key]['practical_low_urban']['accuracy_delta_pp'],urban_mean_gain_pp=comparisons[key]['urban_mean_accuracy_delta_pp'],repeatable_clean_gain_proven=False)
 stages={a:dict(clean_delta_pp=pairs[a+'__NOAS30']['clean']['accuracy_delta_pp'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(pairs[a+'__NOAS30']['clean']['net_repairs'],COUNT),lowurban_gain_pp=comparisons[a+'__NOAS30']['practical_low_urban']['accuracy_delta_pp'],midurban_gain_pp=comparisons[a+'__NOAS30']['practical_mid_urban']['accuracy_delta_pp'],repeatable_clean_gain_proven=False) for a in CONDITIONS[2:]}
 return dict(status='FORMAL_RECOUNT_COMPLETE',run_id=RUN,summary=results,metrics=all_metrics,paired=pairs,paired_strata=paired_strata,comparisons=comparisons,stage=stage,stages=stages,source_decision=source['decision'],stratified_risk_effect_pp=risk_increment,clean_vs_original_NoAs=dict(accuracy_delta_pp=pairs['BANK_STRAT30__NOAS30']['clean']['accuracy_delta_pp'],clean_absolute_0p1pp_guard_met=old.stable.stage_clean_pass(pairs['BANK_STRAT30__NOAS30']['clean']['net_repairs'],COUNT)),suburban_selfclean_gaps_pp={a:{v:100*(r['views']['clean']['accuracy']-r['views'][v]['accuracy']) for v in VIEWS[1:4]} for a,r in results.items()},model_seed_count=1,model_seed=SEED,independent_confusion_matrices=len(all_metrics),partition_reconciliations=partitions,independent_recount='VERIFIED',NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,target_feedback=False,single_seed_only=True,repeatable_clean_positive_from_As_proven=False,formal_state_accuracy_available=False,goal_complete=False,wall_seconds=time.perf_counter()-started)

def cost_summary(project,training):
 meta=ensure_metadata(project,'final');out={}
 for arm,_ in rows(project):
  c=training['rows'][arm]['cost'];native=c['native_training_and_validation_NN'];q=meta[arm]['source_qualification/complete.json'];p=meta[arm]['prediction/complete.json']
  out[arm]=dict(native_NN=native,student_train_NN=c['student_train_NN'],student_eval_NN=c['student_eval_NN'],teacher_train_NN=c['teacher_train_NN'],teacher_eval_NN=c['teacher_eval_NN'],source_readout_NN=q['NN'],formal_prediction_NN=p['actual_NN'],segment_and_own_readouts_NN=native+q['NN']+p['actual_NN'],native_wall_seconds=c['wall_seconds'],formal_prediction_wall_seconds=p['wall_seconds'],reused=arm in CONDITIONS[:2],source_readout_wall_seconds=q.get('wall_seconds'))
 root=Path(project)/'runs'/RUN;smoke=read(root/'synthetic_smoke.json');real=read(root/'real_checkpoint_smoke.json')
 if smoke['status']!='PASS' or real['status']!='PASS' or smoke['target_access'] or real['target_query']:raise ValueError('Actual shared noquery smoke incomplete')
 shared=smoke['total_NN']
 if shared!=smoke['actual_L_NN']+smoke['real_checkpoint_NN'] or real['NN']!=smoke['real_checkpoint_NN']:raise ValueError('Shared smoke total already includes the real-checkpoint forward')
 shared_detail=dict(total_NN=shared,synthetic_L_NN=smoke['actual_L_NN'],real_checkpoint_NN=real['NN'],extra_CE_operator_calls=smoke['extra_CE_operator_calls'],extra_CE_logit_packets=smoke['extra_CE_logit_packets'],real_checkpoint_not_charged_twice=True)
 history_path=root/'technical_history'/'r01'/'cost.json';technical=[];technical_NN=0
 if history_path.is_file():
  h=read(history_path);keys=('total_NN','real_parent_forward_packets','synthetic_ordinary_forward_packets','optimizer_updates','real_IQ','propagation_simulations');expected=(258,2,256,0,0,0)
  if tuple(h[k] for k in keys)!=expected or h['total_NN']!=h['real_parent_forward_packets']+h['synthetic_ordinary_forward_packets']:raise ValueError('Preserved R01 technical-failure cost differs from actual no-update boundary')
  technical=[dict(release_attempt='r01',**{k:h[k] for k in keys})];technical_NN=h['total_NN']
 parents={a:v['cost']['native_training_and_validation_NN'] for a,v in training['parent20'].items()};segments=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[2:])
 return dict(rows=out,new_model_epochs=20,new_updates=4440,risk_accounting={a:training['rows'][a].get('risk_accounting',{}) for a in CONDITIONS[2:]},new_training_and_readouts_NN=segments,new_shared_smoke_NN=shared,shared_smoke_cost=shared_detail,new_total_NN=segments+shared+technical_NN,scientific_new_training_and_readouts_plus_smoke_NN=segments+shared,technical_failure_NN=technical_NN,technical_failure_history=technical,technical_history_counted_once=True,parent20_native_NN_reused=parents,cumulative_candidate_native_NN={a:parents[a]+out[a]['native_NN'] for a in CONDITIONS[2:]},cumulative_candidate_epochs_each=30,cumulative_candidate_updates_each=6660,old_two_E30_training_and_readouts_NN_reused=sum(out[a]['segment_and_own_readouts_NN'] for a in CONDITIONS[:2]),old_costs_not_recharged=True,shared_originalAs20_parent_native_NN_counted_once=next(iter(parents.values())),bank_simulation=training['bank'],analysis_NN=0,analysis_IQ_reads=0,analysis_checkpoint_reads=0,analysis_updates=0,wall_scope='Actual model-seconds;not speed evidence')

def render_report(source,formal,training,costs):
 d=source['decision'];s=formal['stage'];bank=training['bank'];lines=['# 有限 As bank 的质量分层风险：E30 四条件结果','',RUN+'；模型 seed 2026092701。两个新行均由同一个原 P43 B01 ownAs20 学生与 ownEMA 追加 10 轮；原 NoAs30/As30 的完整冻结读出复用。','',f"源三条件 PASS={d['source_criteria_pass']}，固定 stage1 关闭={d['fixed_stage1_closed']}。hard<8 dB 的 S−I 等场景变化 {d['hard_lt8_S_minus_I_equal_scene_accuracy_pp']:+.6f} pp（要求≥+.5），easy≥8 dB {d['easy_ge8_S_minus_I_equal_scene_accuracy_pp']:+.6f} pp（要求≥−.2），fullV clean S−原As {d['fullV_clean_S_minus_originalAs_pp']:+.6f} pp（要求≥−.1）。源判定先于正式读取，失败关闭/noM3/noE50；通过仅有另行登记M3资格，分析不自动启动。",'',f"正式 S clean−原NoAs {s['clean_delta_pp']:+.6f} 个绝对百分点，0.1 pp 阶段约束={s['clean_absolute_0p1pp_guard_met']}。low/mid urban−NoAs {s['lowurban_gain_pp']:+.6f}/{s['midurban_gain_pp']:+.6f} pp。该正式阶段判定与源资格不同；n=1，不能宣称可重复双提升。",'', 'I01/BANK_IID30 与 S01/BANK_STRAT30 使用同一 L-only bank（6300×40=252000 个完整原 practical 观测）。I每步从有限bank有放回均匀抽64、权重1；S每bin有放回抽16，权重4p_g，p_g=N_g/N_bank。四bin依据实际 signal_to_added_noise_db：<3、[3,8)、[8,15)、≥15。它不是总SNR，不是白输入 quality_snr_db，也不以locked/degraded排除BLOCKED或unlocked。', '', '两行对同一有限bank的名义风险有相同期望，不能冒称与在线无限practical风险相同。原ordinary128/ORD128对比、raw128、6L/768NN、总L尺度1.25、As .25份额、LS.01和sameIQ两次RDrop CE/KL保留。mixed原64槽被bank IQ及其真实标签/domain替换，CE和对称KL均按包权重乘后除128，不对selected权重或组内样本重新归一化。全部bank合法L仍可监督；没有额外部署模块、置信门或目标物理标签。','', '|视图|NoAs30 %|原As30 %|bank IID %|bank stratified %|S−I pp|I−原As pp|S−原As pp|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for v in (*VIEWS,'urban_mean'):
  vals=[100*(formal['summary'][a]['urban_mean_accuracy'] if v=='urban_mean' else formal['summary'][a]['views'][v]['accuracy']) for a in CONDITIONS];diff=[formal['comparisons'][a+'__'+b]['urban_mean_accuracy_delta_pp'] if v=='urban_mean' else formal['comparisons'][a+'__'+b][v]['accuracy_delta_pp'] for a,b in PAIRS[:3]];lines.append('|'+v+'|'+'|'.join(f'{x:.6f}' for x in vals+diff)+'|')
 lines+=['','|条件|视图|accuracy %|macro-F1 %|最差RX %|','|---|---|---:|---:|---:|']
 for a in CONDITIONS:
  for v in VIEWS:
   q=formal['summary'][a]['views'][v];lines.append(f"|{a}|{v}|{100*q['accuracy']:.6f}|{100*q['macro_f1']:.6f}|{100*q['worst_RX_accuracy']:.6f}|")
 lines+=['','|比较|视图|修复|新增错误|净修复|','|---|---|---:|---:|---:|']
 for a,b in PAIRS:
  for v in VIEWS:
   q=formal['paired'][a+'__'+b][v];lines.append(f"|{a}−{b}|{v}|{q['repairs']}|{q['new_errors']}|{q['net_repairs']}|")
 lines+=['','S−I是分层加权相对同bank IID的增量；I−原liveAs30体现有限bank替代的近似变化；不能将S−原As全部归为分层收益。全部六配对和RX/day/TX错误转移在formal_summary保留；无虚构target质量/遮挡标签。','', '|条件|fullV clean %|V1800 clean %|两draw mid/low %|','|---|---:|---:|---:|']
 for a in CONDITIONS:
  q=source['summary'][a];lines.append(f"|{a}|{100*q['full_V_clean']:.6f}|{100*q['panel']['clean']['accuracy']:.6f}|{100*q['mid_low_mean']:.6f}|")
 lines+=['','|条件|源场景|分组|n(两draw)|accuracy %|macro-F1 %|真实类margin/30|','|---|---|---|---:|---:|---:|---:|']
 for a in CONDITIONS:
  for v in SOURCE_SCENES[1:]:
   for group,q in source['quality_groups'][a][v].items():lines.append(f"|{a}|{v}|{group}|{q['count']}|{100*q['accuracy']:.6f}|{100*q['macro_F1']:.6f}|{q['mean_trueclass_margin_over30']:.9f}|")
 lines+=['', 'hard/easy在每个mid/low场景先合并两个固定draw，再两场景等权；不按组样本量跨场景加权。source_metrics含各draw与pooled、四bin、全部state×lock及RX/day/TX、count/F1和已有logits算出的margin；这些描述不是新增门槛。fullV27000只有原生accuracy/RX，不虚构逐包F1。V1800两draw不是两个模型seed，源无highurban；既有panel2.45GHz与训练/正式2.462GHz差异保留。','', '|条件|suburban视图|绝对accuracy %|自身clean−suburban pp|','|---|---|---:|---:|']
 for a in CONDITIONS:
  for v,g in formal['suburban_selfclean_gaps_pp'][a].items():lines.append(f"|{a}|{v}|{100*formal['summary'][a]['views'][v]['accuracy']:.6f}|{g:.6f}|")
 lines+=['', 'clean下降也会缩小suburban差距，须同时看绝对准确率；不将缩差自动当保护。全5源RX/L6300/U56700真值隐藏/V27000/split392005/Q256/defaultcos联合训练权限不变，无RX留出。','', '|条件|student train NN|student eval NN|teacher train NN|teacher eval NN|native NN|source NN|formal NN|native秒|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for a in CONDITIONS:
  c=costs['rows'][a];lines.append('|'+a+'|'+'|'.join(str(c[k]) for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_NN','source_readout_NN','formal_prediction_NN'))+f"|{c['native_wall_seconds']:.3f}|")
 lines+=['',f"新增两尾段共20modelEpoch/4440更新，实际训练及读出{costs['new_training_and_readouts_NN']}NN，共享smoke{costs['new_shared_smoke_NN']}NN，科学训练/读出及共享smoke合计{costs['scientific_new_training_and_readouts_plus_smoke_NN']}NN；另有R01技术失败{costs['technical_failure_NN']}NN（0更新/0真实IQ/0传播模拟），整个候选实际{costs['new_total_NN']}NN（科学参考8703898，含已知R01失败参考8704156，以实计为准）。同一原As20父只实际复用一次，各候选累计轨迹分别披露；旧两E30及历史读出不重计。bank构造{bank['observations']}完整物理模拟另计，无模型NN，wall/storage以bank summary实际记录为准。",'', 'bank采样/缓存复制/加权CE-KL约简的CPUwall与实际调用数在training_logs和costs.risk_accounting中；无新NN不等于零计算，也不把CPUwall当CUDAevent或把包含关系的timer相加。source预测wall未单列即缺失，不用离线分析wall替代。', '', f"全部两新行4440步/20轮及L/U流完整解析，旧审计不回扫；4×7固定预测及独立scorer完成后才truth join，重核{formal['independent_confusion_matrices']}CM/{formal['partition_reconciliations']}分区。源先判、正式不反馈候选/参数/seed或救跑。单seed不足可重复clean正收益，目标未由本轮自动完成。分析NN/CP/IQ/updates均0。"]
 return '\n'.join(lines)+'\n'

def metadata_paths(project,stage):
 paths=[]
 for _,root in rows(project):
  paths += [root/'source/completion.json',root/'source/source_frozen.json',root/'source_qualification/complete.json']
  if stage=='FINAL':paths += [root/'prediction/complete.json',root/'scoring/complete.json',root/'complete.json']
 return paths

def ready(paths):
 try:
  for path in paths:read(path)
 except (FileNotFoundError,json.JSONDecodeError):return False
 return True

def owned_handles(project,release):
 base=Path(project)/'runs'/RUN;ids=[];unknown=False
 for name in ('dispatcher.json','launch.json'):
  try:
   r=read(base/name);ids += [r['pid']] if name=='dispatcher.json' else [v['pid'] for v in r['rows']]
  except FileNotFoundError:pass
  except json.JSONDecodeError:unknown=True
 live=[]
 for pid in sorted(set(ids)):
  p=Path('/proc')/str(pid)
  try:
   argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   if Path(os.readlink(p/'cwd')).resolve()==Path(release).resolve() and 'experiments/cvs_quality_risk30/runner.py' in argv:live.append(pid)
  except FileNotFoundError:pass
  except OSError:unknown=True
 return live,unknown

def wait_until(project,release,stage,out,readiness=ready,probe=owned_handles,sleep=time.sleep):
 while not readiness(metadata_paths(project,stage)):
  base=Path(project)/'runs'/RUN
  if (base/'failure.json').is_file():raise RuntimeError('Owned technical failure preserved')
  for arm,own in rows(project):
   if arm in CONDITIONS[2:] and (own/'source_qualification_failure.json').is_file():raise AwaitingArtifacts('Own source readout failed;formal closure remains with original runner')
  live,unknown=probe(project,release)
  if not live and not unknown:raise AwaitingArtifacts('Owned process handles absent before '+stage+' artifacts complete')
  write(out/'state.json',dict(status='WAIT_'+stage,live_owned_PIDs=live,observation_unknown=unknown,truth_opened=False,NN=0));sleep(15)

def validate_output(project,output):
 out=Path(output)
 if out.parent!=Path(project)/'runs'/RUN or re.fullmatch(r'analysis_auto_r[0-9]{2}',out.name) is None:raise ValueError('Only own immutable analysis output root')
 if out.exists():raise FileExistsError('Reconcile existing output;no duplicate launch')
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--truth',type=Path,required=True);p.add_argument('--code-commit',required=True);a=p.parse_args()
 if a.release.name!=TRAINING_RELEASE:raise ValueError('Only registered actual training release')
 out=validate_output(a.project,a.output);out.mkdir();started=time.perf_counter();write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,analysis_code_commit=a.code_commit,NN=0,truth_opened=False))
 try:
  wait_until(a.project,a.release,'SOURCE',out);training=training_analysis(a.project);write(out/'training_logs.json',training);source=source_analysis(a.project);write(out/'source_summary.json',{k:v for k,v in source.items() if k!='metrics'});write(out/'source_confusions.json',source['metrics']);write_csv(out/'source_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in source['metrics']]);write(out/'source_complete.json',dict(status='SOURCE_ANALYZED',training_run=RUN,NN=0,target_scores_read=False));print('SOURCE_ANALYSIS_COMPLETE',flush=True)
  wait_until(a.project,a.release,'FINAL',out);formal=formal_analysis(a.project,a.truth,source);write(out/'formal_summary.json',{k:v for k,v in formal.items() if k!='metrics'});write(out/'formal_confusions.json',formal['metrics']);write_csv(out/'formal_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in formal['metrics']]);costs=cost_summary(a.project,training);write(out/'costs.json',costs);(out/'final_report.md').write_text(render_report(source,formal,training,costs),encoding='utf8');write(out/'complete.json',dict(status='ANALYZED',training_run=RUN,analysis_code_commit=a.code_commit,source_complete=True,formal_complete=True,log_records_complete=True,model_seed_count=1,new_model_rows=2,reused_model_rows=2,NN=0,IQ=0,CP=0,updates=0,training_processes_untouched=True,pipeline_wall_including_wait_seconds=time.perf_counter()-started,repeatable_clean_positive_proven=False,goal_complete=False));write(out/'state.json',dict(status='ANALYZED',NN=0,truth_opened=True));print('ANALYSIS_COMPLETE',flush=True)
 except BaseException as exc:
  write(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),training_processes_untouched=True,automatic_retry=False,NN=0));raise
if __name__=='__main__':main()
