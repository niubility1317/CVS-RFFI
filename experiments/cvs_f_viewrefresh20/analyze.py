"""Independent complete source recount;stdlib only,no IQ/CP/model/target."""
import argparse,csv,gzip,json,math,random,re,statistics
from collections import Counter,defaultdict
from pathlib import Path
RUN='20261017-phase1-FcrossRX-viewrefresh-sourceRX1-e20-s2026092701-m3-r01';ARMS=('RAW','REPLAY','REFRESH');SEED=2026092701;FIT=(3,4,6,8)
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
def lines(p):
 op=gzip.open if str(p).endswith('.gz') else open
 with op(p,'rt',encoding='utf8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def near(a,b,tol=2e-5):
 if not math.isfinite(a+b) or abs(a-b)>tol*max(1.,abs(a),abs(b)):raise ValueError('Independent scalar recount differs')
def metrics(y,p):
 if len(y)!=len(p) or not y or any(a not in range(6) or b not in range(6) for a,b in zip(y,p)):raise ValueError('All six registeredclass decisions required')
 C=[[0]*6 for _ in range(6)]
 for a,b in zip(y,p):C[a][b]+=1
 n=len(y);correct=sum(C[k][k] for k in range(6));f=[]
 for k in range(6):
  den=sum(C[k])+sum(z[k] for z in C);f.append(2*C[k][k]/den if den else 0.)
 return dict(count=n,correct=correct,accuracy=correct/n,macro_F1=sum(f)/6,confusion=C)
def predictions(base):
 result={};data={};common=None
 for view in VIEWS:
  r=read(base/'source_readouts'/(view+'.json'));ids=[tuple(p) for p in r['physical_ids']];y=r['labels'];p=r['predictions']
  if len(ids)!=1260 or len(set(ids))!=1260 or set(z[1] for z in ids)!={1} or set(z[2] for z in ids)!={1,2,3} or len(y)!=1260 or Counter(y)!=Counter({k:210 for k in range(6)}) or len(p)!=1260 or r['head_fit_updates']:raise ValueError('Exact heldsourceL1260 role required')
  if common is None:common=(ids,y)
  elif common!=(ids,y):raise ValueError('View physicalIDs/truth pairing differs')
  m=metrics(y,p);near(m['accuracy'],r['accuracy']);near(m['macro_F1'],r['macro_F1'])
  if 'margin_over30' in r and (len(r['margin_over30'])!=1260 or any(not math.isfinite(x) for x in r['margin_over30'])):raise ValueError('Finite originalcos30 margins required')
  result[view]=m;data[view]=r
 return result,data

def training(base,arm):
 r=read(base/'satmix_result.json');done=read(base/'completion.json');init=read(base/'initialization.json');freeze=read(base/'source_frozen.json');contract=read(base/'source_contract.json');args=read(base/'actual_native_args.json');cfg=read(base/'resolved_config.json')
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access'] or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']:raise ValueError('Verified fixed scratch FINAL20 defaulthead provenance required')
 if args['seed']!=SEED or (args['epochs'],args['label_epochs'],args['pseudo_epochs'],args['label_smoothing'])!=(20,13,7,.01) or args['baseline_ckpt'] or args['teacher_ckpt'] or args['source_fold_held_rx']!=1 or contract['source_receivers']!=list(FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Exact seed and source data contract required')
 if cfg['arm']!=arm or cfg['training_mechanism']['L_forwards_per_packet']!=8 or not args['use_aug'] or args['use_concat_sat_channel_aug']:raise ValueError('Fixed registered8L/ordinary/noAs training required')
 if r['actual_steps']!=4440 or r['L_forward_packets']!=4546560 or r['extra_L_forward_packets']!=1136640 or r['supcon_applied']!=3330 or r['As_training_observations'] or not r['default_head_joint'] or r['head_fit_updates'] or r['target_access'] or r['U_truth_read'] or r['checkpoint_sources'] or r['teacher_additive_calls']:raise ValueError('Actual8L/defaulthead/hiddenU source budget differs')
 steps=list(lines(base/'satmix_steps.jsonl.gz'));units=list(lines(base/'unit_steps.jsonl.gz'));native=list(lines(base/'step_metrics.jsonl.gz'));epochs=list(lines(base/'epoch_metrics.jsonl'));metrics_epochs=list(lines(base/'metrics_epoch.jsonl'))
 if len(steps)!=4440 or len(units)!=4440 or len(native)!=4440 or len(epochs)!=20 or len(metrics_epochs)!=20:raise ValueError('Complete 4440step/20epoch records required')
 for f in ('epoch_metrics.csv','metrics_epoch.csv'):
  with (base/f).open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
  if len(rows)!=20:raise ValueError('Complete CSV records required')
 grad=[];curves=[];sup=0;native_nulls=Counter()
 for i,(a,u,n) in enumerate(zip(steps,units,native),1):
  e=(i-1)//222+1
  if (a['step'],a['epoch'],u['step'],n['step'],n['epoch'])!=(i,e,i,i,e) or u['physical_clip_calls']!=i:raise ValueError('Full ordered native/F steps differ')
  for k in ('total_L','raw_CE','ordinary_CE','s_slot_CE','raw_KL','ordinary_KL','s_slot_KL','supcon','supcon_weight','base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair_weighted','extra_raw_MSE'):
   if not math.isfinite(a[k]):raise ValueError('Nonfinite active L term')
  near(a['base_total_L'],.5*(a['raw_CE']+a['raw_KL'])+.25*(a['ordinary_CE']+a['ordinary_KL'])+.25*(a['s_slot_CE']+a['s_slot_KL'])+a['supcon_weight']*a['supcon']);near(a['total_L'],a['base_total_L']+.25*a['extra_pair_loss']);near(a['extra_pair_weighted'],.25*a['extra_pair_loss']);near(a['extra_pair_loss'],a['extra_pair_CE']+a['extra_pair_KL'])
  if a['As_packets'] or a['sat_raw_MSE'] or a['extra_As_packets'] or a['extra_pair_weight']!=.25 or a['base_raw_total_weight']!=.75 or a['base_ordinary_weight']!=.25 or a['total_pair_weight']!=1.25 or not all(a[k] for k in ('all_L_supervised','same_IQ_pairs','extra_view_once','extra_same_IQ_pair')):raise ValueError('Fixed absolute common supervision differs')
  sup+=a['supcon_weight']>0;m=n['metrics']
  for k in ('train/grad_before_clip','train/grad_clip_limit','train/grad_total','train/optimizer_step_applied','train/pseudo_truth_available'):
   if m.get(k) is None or not math.isfinite(m[k]):raise ValueError('Missing active native gradient/permission field')
  if m['train/pseudo_truth_available'] or m['train/optimizer_step_applied']!=1 or m.get('train/skipped_nonfinite_grad',0) or m.get('train/skipped_nonfinite_loss',0):raise ValueError('HiddenU truth or skipped/nonfinite update')
  grad.append(m['train/grad_before_clip'])
  for k,v in m.items():
   if v is None:native_nulls[k]+=1
 if sup!=3330:raise ValueError('Original SupCon warm5/ramp10 differs')
 for e,row in enumerate(epochs,1):
  if int(row['epoch'])!=e or int(row['epochs'])!=20 or int(row['label_epochs'])!=13 or int(row['pseudo_epochs'])!=7 or float(row['train_pseudo_truth_available']) or float(row['train_skipped_nonfinite_grad']) or float(row['train_skipped_nonfinite_loss']) or float(row['train_optimizer_step_applied'])!=1:raise ValueError('Epoch schedule/hiddenU/health differs')
  subset=steps[(e-1)*222:e*222];curves.append(dict(arm=arm,epoch=e,base_L=statistics.mean(a['base_total_L'] for a in subset),extra_CE=statistics.mean(a['extra_pair_CE'] for a in subset),extra_KL=statistics.mean(a['extra_pair_KL'] for a in subset),total_L=statistics.mean(a['total_L'] for a in subset),raw_CE=statistics.mean(a['raw_CE'] for a in subset),ordinary_CE=statistics.mean(a['ordinary_CE'] for a in subset),supcon=statistics.mean(a['supcon'] for a in subset),supcon_weight=statistics.mean(a['supcon_weight'] for a in subset),grad_before_clip=statistics.mean(grad[(e-1)*222:e*222]),train_tx_accuracy=row['train_tx_acc'],seenV_accuracy=row['val_tx_acc'],lr=row['lr'],epoch_seconds=row['epoch_time_s'],pseudo_selected=row['train_pseudo_selected'],pseudo_total=row['train_pseudo_total']))
 near(statistics.mean(a['base_total_L'] for a in steps[-222:]),r['final_base_L_objective']);near(statistics.mean(z['metrics']['train/loss_tx_labeled'] for z in native[-222:]),r['final_L_CE'])
 cost=sum(r[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets'))-2+r['held_forward_packets']
 if cost!=5823318:raise ValueError('Actual complete8L network-forward accounting differs')
 seq=list(lines(base/'training_sequence.jsonl.gz'))
 if len(seq)!=5994 or Counter(z['role'] for z in seq)!=Counter(L=4440,U=1554):raise ValueError('Complete original L/U physical sequences required')
 refresh=refresh_recount(base,arm,seq,steps,curves)
 text=(base.parents[1]/'logs'/(base.parent.name+'.stdout.log')).read_text(encoding='utf8',errors='replace')
 if re.findall(r'Traceback|CUDA out of memory|RuntimeError:|Killed|FloatingPointError:',text):raise ValueError('Critical full source stdout error')
 return r,curves,dict(refresh=refresh,stdout_lines=len(text.splitlines()),stdout_warning_lines=[z for z in text.splitlines() if re.search(r'(?i)warning|warn:',z)],stdout_nan_mentions=len(re.findall(r'(?i)nan',text)),stdout_epoch_markers=len(re.findall(r'\[EPOCH-END\]',text)),steps=4440,epochs=20,NN=cost,gradient_preclip_mean=statistics.mean(grad),gradient_preclip_max=max(grad),gradient_zero=sum(x==0 for x in grad),gradient_above5=sum(x>5 for x in grad),native_disabled_null_fields=len(native_nulls),all_active_updates_finite=True,optimizer_updates=4440,L_NN=4546560,parent_epochs=0),seq

def labeled_TX(pid):
 # Original source-cache flattened layout only for saved visible-L records.
 # Validate RX/day/sig here and every TX count/pair count against saved native audits.
 eq,rx,day,sig,base=pid;within=base%15000;rx_index=(within//1000)//3
 if eq!=1 or base<0 or base>=90000 or (1,3,4,6,8)[rx_index]!=rx or within//1000%3+1!=day or within%1000!=sig or sig not in range(70):raise ValueError('Saved visible-L metadata layout differs')
 return base//15000

KEYS={'anti':('p_dc_offset','p_bandedge_taper'),'channel':('p_time_shift','p_amp_scale','p_phase_rot','p_cfo','p_phase_noise','p_multipath','p_awgn')}
def validate_refresh_record(z,arm,i,ids):
 e=(i-1)//222+1
 if (z['step'],z['epoch'],z['arm'],z['seed'])!=(i,e,arm,20261017961+i) or z['ids']!=ids or len(ids)!=128 or len(z['labels'])!=128 or not z['all_L_supervised'] or z['new_PA_DAC_applications'] or not z['original_normal_view_unchanged']:raise ValueError('One exact real-L observation consumption per step required')
 if z['labels']!=[labeled_TX(p) for p in ids]:raise ValueError('Saved visible-L TX labels/physical IDs disagree')
 for key in ('first_prepa_raw_MSE','fresh_prepa_raw_MSE','fresh_first_prepa_MSE','normal_prepa_MSE','selected_raw_MSE','selected_first_prepa_MSE','raw_power','first_prepa_power','fresh_prepa_power','selected_power'):
  if not math.isfinite(z[key]) or z[key]<0:raise ValueError('Finite nonnegative IQ scalar required')
 if len(z['first_stages'])!=2 or len(z['fresh_stages'])!=2:raise ValueError('Exactly native anti then channel required')
 for a,b,name in zip(z['first_stages'],z['fresh_stages'],('anti','channel')):
  if a['stage']!=name or b['stage']!=name or set(a['gates'])!=set(b['gates']) or not set(a['gates'])<=set(KEYS[name]):raise ValueError('Same native stage/operator support required')
  for k,v in a['gates'].items():
   w=b['gates'][k]
   if not 0<v['probability']<=1 or v['probability']!=w['probability'] or type(v['triggered']) is not bool or type(w['triggered']) is not bool:raise ValueError('Native original/fresh scalar probabilities differ')
 chosen={'RAW':('raw_power',0.,z['first_prepa_raw_MSE']),'REPLAY':('first_prepa_power',z['first_prepa_raw_MSE'],0.),'REFRESH':('fresh_prepa_power',z['fresh_prepa_raw_MSE'],z['fresh_first_prepa_MSE'])}[arm]
 near(z['selected_power'],z[chosen[0]],1e-12);near(z['selected_raw_MSE'],chosen[1],1e-12);near(z['selected_first_prepa_MSE'],chosen[2],1e-12)
 return True

def refresh_recount(base,arm,seq,steps,curves):
 L=[x['ids'] for x in seq if x['role']=='L'];cov=Counter();unique=set();first=Counter();fresh=Counter();disagree=Counter();eligible=Counter();prob=defaultdict(list);per_epoch=defaultdict(list);n=0;distinct=0
 for i,z in enumerate(lines(base/'refresh_steps.jsonl.gz'),1):
  if i>4440:raise ValueError('Extra fresh IQ calls exceed E20')
  validate_refresh_record(z,arm,i,L[i-1]);near(z['selected_raw_MSE'],steps[i-1]['extra_raw_MSE'],1e-10)
  for tx,pid in zip(z['labels'],z['ids']):cov[tx,pid[1],pid[2]]+=1;unique.add(tuple(pid))
  for a,b in zip(z['first_stages'],z['fresh_stages']):
   for k,v in a['gates'].items():
    w=b['gates'][k];first[k]+=128*int(v['triggered']);fresh[k]+=128*int(w['triggered']);disagree[k]+=128*int(v['triggered']!=w['triggered']);eligible[k]+=128;prob[k].append(v['probability'])
  distinct+=z['fresh_first_prepa_MSE']>0;per_epoch[z['epoch']].append(z);n+=1
 audit=read(base/'refresh_audit.json');sm=audit['summary']
 if n!=4440 or len(unique)!=5040 or set(cov)!={(t,r,d) for t in range(6) for r in FIT for d in (1,2,3)} or sum(cov.values())!=568320 or sm['calls']!=4440 or sm['consumed']!=4440 or sm['fresh_packets']!=568320 or sm['coverage_cells']!=72 or sm['nonidentical_steps']!=distinct or sm['extra_NN_per_L_packet']!=2 or sm['U_fresh_calls'] or sm['new_PA_DAC_applications'] or sm['seed']!=20261017961 or audit['model_parameters_added'] or not audit['original_U_unchanged'] or audit['target_access']:raise ValueError('Complete fresh observation source coverage/budget differs')
 if cov!=Counter({(v['TX'],v['RX'],v['day']):v['count'] for v in audit['coverage']}):raise ValueError('Explicit visible L TX/RX/day coverage recount differs')
 for name,value in [('first_operator_packets',first),('fresh_operator_packets',fresh),('view_gate_disagreement_packets',disagree)]:
  if dict(value)!=sm[name]:raise ValueError('Complete native gate packet recount differs')
 for c in curves:
  rows=per_epoch[c['epoch']]
  for key in ('first_prepa_raw_MSE','fresh_prepa_raw_MSE','fresh_first_prepa_MSE','normal_prepa_MSE','selected_raw_MSE','raw_power','first_prepa_power','fresh_prepa_power','selected_power'):c[key]=statistics.mean(z[key] for z in rows)
  c['fresh_first_nonidentical_fraction']=sum(z['fresh_first_prepa_MSE']>0 for z in rows)/222
 return dict(calls=n,consumed=n,fresh_packets=n*128,original_normal_packets=n*128,unique_L=len(unique),coverage_cells=len(cov),nonidentical_steps=distinct,construct_audit_seconds=sm['construct_audit_seconds'],first_operator_packets=dict(first),fresh_operator_packets=dict(fresh),view_gate_disagreement_packets=dict(disagree),eligible_packets=dict(eligible),native_probability_range={k:[min(v),max(v)] for k,v in prob.items()},U_fresh_calls=0,new_PA_DAC_applications=0,coverage=[dict(TX=t,RX=r,day=d,count=v) for (t,r,d),v in sorted(cov.items())])

def stratify(data):
 strata=[]
 for v in VIEWS:
  r=data['RAW'][v];y=r['labels'];groups=defaultdict(list)
  for i,(pid,tx) in enumerate(zip(r['physical_ids'],y)):
   groups['TX',str(tx)].append(i);groups['day',str(pid[2])].append(i)
   if v!='clean':
    ph=r['physics'][i]
    for dim,k in [('state','state'),('lock','lock'),('AGC_limited','agc_limited'),('kernel_length','residual_kernel_length'),('boundary','boundary')]:groups[dim,str(ph[k])].append(i)
    for dim,k,edges in [('quality','quality_snr_db',(0,5,10,20,30)),('added_noise_SNR','signal_to_added_noise_db',(0,5,10,20,30)),('abs_CFO','output_frequency_hz',(10,100,1000))]:
     x=abs(ph[k]) if dim=='abs_CFO' else ph[k];j=next((j for j,z in enumerate(edges) if x<z),len(edges));bn=('<'+str(edges[0])) if j==0 else ('>='+str(edges[-1])) if j==len(edges) else '['+str(edges[j-1])+','+str(edges[j])+')';groups[dim,bn].append(i)
  for (dim,value),ix in groups.items():
   for a in ARMS:
    z=data[a][v];strata.append(dict(arm=a,scene=v,dimension=dim,value=value,**metrics([y[i] for i in ix],[z['predictions'][i] for i in ix]),mean_margin_over30=statistics.mean(z['margin_over30'][i] for i in ix)))
 return strata

def analyze(root):
 root=Path(root)
 if (root/'failure.json').exists() or not (root/'completion.json').exists():raise ValueError('Full frozen matrix predictions first,then independent scoring')
 records={};scores={};data={};health={};sequences={};curves=[];bases={a:root/(a+'-RX1-s'+str(SEED))/'source' for a in ARMS}
 for a,b in bases.items():
  records[a],c,health[a],sequences[a]=training(b,a);scores[a],data[a]=predictions(b);curves+=c
 for a in ARMS[1:]:
  if read(bases['RAW']/'source_contract.json')!=read(bases[a]/'source_contract.json') or read(bases['RAW']/'actual_channel.json')!=read(bases[a]/'actual_channel.json'):raise ValueError('Identical matched role IDs and actual channel settings required')
 for c in (read(p/'actual_channel.json') for p in bases.values()):
  if not c['EQ_off_G_one'] or not c['formal_test_unchanged'] or set(c['actual_configs'])!=set(VIEWS[1:]):raise ValueError('Original six practical required')
  for z in c['actual_configs'].values():
   if (z['fs_hz'],z['processing_route'],z['mode'],z['equalization_enabled'],z['reference_snr_db'],z['boundary'])!=(25000000,'residual','post_sync',False,30,'reflect') or not all(z[k] for k in ('quality_adaptation_enabled','iq_compensation_enabled','phase_tracking_enabled')):raise ValueError('Unchanged receiver settings required')
 for v in VIEWS:
  for a in ARMS[1:]:
   for k in ('physical_ids','labels'):
    if data['RAW'][v][k]!=data[a][v][k]:raise ValueError('Exact paired readout IDs/labels required')
   if v!='clean' and data['RAW'][v]['physics']!=data[a][v]['physics']:raise ValueError('Fixed eval propagation/states/compensation differ')
 fixed=[i for i,(y,p) in enumerate(zip(data['RAW']['clean']['labels'],data['RAW']['clean']['predictions'])) if y==p];retention={};paired=[]
 for v in VIEWS:
  y=data['RAW'][v]['labels'];retention[v]={a:metrics([y[i] for i in fixed],[data[a][v]['predictions'][i] for i in fixed]) for a in ARMS}
  for ref in ('RAW','REPLAY'):
   p=data['REFRESH'][v]['predictions'];q=data[ref][v]['predictions'];paired.append(dict(reference=ref,scene=v,count=1260,repairs=sum(t==x and t!=z for t,x,z in zip(y,p,q)),breaks=sum(t!=x and t==z for t,x,z in zip(y,p,q)),net_correct=scores['REFRESH'][v]['correct']-scores[ref][v]['correct']))
 checks={};comparisons={};eps=1e-10
 for ref in ('RAW','REPLAY'):
  delta={v:100*(scores['REFRESH'][v]['accuracy']-scores[ref][v]['accuracy']) for v in VIEWS};f1=100*(scores['REFRESH']['clean']['macro_F1']-scores[ref]['clean']['macro_F1']);dv=100*(records['REFRESH']['seen_V']['source_val_accuracy']-records[ref]['seen_V']['source_val_accuracy']);checks.update({ref+'_clean':delta['clean']>=1.-eps,ref+'_macroF1':f1>=-eps,ref+'_seenV':dv>=-.5-eps});comparisons[ref]=dict(all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,seenV_gain_pp=dv,urban_mean_gain_pp=sum(delta[v] for v in VIEWS[4:])/3)
 checks['healthy']=all(r['final_L_CE']<=.5 for r in records.values());online=read(root/'result.json')['qualification']
 if checks!=online['checks'] or all(checks.values())!=online['qualified']:raise ValueError('Independent fixed qualification differs')
 for ref,c in comparisons.items():
  for v in VIEWS:near(c['all_view_gain_pp'][v],online['comparisons'][ref]['all_view_gain_pp'][v],1e-10)
  for k in ('clean_macroF1_gain_pp','seenV_gain_pp','urban_mean_gain_pp'):near(c[k],online['comparisons'][ref][k],1e-10)
 same_seq={a:sequences['RAW']==sequences[a] for a in ARMS[1:]};same_L={a:[z for z in sequences['RAW'] if z['role']=='L']==[z for z in sequences[a] if z['role']=='L'] for a in ARMS[1:]};same_U={a:[z for z in sequences['RAW'] if z['role']=='U']==[z for z in sequences[a] if z['role']=='U'] for a in ARMS[1:]}
 # Compare saved gate/scalar records,not IQ re-generation or a target query.
 shadows={};ignore={'arm','selected_raw_MSE','selected_first_prepa_MSE','selected_power'}
 for a in ARMS[1:]:
  shadows[a]=all({k:v for k,v in x.items() if k not in ignore}=={k:v for k,v in y.items() if k not in ignore} for x,y in zip(lines(bases['RAW']/'refresh_steps.jsonl.gz'),lines(bases[a]/'refresh_steps.jsonl.gz')))
 for a in ARMS:
  for v in VIEWS:scores[a][v].update(mean_margin_over30=statistics.mean(data[a][v]['margin_over30']),mean_CE=data[a][v]['mean_CE'])
 full_curve_summary={}
 for a in ARMS:
  q=[z for z in curves if z['arm']==a];last=q[-5:];full_curve_summary[a]=dict(final_epoch=q[-1],last5_total_L_min=min(z['total_L'] for z in last),last5_total_L_max=max(z['total_L'] for z in last),seenV_curve_unit='percent',last5_seenV_accuracy_min=min(z['seenV_accuracy'] for z in last),last5_seenV_accuracy_max=max(z['seenV_accuracy'] for z in last),last5_seenV_first_to_last_pp=last[-1]['seenV_accuracy']-last[0]['seenV_accuracy'],best_seenV_epoch_diagnostic_only=max(q,key=lambda z:z['seenV_accuracy'])['epoch'])
 divergence={}
 for ref in ('RAW','REPLAY'):
  a=list(lines(bases[ref]/'satmix_steps.jsonl.gz'));b=list(lines(bases['REFRESH']/'satmix_steps.jsonl.gz'));ga=list(lines(bases[ref]/'step_metrics.jsonl.gz'));gb=list(lines(bases['REFRESH']/'step_metrics.jsonl.gz'))
  first=lambda pred:next((i for i,(x,y) in enumerate(zip(a,b),1) if pred(x,y)),None)
  divergence[ref]=dict(first_extra_CE_difference_step=first(lambda x,y:x['extra_pair_CE']!=y['extra_pair_CE']),first_total_L_difference_step=first(lambda x,y:x['total_L']!=y['total_L']),first_gradient_preclip_difference_step=next((i for i,(x,y) in enumerate(zip(ga,gb),1) if x['metrics']['train/grad_before_clip']!=y['metrics']['train/grad_before_clip']),None))
 summary=dict(full_curve_summary=full_curve_summary,first_observed_training_divergence=divergence,status='ANALYZED',execution='VERIFIED',run_id=RUN,source_only=True,model_seeds=[SEED],held_RX=1,scores=scores,comparisons=comparisons,checks=checks,qualified=all(checks.values()),fixed_RAW_clean_correct_count=len(fixed),retention_on_fixed_RAW_clean_correct=retention,paired=paired,training=health,final_L_CE={a:r['final_L_CE'] for a,r in records.items()},seen_V={a:r['seen_V'] for a,r in records.items()},full_LU_sequences_exact=same_seq,full_L_sequences_exact=same_L,full_U_sequences_exact=same_U,saved_native_first_and_fresh_shadow_stats_exact=shadows,all_six_physics_states_parameters_exact=True,new_model_epochs=60,new_optimizer_updates=13320,new_NN=sum(h['NN'] for h in health.values())+2,dispatcher_realCP_smoke_NN=2,head_fit_updates=0,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,limits=['One exposed source RX1/modelseed;no formal target or crossseed evidence','No As training/new physical marginal operator or Ag+As interaction','Full saved visible-L IQ scalars/gates/coverage independently recounted;no IQ inverse reconstruction','No U labels decoded or extra U/teacher views','Budget exhaustion is not convergence'])
 write(root/'analysis_summary.json',summary);write(root/'source_strata.json',stratify(data));write(root/'training_curves.json',curves);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();s=analyze(a.input);print(json.dumps(dict(status=s['status'],qualified=s['qualified'],comparisons=s['comparisons'],new_NN=s['new_NN'])))
