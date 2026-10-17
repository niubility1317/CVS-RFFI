"""Complete frozen source-only E20 audit; no IQ/checkpoint/model or U truth access."""
import argparse,csv,gzip,hashlib,importlib.util,json,math,random,re,statistics,time
from collections import Counter
from pathlib import Path
spec=importlib.util.spec_from_file_location('verified_source_reader',Path(__file__).resolve().parents[1]/'cvs_as_timing20/analyze.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
ARMS=('RAW','IID','SCENE_GROUP','PROP_GROUP');SEED=2026092701;FIT=(1,3,4,8);VIEWS=old.VIEWS;old.ARMS=ARMS
read,write,lines,near,metrics,predictions,labeled_TX=(getattr(old,k) for k in ('read','write','lines','near','metrics','predictions','labeled_TX'))

def predictions(base):
 result={};data={};common=None
 for view in VIEWS:
  r=read(base/'source_readouts'/(view+'.json'));ids=[tuple(p) for p in r['physical_ids']];y=r['labels'];p=r['predictions']
  if len(ids)!=1260 or len(set(ids))!=1260 or set(z[1] for z in ids)!={6} or set(z[2] for z in ids)!={1,2,3} or len(y)!=1260 or Counter(y)!=Counter({k:210 for k in range(6)}) or len(p)!=1260 or r['head_fit_updates']:raise ValueError('Exact heldsourceL1260 role required')
  if [labeled_TX(z) for z in ids]!=y:raise ValueError('Visible heldsource label/physicalID mismatch')
  if common is None:common=(ids,y)
  elif common!=(ids,y):raise ValueError('View physicalIDs/truth pairing differs')
  m=metrics(y,p);near(m['accuracy'],r['accuracy']);near(m['macro_F1'],r['macro_F1'])
  if 'margin_over30' in r and (len(r['margin_over30'])!=1260 or any(not math.isfinite(x) for x in r['margin_over30'])):raise ValueError('Finite originalcos30 margins required')
  result[view]=m;data[view]=r
 return result,data

def stable_seed(seed,*parts):
 raw=json.dumps([int(seed),*parts],ensure_ascii=False,separators=(',',':'));return int.from_bytes(hashlib.sha256(raw.encode('utf8')).digest()[:8],'little')

def grouping(base,arm,sequence,steps,curves):
 L=[z['ids'] for z in sequence if z['role']=='L'];cells=Counter();physical=Counter();packets=set();groups=0;gen=Counter();per_step_groups=[]
 policy='IID' if arm=='RAW' else arm;on=arm!='RAW';names=('session','scene','state','lock','environment_seed','packet_seed','noise_stream_seed','kernel_energy','packet_residual_hz','packet_coarse_error_hz','packet_phase','packet_tracking_z')
 for i,(ids,z) in enumerate(zip(L,steps),1):
  epoch=(i-1)//222+1;namespace='train/epoch%d/step%d'%(epoch,i)
  if len(ids)!=128 or z['group_shadow_generated']!=128 or z['group_policy']!=policy or z['extra_input']!=arm or z['extra_As_packets']!=128*on or z['extra_BN_batch_updates'] or any(len(z['group_'+k])!=128 for k in names):raise ValueError('All128L/fixedpolicy/singleprocess coverage required')
  near(z['extra_raw_MSE'],z['shadow_raw_MSE'] if on else 0.,1e-10);same={}
  for j,pid in enumerate(ids):
   tx=labeled_TX(pid);eq,rx,day,sig,baseindex=pid;packet=(rx,day,eq,sig,baseindex);group=packet[:3];scene=z['group_scene'][j];state=z['group_state'][j];lock=z['group_lock'][j]
   if rx not in FIT or scene not in VIEWS[1:] or state not in ('LOS','PARTIAL','BLOCKED') or lock not in ('locked','degraded','unlocked'):raise ValueError('Legal source/state/scene required')
   expected_session='rx%d:day%d:eq%d'%group;seed=stable_seed(20261016101,namespace,scene,packet);envseed=stable_seed(20261016101,namespace,scene,'environment',group)
   if z['group_session'][j]!=expected_session or z['group_packet_seed'][j]!=seed or z['group_environment_seed'][j]!=envseed or z['group_noise_stream_seed'][j]!=stable_seed(seed,'noise'):raise ValueError('Exact independent packet/group namespace required')
   for k in ('kernel_energy','packet_residual_hz','packet_coarse_error_hz','packet_phase','packet_tracking_z'):
    if not math.isfinite(z['group_'+k][j]):raise ValueError('Finite physical/private state required')
   if z['group_kernel_energy'][j]<0:raise ValueError('Positive kernel energy required')
   values=(scene,state,z['group_kernel_energy'][j]);previous=same.setdefault(group,values)
   if policy!='IID' and previous[0]!=scene or policy=='PROP_GROUP' and previous!=values:raise ValueError('Registered sharedscene/sharedpropagation differs')
   cells[tx,rx,day,scene,state,lock]+=1;physical[tx,rx,day]+=1;packets.add(tuple(pid))
   for key,value in (('scene',scene),('state',state),('lock',lock)):gen[key,value]+=1
  if z['group_session_count']!=len(same):raise ValueError('Group cardinality differs')
  for k in ('scene','state','lock'):
   if Counter(z['group_'+k])!=Counter(z['group_'+k+'_counts']):raise ValueError('Whole group marginal recount differs')
  groups+=len(same);per_step_groups.append(len(same))
 audit=read(base/'group_coverage.json');actual=Counter()
 for z in audit['cells']:
  key=(z['TX'],z['RX'],z['day'],z['scene'],z['state'],z['lock'])
  if key in actual or not isinstance(z['count'],int) or z['count']<=0:raise ValueError('Unique positive savedgroup cell required')
  actual[key]=z['count']
 selected=read(base/'satellite_supervision_coverage.json');sel=Counter({(z['TX'],z['RX'],z['day'],z['scene'],z['state'],z['lock']):z['count'] for z in selected['cells']})
 if len(L)!=4440 or actual!=cells or len(packets)!=5040 or len(physical)!=72 or sum(cells.values())!=568320 or audit['generated']!=568320 or audit['consumed']!=on*568320 or audit['physical_initializations']!=1136640 or audit['physical_processes']!=568320 or not audit['allL_supervised'] or audit['confidence_gate'] or audit['U_truth_read'] or sel!=(cells if on else Counter()) or selected['total']!=on*568320 or not selected['all_legal_L_supervised'] or selected['confidence_gate']:raise ValueError('Full source physicalgroup/dose/permission recount differs')
 for c in curves:
  rr=steps[(c['epoch']-1)*222:c['epoch']*222];c.update(extra_raw_MSE=statistics.mean(z['extra_raw_MSE'] for z in rr),shadow_raw_MSE=statistics.mean(z['shadow_raw_MSE'] for z in rr),As_consumed=sum(z['extra_As_packets'] for z in rr),group_count_mean=statistics.mean(z['group_session_count'] for z in rr),generated=222*128)
 return dict(policy=policy,generated=568320,consumed=on*568320,physical_initializations=1136640,physical_processes=568320,physical_L=5040,TX_RX_day_cells=72,TX_RX_day_scene_cells=len({k[:4] for k in cells}),groups=groups,per_step_group_mean=statistics.mean(per_step_groups),dimensions={name:{k[1]:v for k,v in gen.items() if k[0]==name} for name in ('scene','state','lock')},all_state_cells=[dict(TX=k[0],RX=k[1],day=k[2],scene=k[3],state=k[4],lock=k[5],count=v) for k,v in sorted(cells.items())],new_U_views=0,scope='Saved native physical telemetries and exact named seeds;no IQ replay or simultaneous recording claim')

def training(base,arm):
 r=read(base/'satmix_result.json');done=read(base/'completion.json');init=read(base/'initialization.json');freeze=read(base/'source_frozen.json');contract=read(base/'source_contract.json');args=read(base/'actual_native_args.json');cfg=read(base/'resolved_config.json')
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access'] or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']:raise ValueError('Fixed scratch FINAL20 default-head provenance required')
 if args['seed']!=SEED or (args['epochs'],args['label_epochs'],args['pseudo_epochs'],args['label_smoothing'])!=(20,13,7,.01) or args['baseline_ckpt'] or args['teacher_ckpt'] or args['source_fold_held_rx']!=6 or contract['source_receivers']!=list(FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Exact seed, roles, original 13L+7U recipe required')
 if cfg['arm']!=arm or cfg['training_mechanism']['L_forwards_per_packet']!=8 or not args['use_aug'] or args['use_concat_sat_channel_aug'] or cfg['training_mechanism']['As']!=(arm!='RAW'):raise ValueError('Fixed matched8L and registered As consumption required')
 if r['actual_steps']!=4440 or r['L_forward_packets']!=4546560 or r['extra_L_forward_packets']!=1136640 or r['supcon_applied']!=3330 or r['As_training_observations']!=(0 if arm=='RAW' else 568320) or r['group_generated']!=568320 or r['group_steps']!=4440 or not r['default_head_joint'] or r['head_fit_updates'] or r['target_access'] or r['U_truth_read'] or r['checkpoint_sources'] or r['teacher_additive_calls']:raise ValueError('Actual complete matched source budget differs')
 steps=list(lines(base/'satmix_steps.jsonl.gz'));units=list(lines(base/'unit_steps.jsonl.gz'));native=list(lines(base/'step_metrics.jsonl.gz'));epochs=list(lines(base/'epoch_metrics.jsonl'));me=list(lines(base/'metrics_epoch.jsonl'))
 if [len(v) for v in (steps,units,native,epochs,me)]!=[4440,4440,4440,20,20]:raise ValueError('Complete native and branch step/epoch records required')
 for name in ('epoch_metrics.csv','metrics_epoch.csv'):
  with (base/name).open(encoding='utf-8-sig',newline='') as f:cr=list(csv.DictReader(f))
  if len(cr)!=20:raise ValueError('Complete CSV epoch records required')
 grad=[];curves=[];sup=0
 for i,(a,u,n) in enumerate(zip(steps,units,native),1):
  e=(i-1)//222+1
  if (a['step'],a['epoch'],u['step'],n['step'],n['epoch'])!=(i,e,i,i,e) or u['physical_clip_calls']!=i:raise ValueError('Full native physical Sinc updates differ')
  for k in ('total_L','raw_CE','ordinary_CE','s_slot_CE','raw_KL','ordinary_KL','s_slot_KL','supcon','supcon_weight','base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_raw_MSE','shadow_raw_MSE'):
   if not math.isfinite(a[k]):raise ValueError('Nonfinite active L field')
  near(a['base_total_L'],.5*(a['raw_CE']+a['raw_KL'])+.25*(a['ordinary_CE']+a['ordinary_KL'])+.25*(a['s_slot_CE']+a['s_slot_KL'])+a['supcon_weight']*a['supcon']);near(a['total_L'],a['base_total_L']+.25*a['extra_pair_loss']);near(a['extra_pair_loss'],a['extra_pair_CE']+a['extra_pair_KL'])
  if a['As_packets'] or a['sat_raw_MSE'] or a['extra_pair_weight']!=.25 or a['base_raw_total_weight']!=.75 or a['base_ordinary_weight']!=.25 or a['total_pair_weight']!=1.25 or not all(a[k] for k in ('all_L_supervised','same_IQ_pairs','extra_view_once','extra_same_IQ_pair')):raise ValueError('Fixed absolute clean supervision differs')
  sup+=a['supcon_weight']>0;m=n['metrics']
  for k in ('train/grad_before_clip','train/grad_clip_limit','train/grad_total','train/optimizer_step_applied','train/pseudo_truth_available'):
   if m.get(k) is None or not math.isfinite(m[k]):raise ValueError('Missing active update/permission telemetry')
  if m['train/pseudo_truth_available'] or m['train/optimizer_step_applied']!=1 or m.get('train/skipped_nonfinite_grad',0) or m.get('train/skipped_nonfinite_loss',0):raise ValueError('Hidden U truth or nonfinite/skipped update')
  grad.append(m['train/grad_before_clip'])
 if sup!=3330:raise ValueError('Original SupCon warm5/ramp10 differs')
 for e,row in enumerate(epochs,1):
  if int(row['epoch'])!=e or int(row['epochs'])!=20 or int(row['label_epochs'])!=13 or int(row['pseudo_epochs'])!=7 or float(row['train_pseudo_truth_available']) or float(row['train_skipped_nonfinite_grad']) or float(row['train_skipped_nonfinite_loss']) or float(row['train_optimizer_step_applied'])!=1:raise ValueError('Original complete epoch schedule/health differs')
  sub=steps[(e-1)*222:e*222];curves.append(dict(arm=arm,epoch=e,base_L=statistics.mean(a['base_total_L'] for a in sub),extra_CE=statistics.mean(a['extra_pair_CE'] for a in sub),extra_KL=statistics.mean(a['extra_pair_KL'] for a in sub),total_L=statistics.mean(a['total_L'] for a in sub),raw_CE=statistics.mean(a['raw_CE'] for a in sub),ordinary_CE=statistics.mean(a['ordinary_CE'] for a in sub),supcon=statistics.mean(a['supcon'] for a in sub),supcon_weight=statistics.mean(a['supcon_weight'] for a in sub),grad_before_clip=statistics.mean(grad[(e-1)*222:e*222]),train_tx_accuracy=row['train_tx_acc'],seenV_accuracy=row['val_tx_acc'],lr=row['lr'],epoch_seconds=row['epoch_time_s'],pseudo_selected=row['train_pseudo_selected'],pseudo_total=row['train_pseudo_total']))
 near(statistics.mean(a['base_total_L'] for a in steps[-222:]),r['final_base_L_objective']);near(statistics.mean(z['metrics']['train/loss_tx_labeled'] for z in native[-222:]),r['final_L_CE'])
 cost=sum(r[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets'))-2+r['held_forward_packets'];seq=list(lines(base/'training_sequence.jsonl.gz'))
 if len(seq)!=5994 or Counter(z['role'] for z in seq)!=Counter(L=4440,U=1554):raise ValueError('Complete L/U sequence required;never decode U TX')
 cov=grouping(base,arm,seq,steps,curves);text=(base.parents[1]/'logs'/(base.parent.name+'.stdout.log')).read_text(encoding='utf8',errors='replace')
 if re.findall(r'Traceback|CUDA out of memory|RuntimeError:|Killed|FloatingPointError:',text):raise ValueError('Critical complete source stdout error')
 return r,curves,dict(grouping=cov,steps=4440,epochs=20,NN=cost,teacher_inherited_eval_counter_deducted=2,held_default_head_NN=8820,L_NN=4546560,extra_L_NN=1136640,gradient_preclip_mean=statistics.mean(grad),gradient_preclip_max=max(grad),gradient_zero=sum(x==0 for x in grad),gradient_above5=sum(x>5 for x in grad),all_updates_finite=True,optimizer_updates=4440,parent_training_epochs=0,training_wall_seconds=r['elapsed_seconds'],stdout_lines=len(text.splitlines()),stdout_warning_lines=[z for z in text.splitlines() if re.search(r'(?i)warning|warn:',z)],stdout_nan_mentions=len(re.findall(r'(?i)nan',text))),seq

def qualify(scores,records):
 checks={};comp={};thresholds={'RAW':(0.,0.,5.,3.),'IID':(1.,0.,.5,1.),'SCENE_GROUP':(1.,0.,.5,1.)}
 for ref in ARMS[:-1]:
  delta={v:100*(scores['PROP_GROUP'][v]['accuracy']-scores[ref][v]['accuracy']) for v in VIEWS};f1=100*(scores['PROP_GROUP']['clean']['macro_F1']-scores[ref]['clean']['macro_F1']);seen=100*(records['PROP_GROUP']['seen_V']['source_val_accuracy']-records[ref]['seen_V']['source_val_accuracy']);urban=sum(delta[v] for v in VIEWS[4:])/3;t=thresholds[ref];part=dict(clean=delta['clean']>=t[0]-1e-10,F1=f1>=t[1]-1e-10,low=delta[VIEWS[-1]]>=t[2]-1e-10,urban=urban>=t[3]-1e-10,each_practical=all(delta[v]>=-.5-1e-10 for v in VIEWS[1:]),seenV=seen>=-.5-1e-10);checks.update({ref+'_'+k:v for k,v in part.items()});comp[ref]=dict(all_view_gain_pp=delta,clean_F1_gain_pp=f1,seenV_gain_pp=seen,urban_mean_gain_pp=urban)
 checks['healthy']=all(r['actual_steps']==4440 and r['L_forward_packets']==4546560 and r['extra_L_forward_packets']==1136640 and r['group_generated']==568320 and r['supcon_applied']==3330 and r['As_training_observations']==(0 if a=='RAW' else 568320) and r['final_base_L_objective']<=.5 and r['held_RX']==6 and r['fit_RX']==list(FIT) and not r['head_fit_updates'] and not r['target_access'] and not r['U_truth_read'] and not r['checkpoint_sources'] for a,r in records.items())
 return checks,comp

def analyze(root):
 start=time.perf_counter();root=Path(root)
 if (root/'failure.json').exists() or not (root/'completion.json').exists():raise ValueError('Whole frozen matrix and seven predictions/row before scoring')
 records={};scores={};data={};health={};seq={};curves=[];bases={a:root/(a+'-RX6-s'+str(SEED))/'source' for a in ARMS}
 for a,b in bases.items():records[a],c,health[a],seq[a]=training(b,a);scores[a],data[a]=predictions(b);curves+=c
 for a in ARMS[1:]:
  if read(bases['RAW']/'source_contract.json')!=read(bases[a]/'source_contract.json') or read(bases['RAW']/'actual_channel.json')!=read(bases[a]/'actual_channel.json'):raise ValueError('Matched role IDs/channel settings differ')
 for base in bases.values():
  channel=read(base/'actual_channel.json')
  if not channel['EQ_off_G_one'] or not channel['formal_test_unchanged'] or set(channel['actual_configs'])!=set(VIEWS[1:]):raise ValueError('Original six practical settings required')
  for c in channel['actual_configs'].values():
   if (c['fs_hz'],c['processing_route'],c['mode'],c['equalization_enabled'],c['reference_snr_db'],c['boundary'])!=(25000000,'residual','post_sync',False,30,'reflect') or not all(c[k] for k in ('quality_adaptation_enabled','iq_compensation_enabled','phase_tracking_enabled')):raise ValueError('Unchanged physical receiver required')
 for v in VIEWS:
  for a in ARMS[1:]:
   if any(data['RAW'][v][k]!=data[a][v][k] for k in ('physical_ids','labels')) or v!='clean' and data['RAW'][v]['physics']!=data[a][v]['physics']:raise ValueError('Paired source observation states/IDs differ')
 fixed=[i for i,(y,p) in enumerate(zip(data['RAW']['clean']['labels'],data['RAW']['clean']['predictions'])) if y==p];retention={};paired=[]
 for v in VIEWS:
  y=data['RAW'][v]['labels'];retention[v]={a:metrics([y[i] for i in fixed],[data[a][v]['predictions'][i] for i in fixed]) for a in ARMS}
  for ref in ARMS[:-1]:
   p=data['PROP_GROUP'][v]['predictions'];q=data[ref][v]['predictions'];paired.append(dict(reference=ref,scene=v,count=1260,repairs=sum(t==x and t!=z for t,x,z in zip(y,p,q)),breaks=sum(t!=x and t==z for t,x,z in zip(y,p,q)),net_correct=scores['PROP_GROUP'][v]['correct']-scores[ref][v]['correct']))
 checks,comparisons=qualify(scores,records);online=read(root/'result.json')['qualification']
 if checks!=online['checks'] or all(checks.values())!=online['qualified']:raise ValueError('Independent predeclared scientific judgment differs')
 for ref,c in comparisons.items():
  for v in VIEWS:near(c['all_view_gain_pp'][v],online['comparisons'][ref]['all_view_gain_pp'][v],1e-10)
  for k in ('clean_F1_gain_pp','seenV_gain_pp','urban_mean_gain_pp'):near(c[k],online['comparisons'][ref][k],1e-10)
 same_seq={a:seq['RAW']==seq[a] for a in ARMS[1:]};shadows={}
 for ref,other,keys in [('RAW','IID',('group_session','group_scene','group_state','group_lock','group_packet_seed','group_environment_seed','group_noise_stream_seed','group_kernel_energy','group_packet_residual_hz','group_packet_coarse_error_hz','group_packet_phase','group_packet_tracking_z','shadow_raw_MSE')),('SCENE_GROUP','PROP_GROUP',('group_session','group_scene','group_packet_seed','group_environment_seed','group_noise_stream_seed','group_packet_residual_hz','group_packet_coarse_error_hz','group_packet_phase','group_packet_tracking_z'))]:
  exact={k:True for k in keys}
  for x,y in zip(lines(bases[ref]/'satmix_steps.jsonl.gz'),lines(bases[other]/'satmix_steps.jsonl.gz')):
   for k in keys:exact[k]=exact[k] and x[k]==y[k]
  if not all(exact.values()):raise ValueError('Matched IIDshadow/privategrouped streams differ')
  shadows[ref+'_vs_'+other]=exact
 for a in ARMS:
  for v in VIEWS:scores[a][v].update(mean_margin_over30=statistics.mean(data[a][v]['margin_over30']),mean_CE=data[a][v]['mean_CE'])
 summary=dict(status='ANALYZED_SOURCE_ONLY',execution='VERIFIED',run_id=root.name,model_seeds=[SEED],held_RX=6,source_only=True,scores=scores,comparisons=comparisons,checks=checks,qualified=all(checks.values()),fixed_RAW_clean_correct_count=len(fixed),retention_on_fixed_RAW_clean_correct=retention,paired=paired,training=health,final_L_CE={a:r['final_L_CE'] for a,r in records.items()},final_base_L={a:r['final_base_L_objective'] for a,r in records.items()},seen_V={a:r['seen_V'] for a,r in records.items()},full_LU_sequences_exact=same_seq,saved_named_streams_exact=shadows,all_six_physics_states_parameters_exact=True,new_model_epochs=80,new_optimizer_updates=17760,new_training_and_source_readout_NN=sum(v['NN'] for v in health.values()),guard_source_NN=0,dispatcher_realCP_smoke_NN=2,previous_failed_attempt_realCP_smoke_NN=0,total_actual_remote_NN=sum(v['NN'] for v in health.values())+2,target_access=False,head_fit_updates=0,Ag_efficacy_proved=False,joint_goal_complete=False,budget_exhaustion_is_convergence=False,analysis_wall_seconds=time.perf_counter()-start,limits=['One held source RX6, one new model seed, no formal target/generalization claim','As grouping only;no new Ag or Ag×As factorial;all practical readouts unchanged','Hard decisions/accuracy/F1 fully recounted;CE/cos30 margins are saved runtime telemetry without full logits','Coverage and physical named streams fully recounted without IQ regeneration;stream equality is not processed-IQ equality','Teacher eval counter includes two copied student smoke entries;deduct2 once per row','No U truth decoded;U IDs compared only for matched order'])
 write(root/'analysis_summary.json',summary);write(root/'training_curves.json',curves);write(root/'source_strata.json',old.stratify(data));return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();s=analyze(a.input);print(json.dumps(dict(status=s['status'],qualified=s['qualified'],comparisons=s['comparisons'],total_actual_remote_NN=s['total_actual_remote_NN'])))
