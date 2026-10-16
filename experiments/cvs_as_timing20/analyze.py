"""Independent complete source recount;stdlib only,no IQ/CP/model/target."""
import argparse,csv,gzip,json,math,random,re,statistics
from collections import Counter,defaultdict
from pathlib import Path
RUN='20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01';ARMS=('RAW','ALL','EARLY7','LATE7');SEED=2026092701;FIT=(3,4,6,8)
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
  if [labeled_TX(z) for z in ids]!=y:raise ValueError('Visible heldsource label/physicalID mismatch')
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
 if cfg['arm']!=arm or cfg['training_mechanism']['L_forwards_per_packet']!=8 or not args['use_aug'] or args['use_concat_sat_channel_aug']:raise ValueError('Fixed registered8L/ordinary/As-timing training required')
 if r['actual_steps']!=4440 or r['L_forward_packets']!=4546560 or r['extra_L_forward_packets']!=1136640 or r['supcon_applied']!=3330 or r['As_training_observations']!=expected_packets(arm) or r['As_generated_packets']!=568320 or r['timing_steps']!=4440 or not r['default_head_joint'] or r['head_fit_updates'] or r['target_access'] or r['U_truth_read'] or r['checkpoint_sources'] or r['teacher_additive_calls']:raise ValueError('Actual8L/defaulthead/hiddenU source budget differs')
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
  if a['As_packets'] or a['sat_raw_MSE'] or a['extra_As_packets']!=128*active(arm,e) or a['As_shadow_generated']!=128 or a['As_active']!=active(arm,e) or a['extra_pair_weight']!=.25 or a['base_raw_total_weight']!=.75 or a['base_ordinary_weight']!=.25 or a['total_pair_weight']!=1.25 or not all(a[k] for k in ('all_L_supervised','same_IQ_pairs','extra_view_once','extra_same_IQ_pair')):raise ValueError('Fixed absolute common supervision differs')
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
 if cost<4546560+8820:raise ValueError('Actual complete8L network-forward accounting missing')
 seq=list(lines(base/'training_sequence.jsonl.gz'))
 if len(seq)!=5994 or Counter(z['role'] for z in seq)!=Counter(L=4440,U=1554):raise ValueError('Complete original L/U physical sequences required')
 timing=timing_recount(base,arm,seq,steps,curves)
 text=(base.parents[1]/'logs'/(base.parent.name+'.stdout.log')).read_text(encoding='utf8',errors='replace')
 if re.findall(r'Traceback|CUDA out of memory|RuntimeError:|Killed|FloatingPointError:',text):raise ValueError('Critical full source stdout error')
 return r,curves,dict(timing=timing,stdout_lines=len(text.splitlines()),stdout_warning_lines=[z for z in text.splitlines() if re.search(r'(?i)warning|warn:',z)],stdout_nan_mentions=len(re.findall(r'(?i)nan',text)),stdout_epoch_markers=len(re.findall(r'\[EPOCH-END\]',text)),steps=4440,epochs=20,NN=cost,gradient_preclip_mean=statistics.mean(grad),gradient_preclip_max=max(grad),gradient_zero=sum(x==0 for x in grad),gradient_above5=sum(x>5 for x in grad),native_disabled_null_fields=len(native_nulls),all_active_updates_finite=True,optimizer_updates=4440,L_NN=4546560,parent_epochs=0),seq

def labeled_TX(pid):
 # Original source-cache flattened layout only for saved visible-L records.
 # Validate RX/day/sig here and every TX count/pair count against saved native audits.
 eq,rx,day,sig,base=pid;within=base%15000;rx_index=(within//1000)//3
 if eq!=1 or base<0 or base>=90000 or (1,3,4,6,8)[rx_index]!=rx or within//1000%3+1!=day or within%1000!=sig or sig not in range(70):raise ValueError('Saved visible-L metadata layout differs')
 return base//15000

def active(arm,epoch):
 if arm not in ARMS or not 1<=epoch<=20:raise ValueError('Fixed arm/epoch required')
 return arm=='ALL' or arm=='EARLY7' and epoch<=7 or arm=='LATE7' and epoch>=14
def expected_packets(arm):return {'RAW':0,'ALL':568320,'EARLY7':198912,'LATE7':198912}[arm]
def counter_cells(rows):
 out=Counter()
 for z in rows:
  if z['TX'] not in range(6) or z['RX'] not in FIT or z['day'] not in (1,2,3) or z['scene'] not in VIEWS[1:] or z['state'] not in ('LOS','PARTIAL','BLOCKED') or z['lock'] not in ('locked','degraded','unlocked') or not isinstance(z['count'],int) or z['count']<=0:raise ValueError('Legal visibleL source-state coverage required')
  key=(z['TX'],z['RX'],z['day'],z['scene'],z['state'],z['lock'])
  if key in out:raise ValueError('Duplicate joint source coverage cell')
  out[key]=z['count']
 return out
def timing_recount(base,arm,seq,steps,curves):
 L=[x['ids'] for x in seq if x['role']=='L'];cov=Counter();used=Counter();unique=set();uq=set();gen=Counter();con=Counter();n=0;consumed=0;per_epoch=[]
 for i,(ids,z) in enumerate(zip(L,steps),1):
  epoch=(i-1)//222+1;on=active(arm,epoch)
  if len(ids)!=128 or z['As_active']!=on or z['extra_As_packets']!=128*on or z['As_shadow_generated']!=128:raise ValueError('Fixed wholebatch As activation required')
  if not math.isfinite(z['shadow_raw_MSE']) or z['shadow_raw_MSE']<0 or z['extra_BN_batch_updates'] or z['extra_input']!=('As(raw)' if on else 'raw'):raise ValueError('Real fixed extraIQ/zeroBatchNorm required')
  near(z['extra_raw_MSE'],z['shadow_raw_MSE'] if on else 0.,1e-10)
  for pid in ids:
   tx=labeled_TX(pid);key=(tx,pid[1],pid[2]);cov[key]+=1;unique.add(tuple(pid))
   if on:used[key]+=1;uq.add(tuple(pid))
  for name,allowed in [('scene',VIEWS[1:]),('state',('LOS','PARTIAL','BLOCKED')),('lock',('locked','degraded','unlocked'))]:
   counts=z['shadow_'+name+'_counts']
   if not set(counts)<=set(allowed) or sum(counts.values())!=128 or any(not isinstance(v,int) or v<=0 for v in counts.values()):raise ValueError('Complete physical state counts required')
   for k,v in counts.items():
    gen[name,k]+=v
    if on:con[name,k]+=v
  consumed+=128*on;n+=1
 audit=read(base/'timing_audit.json');generated=counter_cells(audit['generated_coverage']);observed=read(base/'satellite_supervision_coverage.json');selected=counter_cells(observed['cells']);planned=[e for e in range(1,21) if active(arm,e)]
 if n!=4440 or len(unique)!=5040 or len(cov)!=72 or sum(cov.values())!=568320 or consumed!=expected_packets(arm) or audit['steps']!=4440 or audit['generated']!=568320 or audit['consumed']!=consumed or audit['timing']['active_epochs']!=planned or audit['new_parameters'] or audit['extra_U'] or audit['head_fit_updates'] or audit['target_access'] or observed['total']!=consumed or not observed['all_legal_L_supervised'] or observed['confidence_gate']:raise ValueError('Complete source timing/dose/permission counters differ')
 if sum(generated.values())!=568320 or sum(selected.values())!=consumed or Counter({k:sum(v for p,v in generated.items() if p[:3]==k) for k in cov})!=cov or Counter({k:sum(v for p,v in selected.items() if p[:3]==k) for k in used})!=used:raise ValueError('ActualL physical source coverage differs')
 for name,pos in [('scene',3),('state',4),('lock',5)]:
  for counts,truth in ((generated,gen),(selected,con)):
   ag=Counter()
   for k,v in counts.items():ag[k[pos]]+=v
   if ag!=Counter({k[1]:v for k,v in truth.items() if k[0]==name}):raise ValueError('Full saved state/scene/lock aggregation differs')
 for c in curves:
  rr=steps[(c['epoch']-1)*222:c['epoch']*222];c.update(As_active=active(arm,c['epoch']),As_consumed=sum(z['extra_As_packets'] for z in rr),As_generated=sum(z['As_shadow_generated'] for z in rr),extra_raw_MSE=statistics.mean(z['extra_raw_MSE'] for z in rr),shadow_raw_MSE=statistics.mean(z['shadow_raw_MSE'] for z in rr))
 return dict(calls=n,generated=568320,consumed=consumed,active_epochs=planned,generated_unique_L=len(unique),consumed_unique_L=len(uq),generated_TX_RX_day_cells=len(cov),consumed_TX_RX_day_cells=len(used),generated_TX_RX_day_scene_cells=len({p[:4] for p in generated}),consumed_TX_RX_day_scene_cells=len({p[:4] for p in selected}),generated_dimensions={name:{k[1]:v for k,v in gen.items() if k[0]==name} for name in ('scene','state','lock')},consumed_dimensions={name:{k[1]:v for k,v in con.items() if k[0]==name} for name in ('scene','state','lock')},generated_physical_counts=[dict(TX=k[0],RX=k[1],day=k[2],count=v) for k,v in sorted(cov.items())],consumed_physical_counts=[dict(TX=k[0],RX=k[1],day=k[2],count=v) for k,v in sorted(used.items())],limits='Joint states are saved native telemetry;physicalTX/RX/day and scene/state/lock marginals independently recounted without IQ replay')

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

def qualification(scores,records):
 checks={};comparisons={};eps=1e-10;thresholds={'RAW':(0.,0.,5.,3.),'ALL':(1.,0.,-.5,-.5),'EARLY7':(1.,0.,1.,1.)}
 for ref in ('RAW','ALL','EARLY7'):
  delta={v:100*(scores['LATE7'][v]['accuracy']-scores[ref][v]['accuracy']) for v in VIEWS};f1=100*(scores['LATE7']['clean']['macro_F1']-scores[ref]['clean']['macro_F1']);dv=100*(records['LATE7']['seen_V']['source_val_accuracy']-records[ref]['seen_V']['source_val_accuracy']);urban=sum(delta[v] for v in VIEWS[4:])/3;t=thresholds[ref];part=dict(clean=delta['clean']>=t[0]-eps,clean_F1=f1>=t[1]-eps,low=delta[VIEWS[-1]]>=t[2]-eps,urban_mean=urban>=t[3]-eps,each_practical=all(delta[v]>=-.5-eps for v in VIEWS[1:]),seenV=dv>=-.5-eps);checks.update({ref+'_'+k:v for k,v in part.items()});comparisons[ref]=dict(conditions=part,all_view_gain_pp=delta,clean_macroF1_gain_pp=f1,seenV_gain_pp=dv,urban_mean_gain_pp=urban)
 checks['healthy']=all(r['actual_steps']==4440 and r['L_forward_packets']==4546560 and r['extra_L_forward_packets']==1136640 and r['supcon_applied']==3330 and r['final_base_L_objective']<=.5 and r['As_generated_packets']==568320 and r['As_training_observations']==expected_packets(arm) and r['timing_steps']==4440 and not r['head_fit_updates'] and not r['target_access'] and not r['U_truth_read'] and not r['checkpoint_sources'] for arm,r in records.items())
 return checks,comparisons

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
  for ref in ('RAW','ALL','EARLY7'):
   p=data['LATE7'][v]['predictions'];q=data[ref][v]['predictions'];paired.append(dict(reference=ref,scene=v,count=1260,repairs=sum(t==x and t!=z for t,x,z in zip(y,p,q)),breaks=sum(t!=x and t==z for t,x,z in zip(y,p,q)),net_correct=scores['LATE7'][v]['correct']-scores[ref][v]['correct']))
 checks,comparisons=qualification(scores,records);online=read(root/'result.json')['qualification']
 if checks!=online['checks'] or all(checks.values())!=online['qualified']:raise ValueError('Independent fixed qualification differs')
 for ref,c in comparisons.items():
  for v in VIEWS:near(c['all_view_gain_pp'][v],online['comparisons'][ref]['all_view_gain_pp'][v],1e-10)
  for k in ('clean_macroF1_gain_pp','seenV_gain_pp','urban_mean_gain_pp'):near(c[k],online['comparisons'][ref][k],1e-10)
 same_seq={a:sequences['RAW']==sequences[a] for a in ARMS[1:]};same_L={a:[z for z in sequences['RAW'] if z['role']=='L']==[z for z in sequences[a] if z['role']=='L'] for a in ARMS[1:]};same_U={a:[z for z in sequences['RAW'] if z['role']=='U']==[z for z in sequences[a] if z['role']=='U'] for a in ARMS[1:]}
 # Same originalphysical shadow stream in all four rows;active use differs.
 shadows={};shadowkeys=('As_shadow_generated','shadow_raw_MSE','shadow_scene_counts','shadow_state_counts','shadow_lock_counts')
 for a in ARMS[1:]:
  shadows[a]=all({k:x[k] for k in shadowkeys}=={k:y[k] for k in shadowkeys} for x,y in zip(lines(bases['RAW']/'satmix_steps.jsonl.gz'),lines(bases[a]/'satmix_steps.jsonl.gz')))
 for a in ARMS:
  for v in VIEWS:scores[a][v].update(mean_margin_over30=statistics.mean(data[a][v]['margin_over30']),mean_CE=data[a][v]['mean_CE'])
 full_curve_summary={}
 for a in ARMS:
  q=[z for z in curves if z['arm']==a];last=q[-5:];full_curve_summary[a]=dict(final_epoch=q[-1],last5_total_L_min=min(z['total_L'] for z in last),last5_total_L_max=max(z['total_L'] for z in last),seenV_curve_unit='percent',last5_seenV_accuracy_min=min(z['seenV_accuracy'] for z in last),last5_seenV_accuracy_max=max(z['seenV_accuracy'] for z in last),last5_seenV_first_to_last_pp=last[-1]['seenV_accuracy']-last[0]['seenV_accuracy'],best_seenV_epoch_diagnostic_only=max(q,key=lambda z:z['seenV_accuracy'])['epoch'])
 divergence={}
 for ref in ('RAW','ALL','EARLY7'):
  a=list(lines(bases[ref]/'satmix_steps.jsonl.gz'));b=list(lines(bases['LATE7']/'satmix_steps.jsonl.gz'));ga=list(lines(bases[ref]/'step_metrics.jsonl.gz'));gb=list(lines(bases['LATE7']/'step_metrics.jsonl.gz'))
  first=lambda pred:next((i for i,(x,y) in enumerate(zip(a,b),1) if pred(x,y)),None)
  divergence[ref]=dict(first_extra_CE_difference_step=first(lambda x,y:x['extra_pair_CE']!=y['extra_pair_CE']),first_total_L_difference_step=first(lambda x,y:x['total_L']!=y['total_L']),first_gradient_preclip_difference_step=next((i for i,(x,y) in enumerate(zip(ga,gb),1) if x['metrics']['train/grad_before_clip']!=y['metrics']['train/grad_before_clip']),None))
 summary=dict(full_curve_summary=full_curve_summary,first_observed_training_divergence=divergence,status='ANALYZED',execution='VERIFIED',run_id=RUN,source_only=True,model_seeds=[SEED],held_RX=1,scores=scores,comparisons=comparisons,checks=checks,qualified=all(checks.values()),fixed_RAW_clean_correct_count=len(fixed),retention_on_fixed_RAW_clean_correct=retention,paired=paired,training=health,final_L_CE={a:r['final_L_CE'] for a,r in records.items()},seen_V={a:r['seen_V'] for a,r in records.items()},full_LU_sequences_exact=same_seq,full_L_sequences_exact=same_L,full_U_sequences_exact=same_U,saved_full_practical_shadow_stats_exact=shadows,all_six_physics_states_parameters_exact=True,new_model_epochs=80,new_optimizer_updates=17760,new_NN=sum(h['NN'] for h in health.values())+2,dispatcher_realCP_smoke_NN=2,head_fit_updates=0,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,limits=['One exposed source RX1/modelseed;no formal target or crossseed evidence','As timing policy only;no newAg or Ag+As interaction','Full savedL timing/scalars/source coverage recounted;no physicalIQ replay or exact same early-late multiset claim','No U labels decoded or extra U/teacher views','CE/margins are saved runtime scalars;hard decisions/accuracy/F1 fully recounted,not CE from absent logits','Timepolicy includes LR/SSLcalendar;not isolated identity-first causality','Budget exhaustion is not convergence'])
 write(root/'analysis_summary.json',summary);write(root/'source_strata.json',stratify(data));write(root/'training_curves.json',curves);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();s=analyze(a.input);print(json.dumps(dict(status=s['status'],qualified=s['qualified'],comparisons=s['comparisons'],new_NN=s['new_NN'])))
