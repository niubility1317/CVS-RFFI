"""Independent complete source recount;stdlib only,no IQ/CP/model/target."""
import argparse,csv,gzip,json,math,random,re,statistics
from collections import Counter,defaultdict
from pathlib import Path
RUN='20261017-phase1-FcrossRX-timegrid-sourceRX1-e20-s2026092701-m1-r01';ROW='TIMEGRID-RX1-s2026092701';SEED=2026092701;FIT=(3,4,6,8)
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
 r=read(base/'satmix_result.json');done=read(base/'completion.json');init=read(base/'initialization.json');freeze=read(base/'source_frozen.json');contract=read(base/'source_contract.json');args=read(base/'actual_native_args.json')
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access'] or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']:raise ValueError('Verified fixed scratch FINAL20 defaulthead provenance required')
 if args['seed']!=SEED or (args['epochs'],args['label_epochs'],args['pseudo_epochs'],args['label_smoothing'])!=(20,13,7,.01) or args['baseline_ckpt'] or args['teacher_ckpt'] or args['source_fold_held_rx']!=1 or contract['source_receivers']!=list(FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Exact seed and source data contract required')
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
  near(a['total_L'],a['base_total_L']+.25*a['extra_pair_loss']);near(a['extra_pair_weighted'],.25*a['extra_pair_loss']);near(a['extra_pair_loss'],a['extra_pair_CE']+a['extra_pair_KL'])
  if a['As_packets'] or a['extra_As_packets'] or a['extra_pair_weight']!=.25 or a['base_raw_total_weight']!=.75 or a['base_ordinary_weight']!=.25 or a['total_pair_weight']!=1.25 or not all(a[k] for k in ('all_L_supervised','same_IQ_pairs','extra_view_once','extra_same_IQ_pair')):raise ValueError('Fixed absolute common supervision differs')
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
 near(statistics.mean(a['base_total_L'] for a in steps[-222:]),r['final_base_L_objective'])
 cost=sum(r[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets'))-2+r['held_forward_packets']
 if cost!=5823318:raise ValueError('Actual complete8L network-forward accounting differs')
 return r,curves,dict(steps=4440,epochs=20,NN=cost,gradient_preclip_mean=statistics.mean(grad),gradient_preclip_max=max(grad),gradient_zero=sum(x==0 for x in grad),gradient_above5=sum(x>5 for x in grad),native_disabled_null_fields=len(native_nulls),all_active_updates_finite=True,optimizer_updates=4440,parent_epochs=0)

def analyze(root,reference):
 root=Path(root);reference=Path(reference);base=root/ROW/'source'
 if (root/'failure.json').exists() or not (root/'completion.json').exists():raise ValueError('Complete fixed prediction before offline scoring')
 new,ncurve,nhealth=training(base,'TIMEGRID');old,ocurve,ohealth=training(reference,'D_raw')
 ns,nd=predictions(base);os,od=predictions(reference)
 ignore={'native_role_comparison','checkpoint_effective_training_contract','carrier_contract_ref'}
 if {k:v for k,v in read(base/'source_contract.json').items() if k not in ignore}!={k:v for k,v in read(reference/'source_contract.json').items() if k not in ignore}:raise ValueError('Actual scientific data contract/physicalroles differ')
 if nd['clean']['physical_ids']!=od['clean']['physical_ids'] or nd['clean']['labels']!=od['clean']['labels']:raise ValueError('Reference evaluation physicalID/labels differ')
 seq=list(lines(base/'training_sequence.jsonl.gz'));refseq=list(lines(reference/'training_sequence.jsonl.gz'))
 if seq!=refseq or len(seq)!=5994 or Counter(r['role'] for r in seq)!=Counter(L=4440,U=1554):raise ValueError('Complete unchanged primary L/U sequences required')
 L=[r['ids'] for r in seq if r['role']=='L'];coverage=Counter();unique=set();sessions={};stepcounts=Counter();per_epoch=defaultdict(list);max_change=0.;power_min=math.inf;power_max=0.;n=0
 for r in lines(base/'timegrid_views.jsonl.gz'):
  step=r['step'];e=r['epoch'];pid=tuple(r['physical_id']);tx=r['TX'];key=(e,pid[0],pid[1],pid[2]);expected=(1+random.Random(20261017701+e*4096+pid[1]*64+pid[2]*8+pid[0]).random())/8192
  if step not in range(1,4441) or e!=(step-1)//222+1 or pid[0]!=1 or pid[1] not in FIT or pid[2] not in (1,2,3) or tx not in range(6) or r['session']!=list(key) or r['padding'] or r['normalization']:raise ValueError('Timegrid source/session/step differs')
  j=stepcounts[step]
  if j>=128 or list(pid)!=L[step-1][j]:raise ValueError('Timegrid actual primaryphysical sequence differs')
  near(expected,r['rho'],1e-14);near(127.5*expected,r['max_displacement_samples'],1e-12)
  if sessions.setdefault(key,expected)!=expected or r['relative_L2']<0 or r['relative_L2']>255/4096+1e-5 or not ((r['power_ratio']==0 and r['relative_L2']==0) or (1-255/4096)**2-1e-5<=r['power_ratio']<=(1+255/4096)**2+1e-5):raise ValueError('Fixed numerical bound/session differs')
  coverage[tx,pid[1],pid[2]]+=1;unique.add(pid);stepcounts[step]+=1;per_epoch[e].append(r['relative_L2']);max_change=max(max_change,r['relative_L2']);power_min=min(power_min,r['power_ratio']);power_max=max(power_max,r['power_ratio']);n+=1
 if n!=568320 or len(unique)!=5040 or len(sessions)!=240 or set(coverage)!={(t,r,day) for t in range(6) for r in FIT for day in (1,2,3)} or set(stepcounts)!=set(range(1,4441)) or set(stepcounts.values())!={128}:raise ValueError('Complete grid568320/5040/240/72 coverage required')
 audit=read(base/'timegrid_audit.json');near(audit['summary']['max_relative_L2'],max_change)
 strata=[];paired=[];fixed=[i for i,(y,p) in enumerate(zip(od['clean']['labels'],od['clean']['predictions'])) if y==p];retention={}
 for scene in VIEWS:
  a=nd[scene];b=od[scene];y=a['labels'];pp=a['predictions'];qq=b['predictions'];retention[scene]={arm:metrics([y[i] for i in fixed],[p[i] for i in fixed]) for arm,p in [('D_raw',qq),('TIMEGRID',pp)]};paired.append(dict(scene=scene,count=1260,repairs=sum(p==t and q!=t for p,q,t in zip(pp,qq,y)),breaks=sum(p!=t and q==t for p,q,t in zip(pp,qq,y)),net_correct=ns[scene]['correct']-os[scene]['correct']))
  groups=defaultdict(list)
  for i,(pid,tx) in enumerate(zip(a['physical_ids'],y)):
   groups['TX',str(tx)].append(i);groups['day',str(pid[2])].append(i)
   if scene!='clean':
    p=a['physics'][i]
    for dim,key in [('state','state'),('lock','lock'),('agc_limited','agc_limited'),('kernel_length','residual_kernel_length'),('boundary','boundary')]:groups[dim,str(p[key])].append(i)
    for dim,key,edges in [('quality','quality_snr_db',(0,5,10,20,30)),('added_noise_SNR','signal_to_added_noise_db',(0,5,10,20,30)),('abs_CFO','output_frequency_hz',(10,100,1000))]:
     x=abs(p[key]) if dim=='abs_CFO' else p[key];bin=next(('<'+str(z) for z in edges if x<z),'>='+str(edges[-1]));groups[dim,bin].append(i)
  for (dim,value),ix in groups.items():
   for arm,p in [('D_raw',qq),('TIMEGRID',pp)]:strata.append(dict(arm=arm,scene=scene,dimension=dim,value=value,**metrics([y[i] for i in ix],[p[i] for i in ix])))
 delta={v:100*(ns[v]['accuracy']-os[v]['accuracy']) for v in VIEWS};df1=100*(ns['clean']['macro_F1']-os['clean']['macro_F1']);dv=100*(new['seen_V']['source_val_accuracy']-old['seen_V']['source_val_accuracy']);checks=dict(clean=delta['clean']>=1.-1e-10,macroF1=df1>=-1e-10,seenV=dv>=-.5-1e-10,complete_health=new['final_base_L_objective']<=.5)
 online=read(root/'result.json')['qualification']
 if all(checks.values())!=online['qualified']:raise ValueError('Independent predefined source qualification differs')
 stdout_paths=[root/'logs'/(ROW+'.stdout.log'),reference.parent.parent/'logs'/(reference.parent.name+'.stdout.log')];stdout_audits=[]
 for path in stdout_paths:
  text=path.read_text(encoding='utf8',errors='replace');bad=re.findall(r'Traceback|CUDA out of memory|RuntimeError:|Killed|FloatingPointError:',text)
  if bad:raise ValueError('Full source stdout critical errors')
  stdout_audits.append(dict(arm='TIMEGRID' if path==stdout_paths[0] else 'D_raw',lines=len(text.splitlines()),critical_errors=0,epoch_end_markers=len(re.findall(r'\[EPOCH-END\]',text)),unavailable_metric_nan_mentions=len(re.findall(r'(?i)nan',text))))
 for i,c in enumerate(ncurve,1):c.update(grid_relative_L2_mean=statistics.mean(per_epoch[i]),grid_relative_L2_max=max(per_epoch[i]))
 summary=dict(status='ANALYZED',execution='VERIFIED',run_id=RUN,source_only=True,model_seeds=[SEED],held_RX=1,scores=dict(D_raw=os,TIMEGRID=ns),all_view_gain_pp=delta,urban_mean_gain_pp=sum(delta[v] for v in VIEWS[4:])/3,clean_macroF1_gain_pp=df1,seenV_gain_pp=dv,checks=checks,qualified=all(checks.values()),retention_on_fixed_D_raw_clean_correct=retention,fixed_clean_correct_count=len(fixed),paired=paired,training=dict(D_raw=ohealth,TIMEGRID=nhealth),timegrid=dict(views=n,physical_unique=len(unique),sessions=len(sessions),coverage_cells=len(coverage),max_relative_L2=max_change,power_ratio_min=power_min,power_ratio_max=power_max,construct_seconds=audit['summary']['construct_seconds']),full_primary_L_U_sequence_exact=True,full_stdout_audits=stdout_audits,new_model_epochs=20,new_optimizer_updates=4440,new_NN=nhealth['NN']+2,reused_model_epochs=20,reused_optimizer_updates=4440,reused_NN=ohealth['NN'],head_fit_updates=0,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,limits=['One exposedsourceRX1/one modelseed,not target orcrossseed','No As training orAg+As interaction','Tracebounds independentlyrecount savedscalars,not originalIQ recomputation','OldD_raw perpacket practical margins/physics missing;unchangednamedphysicaldraw/config andmatchingIDs reused','Budgetexhaustionnotconvergence'])
 write(root/'analysis_summary.json',summary);write(root/'source_strata.json',strata);write(root/'training_curves.json',ocurve+ncurve);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--reference',required=True);a=p.parse_args();print(json.dumps(analyze(a.input,a.reference)))
