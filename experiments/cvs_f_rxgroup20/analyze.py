"""Independent full source recount;stdlib only,no IQ/checkpoint/model/target."""
import argparse,csv,gzip,json,math,re,statistics
from collections import Counter,defaultdict
from pathlib import Path
RUN='20261017-phase1-FcrossRX-rxgroup-sourceRX1-e20-s2026092701-m2-r01';ARMS=('BATCH','RXGROUP');SEED=2026092701;FIT=(3,4,6,8)
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
KEYS={'anti':('p_dc_offset','p_bandedge_taper'),'channel':('p_time_shift','p_amp_scale','p_phase_rot','p_cfo','p_phase_noise','p_multipath','p_awgn')}
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
 if len(y)!=len(p) or not y or any(a not in range(6) or b not in range(6) for a,b in zip(y,p)):raise ValueError('All six registered-class decisions required')
 C=[[0]*6 for _ in range(6)]
 for a,b in zip(y,p):C[a][b]+=1
 correct=sum(C[k][k] for k in range(6));f=[]
 for k in range(6):
  den=sum(C[k])+sum(z[k] for z in C);f.append(2*C[k][k]/den if den else 0.)
 return dict(count=len(y),correct=correct,accuracy=correct/len(y),macro_F1=sum(f)/6,confusion=C)
def predictions(base):
 result={};data={};common=None
 for view in VIEWS:
  r=read(base/'source_readouts'/(view+'.json'));ids=[tuple(p) for p in r['physical_ids']];y=r['labels'];p=r['predictions']
  if len(ids)!=1260 or len(set(ids))!=1260 or set(z[1] for z in ids)!={1} or set(z[2] for z in ids)!={1,2,3} or Counter(y)!=Counter({k:210 for k in range(6)}) or r['head_fit_updates']:raise ValueError('Exact source-heldL1260 role required')
  if common is None:common=(ids,y)
  elif common!=(ids,y):raise ValueError('View physical IDs/truth pairing differs')
  m=metrics(y,p);near(m['accuracy'],r['accuracy']);near(m['macro_F1'],r['macro_F1'])
  if len(r['margin_over30'])!=1260 or any(not math.isfinite(x) for x in r['margin_over30']):raise ValueError('Finite defaultcos30 margins required')
  m['mean_margin_over30']=statistics.mean(r['margin_over30']);m['mean_CE']=r['mean_CE'];result[view]=m;data[view]=r
 return result,data

def labeled_TX(pid):
 # Original source-cache flattened layout only for saved visible-L records.
 # Validate RX/day/sig here and every TX count/pair count against saved native audits.
 eq,rx,day,sig,base=pid;within=base%15000;rx_index=(within//1000)//3
 if eq!=1 or base<0 or base>=90000 or (1,3,4,6,8)[rx_index]!=rx or within//1000%3+1!=day or within%1000!=sig or sig not in range(70):raise ValueError('Saved visible-L metadata layout differs')
 return base//15000

def training(base,arm):
 r=read(base/'satmix_result.json');done=read(base/'completion.json');init=read(base/'initialization.json');freeze=read(base/'source_frozen.json');contract=read(base/'source_contract.json');args=read(base/'actual_native_args.json');cfg=read(base/'resolved_config.json')
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access'] or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']:raise ValueError('Fixed scratch FINAL20 default-head provenance required')
 if (args['epochs'],args['label_epochs'],args['pseudo_epochs'],args['seed'],args['label_smoothing'])!=(20,13,7,SEED,.01) or args['baseline_ckpt'] or args['teacher_ckpt'] or args['source_fold_held_rx']!=1 or args['use_concat_sat_channel_aug'] or not args['use_aug'] or contract['source_receivers']!=list(FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Exact source training data/recipe required')
 if cfg['arm']!=arm or cfg['updates']!=4440 or cfg['training_mechanism']['L_forwards_per_packet']!=6 or r['actual_steps']!=4440 or r['L_forward_packets']!=3409920 or r['supcon_applied']!=3330 or r['As_training_observations'] or not r['default_head_joint'] or r['head_fit_updates'] or r['target_access'] or r['U_truth_read'] or r['checkpoint_sources']:raise ValueError('Actual sixL/default-head/hiddenU budget differs')
 steps=list(lines(base/'satmix_steps.jsonl.gz'));units=list(lines(base/'unit_steps.jsonl.gz'));native=list(lines(base/'step_metrics.jsonl.gz'));epochs=list(lines(base/'epoch_metrics.jsonl'));metrics_epochs=list(lines(base/'metrics_epoch.jsonl'));seq=list(lines(base/'training_sequence.jsonl.gz'))
 if len(steps)!=4440 or len(units)!=4440 or len(native)!=4440 or len(epochs)!=20 or len(metrics_epochs)!=20 or len(seq)!=5994 or Counter(z['role'] for z in seq)!=Counter(L=4440,U=1554):raise ValueError('Full step/epoch/original L-U sequences required')
 for f in ('epoch_metrics.csv','metrics_epoch.csv'):
  with (base/f).open(encoding='utf-8-sig',newline='') as stream:csvrows=list(csv.DictReader(stream))
  if len(csvrows)!=20:raise ValueError('Complete CSV epoch rows required')
 L=[x['ids'] for x in seq if x['role']=='L'];grads=[];sup=0;curves=[];nulls=Counter()
 for i,(a,u,n) in enumerate(zip(steps,units,native),1):
  e=(i-1)//222+1
  if (a['step'],a['epoch'],u['step'],n['step'],n['epoch'])!=(i,e,i,i,e) or u['physical_clip_calls']!=i:raise ValueError('Ordered native/F steps differ')
  for k in ('total_L','raw_CE','ordinary_CE','s_slot_CE','raw_KL','ordinary_KL','s_slot_KL','supcon','supcon_weight','ordinary_raw_MSE','sat_raw_MSE'):
   if not math.isfinite(a[k]):raise ValueError('Nonfinite active labeled term')
  near(a['total_L'],.5*(a['raw_CE']+a['raw_KL'])+.25*(a['ordinary_CE']+a['ordinary_KL'])+.25*(a['s_slot_CE']+a['s_slot_KL'])+a['supcon_weight']*a['supcon'])
  if a['As_packets'] or a['sat_raw_MSE'] or not a['all_L_supervised'] or not a['same_IQ_pairs']:raise ValueError('Original absolute L supervision differs')
  sup+=a['supcon_weight']>0;m=n['metrics']
  for k in ('train/grad_before_clip','train/grad_clip_limit','train/grad_total','train/optimizer_step_applied','train/pseudo_truth_available'):
   if m.get(k) is None or not math.isfinite(m[k]):raise ValueError('Missing active native field')
  if m['train/pseudo_truth_available'] or m['train/optimizer_step_applied']!=1 or m.get('train/skipped_nonfinite_grad',0) or m.get('train/skipped_nonfinite_loss',0):raise ValueError('Hidden U truth or nonfinite/skipped update')
  grads.append(m['train/grad_before_clip'])
  for k,v in m.items():
   if v is None:nulls[k]+=1
 if sup!=3330:raise ValueError('Original SupCon warm5/ramp10 differs')
 audit=read(base/'coupling_audit.json');sm=audit['summary'];cov=Counter();unique=set();totals=defaultdict(Counter);gates=defaultdict(Counter);gate_epoch=defaultdict(lambda:defaultdict(Counter));mse=defaultdict(list);stage_mse=defaultdict(list);seconds=0.;groups=0;nonzero=0;count=0
 for i,z in enumerate(lines(base/'coupling_steps.jsonl.gz'),1):
  e=(i-1)//222+1
  if z['step']!=i or z['epoch']!=e or z['arm']!=arm or z['ids']!=L[i-1] or len(z['ids'])!=128 or len(z['stages'])!=2 or not z['all_L_supervised'] or not z['PA_DAC_class_signature_rules_unchanged'] or not math.isfinite(z['actual_ordinary_raw_MSE']):raise ValueError('One normal real-L coupling call required')
  near(z['actual_ordinary_raw_MSE'],steps[i-1]['ordinary_raw_MSE'])
  counts=Counter((labeled_TX(pid),pid[1]) for pid in z['ids']);rxcount=Counter(pid[1] for pid in z['ids']);pairs=sum(n*m for (y,rx),n in counts.items() for (yy,rr),m in counts.items() if y==yy and rx!=rr)
  for pid in z['ids']:cov[labeled_TX(pid),pid[1],pid[2]]+=1;unique.add(tuple(pid))
  mse[e].append(z['actual_ordinary_raw_MSE'])
  for stage,name in zip(z['stages'],('anti','channel')):
   if stage['stage']!=name or set(stage['whole_gates'])!=set(KEYS[name]) or {g['RX']:g['count'] for g in stage['groups']}!=dict(rxcount):raise ValueError('Same original operators/actual RX groups required')
   byrx={g['RX']:g for g in stage['groups']};groups+=len(byrx)
   for rx,g in byrx.items():
    if g['seed']!=20261017831+i*64+(0 if name=='anti' else 32)+rx or set(g['gates'])!=set(KEYS[name]):raise ValueError('Private stream depends on fixed step/stage/RX only')
   for k in KEYS[name]:
    a=stage['crossRX_sameTX_gate_disagreement'][k];dis=sum(n*m for (y,rx),n in counts.items() for (yy,rr),m in counts.items() if y==yy and rx!=rr and byrx[rx]['gates'][k]!=byrx[rr]['gates'][k]);actual=dis if arm=='RXGROUP' else 0
    if (a['positive_pairs'],a['would_group_disagree'],a['actual_disagree'])!=(pairs,dis,actual) or not 0<a['probability']<=1:raise ValueError('Full cross-RX sameTX pair-gate recount differs')
    totals[k].update(positive_pairs=pairs,would_group_disagree=dis,actual_disagree=actual)
    on=(128*int(stage['whole_gates'][k])) if arm=='BATCH' else sum(g['count']*int(g['gates'][k]) for g in byrx.values())
    gates[k].update(eligible=128,actual_on=on);gate_epoch[e][k].update(positive_pairs=pairs,actual_disagree=actual,eligible=128,actual_on=on)
   if stage['group_vs_whole_stage_MSE']<0 or not math.isfinite(stage['group_vs_whole_stage_MSE']+stage['seconds']):raise ValueError('Finite stage diagnostics required')
   nonzero+=stage['group_vs_whole_stage_MSE']>0;seconds+=stage['seconds'];stage_mse[e].append(stage['group_vs_whole_stage_MSE'])
  count+=1
 if count!=4440 or len(unique)!=5040 or len(cov)!=72 or sum(cov.values())!=568320 or sm['calls']!=4440 or sm['stages']!=8880 or sm['packets']!=568320 or sm['coverage_cells']!=72 or sm['group_calls']!=groups or sm['nonzero_group_stages']!=nonzero or sm['new_NN'] or sm['U_grouped_calls'] or audit['model_parameters_added'] or not audit['original_U_unchanged']:raise ValueError('Complete native coupling coverage differs')
 if cov!=Counter({(x['TX'],x['RX'],x['day']):x['count'] for x in audit['coverage']}):raise ValueError('Full visible-L TX/RX/day coverage recount differs')
 for k,a in totals.items():
  if dict(a)!=sm['pair_gate_counts'][k]:raise ValueError('Full gate sum differs')
 near(seconds,sm['construct_audit_seconds'],1e-10)
 for e,row in enumerate(epochs,1):
  if (row['epoch'],row['epochs'],row['label_epochs'],row['pseudo_epochs'])!=(e,20,13,7) or row['train_pseudo_truth_available'] or row['train_skipped_nonfinite_grad'] or row['train_skipped_nonfinite_loss'] or row['train_optimizer_step_applied']!=1:raise ValueError('Epoch schedule/health differs')
  sl=steps[(e-1)*222:e*222];c=dict(arm=arm,epoch=e,**{k:statistics.mean(x[k] for x in sl) for k in ('total_L','raw_CE','ordinary_CE','supcon','supcon_weight')},ordinary_MSE=statistics.mean(mse[e]),group_stage_MSE=statistics.mean(stage_mse[e]),grad_before_clip=statistics.mean(grads[(e-1)*222:e*222]),train_tx_accuracy=row['train_tx_acc'],seenV_accuracy=row['val_tx_acc'],lr=row['lr'],epoch_seconds=row['epoch_time_s'],pseudo_selected=row['train_pseudo_selected'],pseudo_total=row['train_pseudo_total']);curves.append(c)
 near(statistics.mean(x['metrics']['train/loss_tx_labeled'] for x in native[-222:]),r['final_L_CE'])
 NN=sum(r[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets'))-2+r['held_forward_packets']
 text=(base.parents[1]/'logs'/(base.parent.name+'.stdout.log')).read_text(encoding='utf8',errors='replace')
 if re.findall(r'Traceback|CUDA out of memory|RuntimeError:|Killed|FloatingPointError:',text):raise ValueError('Critical full stdout error')
 return r,curves,dict(epochs=20,steps=4440,optimizer_updates=4440,L_NN=3409920,NN=NN,gradient_preclip_mean=statistics.mean(grads),gradient_preclip_max=max(grads),gradient_zero=sum(x==0 for x in grads),gradient_above5=sum(x>5 for x in grads),native_disabled_null_fields=len(nulls),all_active_updates_finite=True,original_LU_records=5994,Lunique=len(unique),coverage_cells=72,coupling_calls=count,stages=8880,group_stage_calls=groups,construct_audit_seconds=seconds,nonzero_group_stages=nonzero,pair_gate_counts={k:dict(v) for k,v in totals.items()},actual_operator_packet_counts={k:dict(v) for k,v in gates.items()},gate_epoch={str(e):{k:dict(v) for k,v in a.items()} for e,a in gate_epoch.items()},stdout_lines=len(text.splitlines()),stdout_epoch_markers=len(re.findall(r'\[EPOCH-END\]',text)),parent_epochs=0),seq

def analyze(root):
 root=Path(root)
 if (root/'failure.json').exists() or not (root/'completion.json').exists():raise ValueError('Full matrix predictions first,then independent scoring')
 records={};scores={};data={};health={};sequences={};curves=[]
 for a in ARMS:
  b=root/(a+'-RX1-s'+str(SEED))/'source';records[a],c,health[a],sequences[a]=training(b,a);scores[a],data[a]=predictions(b);curves+=c
 for v in VIEWS:
  for k in ('physical_ids','labels'):
   if data['BATCH'][v][k]!=data['RXGROUP'][v][k]:raise ValueError('Matched held physical IDs/labels required')
  if v!='clean' and data['BATCH'][v]['physics']!=data['RXGROUP'][v]['physics']:raise ValueError('Exact fixed evaluation channel states and compensation must match')
 fixed=[i for i,(y,p) in enumerate(zip(data['BATCH']['clean']['labels'],data['BATCH']['clean']['predictions'])) if y==p];retention={};paired=[];strata=[]
 for v in VIEWS:
  b=data['BATCH'][v];r=data['RXGROUP'][v];y=r['labels'];p=r['predictions'];q=b['predictions'];retention[v]={a:metrics([y[i] for i in fixed],[z[i] for i in fixed]) for a,z in [('BATCH',q),('RXGROUP',p)]};paired.append(dict(scene=v,count=1260,repairs=sum(t==x and t!=z for t,x,z in zip(y,p,q)),breaks=sum(t!=x and t==z for t,x,z in zip(y,p,q)),net_correct=scores['RXGROUP'][v]['correct']-scores['BATCH'][v]['correct']))
  groups=defaultdict(list)
  for i,(pid,tx) in enumerate(zip(r['physical_ids'],y)):
   groups['TX',str(tx)].append(i);groups['day',str(pid[2])].append(i)
   if v!='clean':
    ph=r['physics'][i]
    for dim,k in [('state','state'),('lock','lock'),('AGC_limited','agc_limited'),('kernel_length','residual_kernel_length'),('boundary','boundary')]:groups[dim,str(ph[k])].append(i)
    for dim,k,edges in [('quality','quality_snr_db',(0,5,10,20,30)),('added_noise_SNR','signal_to_added_noise_db',(0,5,10,20,30)),('abs_CFO','output_frequency_hz',(10,100,1000))]:
     x=abs(ph[k]) if dim=='abs_CFO' else ph[k];bn=next(('<'+str(z) for z in edges if x<z),'>='+str(edges[-1]));groups[dim,bn].append(i)
  for (dim,value),ix in groups.items():
   for a in ARMS:
    z=data[a][v];strata.append(dict(arm=a,scene=v,dimension=dim,value=value,**metrics([y[i] for i in ix],[z['predictions'][i] for i in ix]),mean_margin_over30=statistics.mean(z['margin_over30'][i] for i in ix)))
 delta={v:100*(scores['RXGROUP'][v]['accuracy']-scores['BATCH'][v]['accuracy']) for v in VIEWS};f1=100*(scores['RXGROUP']['clean']['macro_F1']-scores['BATCH']['clean']['macro_F1']);dv=100*(records['RXGROUP']['seen_V']['source_val_accuracy']-records['BATCH']['seen_V']['source_val_accuracy']);checks=dict(clean=delta['clean']>=1.-1e-10,macroF1=f1>=-1e-10,seenV=dv>=-.5-1e-10,healthy=all(r['final_L_CE']<=.5 for r in records.values()))
 online=read(root/'result.json')['qualification']
 if checks!=online['checks'] or all(checks.values())!=online['qualified']:raise ValueError('Independent fixed source qualification differs')
 for v in VIEWS:near(delta[v],online['all_view_gain_pp'][v],1e-10)
 same_seq=sequences['BATCH']==sequences['RXGROUP'];same_L=[x for x in sequences['BATCH'] if x['role']=='L']==[x for x in sequences['RXGROUP'] if x['role']=='L'];same_U=[x for x in sequences['BATCH'] if x['role']=='U']==[x for x in sequences['RXGROUP'] if x['role']=='U']
 summary=dict(status='ANALYZED',execution='VERIFIED',run_id=RUN,source_only=True,model_seeds=[SEED],held_RX=1,scores=scores,all_view_gain_pp=delta,urban_mean_gain_pp=sum(delta[v] for v in VIEWS[4:])/3,clean_macroF1_gain_pp=f1,seenV_gain_pp=dv,checks=checks,qualified=all(checks.values()),fixed_Batch_clean_correct_count=len(fixed),retention_on_fixed_Batch_clean_correct=retention,paired=paired,training=health,final_L_CE={a:r['final_L_CE'] for a,r in records.items()},seen_V={a:r['seen_V'] for a,r in records.items()},full_LU_sequences_exact=same_seq,full_L_sequences_exact=same_L,full_U_sequences_exact=same_U,all_six_physics_states_parameters_exact=True,new_model_epochs=40,new_optimizer_updates=8880,new_NN=sum(h['NN'] for h in health.values())+2,dispatcher_realCP_smoke_NN=2,head_fit_updates=0,target_access=False,Ag_efficacy_proved=False,joint_goal_complete=False,limits=['One exposed sourceRX1/modelseed;no formal target or crossseed evidence','No As training/new physical marginal impairment or Ag+As interaction','Private native stream parity synthetic checks;live IQ not reread for inverse reconstruction','Gate and visible-L coverage independently recounted from complete saved records;U labels never decoded','Budget exhaustion is not convergence'])
 write(root/'analysis_summary.json',summary);write(root/'source_strata.json',strata);write(root/'training_curves.json',curves);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();print(json.dumps(analyze(a.input)))
