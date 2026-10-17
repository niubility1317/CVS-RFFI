"""Independent all-record source scoring;stdlib only,no candidate engine/model."""
import argparse,gzip,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path
ARMS=('original','bank1','bank2','bank3');SCENES=('practical_mid_urban','practical_low_urban')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
 with gzip.open(p,'rt',encoding='utf8') as f:
  for line in f:yield json.loads(line)
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def source_metadata(records,allowed):
 meta={}
 for r in records:
  pid=r['physical_id']
  if not isinstance(pid,str) or not pid or pid not in allowed or pid in meta or type(r['TX']) is not int or r['TX'] not in range(6) or type(r['RX']) is not int or r['RX'] not in (3,4,6,8) or type(r['day']) is not int or r['day'] not in (1,2,3) or type(r['eq']) is not int or r['eq']!=1:raise ValueError('Only exact authorized visiblefitL metadata,opaque IDs never decoded')
  meta[pid]=r
 if set(meta)!=set(allowed):raise ValueError('All and only authorized fitL ID metadata required')
 return meta

def metrics(recs):
 n=len(recs)
 if not n:raise ValueError('Missing registered source observations')
 cm=[[0]*6 for _ in range(6)];ce=[];margin=[]
 for r in recs:
  y,p=r['TX'],r['prediction'];v=r['logits']
  if type(y) is not int or y not in range(6) or type(p) is not int or p not in range(6) or len(v)!=6 or any(not math.isfinite(z) for z in v) or p!=max(range(6),key=lambda k:v[k]):raise ValueError('Complete original defaulthead6 logits/argmax required')
  mm=(v[y]-max(z for j,z in enumerate(v) if j!=y))/30
  if abs(mm-r['margin_over30'])>1e-10:raise ValueError('Fixed cos30 margin differs')
  top=max(v);ce.append(top+math.log(sum(math.exp(z-top) for z in v))-v[y]);margin.append(mm);cm[y][p]+=1
 f=[]
 for k in range(6):
  den=sum(cm[k])+sum(z[k] for z in cm);f.append(2*cm[k][k]/den if den else 0.)
 return dict(count=n,accuracy=sum(cm[k][k] for k in range(6))/n,macro_F1=sum(f)/6,confusion=cm,mean_CE=statistics.mean(ce),mean_margin_over30=statistics.mean(margin))

def qualify(groups):
 if set(groups)!=set(SCENES):raise ValueError('Both complete scenes required')
 c={}
 for s,g in groups.items():
  if g['count']!=5040 or set(g['banks'])!=set(ARMS):raise ValueError('Full four-bank source population required')
  ref=g['banks']['original'];others=[g['banks'][b] for b in ARMS[1:]]
  c[s+'/advantage']=100*(ref['accuracy']-statistics.mean(z['accuracy'] for z in others))>=1.-1e-10
  c[s+'/disagreement']=statistics.mean(z['disagreement_vs_original'] for z in others)>=.03-1e-10
  c[s+'/no_better_bank']=all(100*(z['accuracy']-ref['accuracy'])<=.5+1e-10 for z in others)
 return c

def analyze(root):
 root=Path(root);base=root/'frozen-ALL-s2026092701'
 if not (root/'completion.json').exists() or (root/'failure.json').exists():raise ValueError('Full fixed prediction matrix before independent source scoring')
 s=read(base/'summary.json');legal=read(base/'source_contract.json')['role_ids']['L_s'];source=source_metadata(read(base/'source_labels.json'),legal);p={};expected={('clean','raw')}|{(sc,b) for sc in SCENES for b in ARMS};truth={};meta={};count=0
 for r in rows(base/'predictions.jsonl.gz'):
  key=(r['scene'],r['bank']);pid=r['physical_id'];a=source.get(pid) if isinstance(pid,str) else None;tx=a['TX'] if a else None
  if a is None or key not in expected or r['TX']!=tx or (r['RX'],r['day'])!=(a['RX'],a['day']) or (key,pid) in p:raise ValueError('Illegal role/label/view/duplicate source prediction')
  p[key,pid]=r;truth.setdefault(pid,tx);count+=1
 if len(truth)!=5040 or count!=45360 or Counter((source[pid]['TX'],source[pid]['RX'],source[pid]['day']) for pid in truth)!=Counter({(t,r,dd):70 for t in range(6) for r in (3,4,6,8) for dd in (1,2,3)}):raise ValueError('Exact all72fitLcells/9views required')
 ids=sorted(truth)
 if len(p)!=len(expected)*5040 or any((k,pid) not in p for k in expected for pid in ids):raise ValueError('Missing registered physical prediction')
 for r in rows(base/'physics.jsonl.gz'):
  sc=r['scene'];pid=r['physical_id'];key=(sc,pid)
  if sc not in SCENES or pid not in truth or key in meta or set(r['banks'])!=set(ARMS) or set(r['relative_inputs'])!=set(ARMS) or r['native_process_calls']!=5 or r['original_replay_MSE']!=0 or not r['same_underlying_H_and_noise_seed'] or not r['processed_noise_not_claimed_equal']:raise ValueError('Full paired physical controls required')
  nom=r['banks']['original'];rx,day=source[pid]['RX'],source[pid]['day'];session=f'rx{rx}:day{day}:eq1'
  for b,m in r['banks'].items():
   sid=session if b=='original' else session+':virtual-'+b
   if m['receiver']['session_id']!=sid or m['seed']!=r['seed'] or m['scenario']!=sc or (m['fs_hz'],m['fc_hz'],m['processing_route'],m['mode'],m['channel_equalization_applied'],m['boundary'],m['added_noise_variance_reference'])!=(25000000.,2462000000.,'residual','post_sync',False,'synthetic_reflect',.001) or m['tx_impairment_correction_applied'] or m['true_total_snr_known']:raise ValueError('Fixed source session/fullpractical receiver only')
   for k in ('geometry','state_end','state_counts','state_probabilities','quality_snr_db','residual_scale','receiver_lock_status','propagation_kernel_length','residual_kernel_length','residual_kernel_energy','numerical_filter_latency_samples','boundary','noise_boundary'):
    if m[k]!=nom[k]:raise ValueError('H/geometry/quality/state changed acrosssession control')
   a=r['relative_inputs'][b]
   if any(not math.isfinite(a[k]) or a[k]<0 for k in ('MSE','relative_IQ_RMS')) or b=='original' and any(a[k]!=0 for k in a):raise ValueError('Finite observed input perturbation telemetry required')
  meta[key]=r
 if len(meta)!=10080:raise ValueError('Both scenes all5040 physical controls required')
 receivers=read(base/'receivers.json')
 if len(receivers)!=48 or {m['receiver']['session_id'] for r in meta.values() for m in r['banks'].values()}!=set(receivers):raise ValueError('12original plus36virtual stable RXsessions required')
 for r in meta.values():
  for m in r['banks'].values():
   if m['receiver']!=receivers[m['receiver']['session_id']]:raise ValueError('RX hardware draw not stable acrossTX/daypackets/scenes')
 groups={};paired=[];retention={};strata=[];inputstats={};clean=metrics([p[('clean','raw'),pid] for pid in ids]);correct=[pid for pid in ids if p[('clean','raw'),pid]['prediction']==truth[pid]]
 for sc in SCENES:
  banks={};retention[sc]={};inputstats[sc]={}
  for b in ARMS:
   q=metrics([p[(sc,b),pid] for pid in ids]);q['disagreement_vs_original']=sum(p[(sc,b),pid]['prediction']!=p[(sc,'original'),pid]['prediction'] for pid in ids)/5040;banks[b]=q
   retention[sc][b]=metrics([p[(sc,b),pid] for pid in correct]);qq=[meta[sc,pid]['relative_inputs'][b]['relative_IQ_RMS'] for pid in ids];inputstats[sc][b]=dict(min=min(qq),median=statistics.median(qq),max=max(qq),mean=statistics.mean(qq))
   a=sum(p[(sc,b),pid]['prediction']==truth[pid] and p[(sc,'original'),pid]['prediction']!=truth[pid] for pid in ids);z=sum(p[(sc,b),pid]['prediction']!=truth[pid] and p[(sc,'original'),pid]['prediction']==truth[pid] for pid in ids);paired.append(dict(scene=sc,bank=b,repairs=a,breaks=z,net=a-z))
  groups[sc]=dict(count=5040,banks=banks)
  bins=defaultdict(list)
  for pid in ids:
   nom=meta[sc,pid]['banks']['original'];q=nom['quality_snr_db'];quality='<0' if q<0 else '[0,5)' if q<5 else '[5,10)' if q<10 else '[10,20)' if q<20 else '>=20'
   for dim,value in [('TX',truth[pid]),('RX',source[pid]['RX']),('day',source[pid]['day']),('state',nom['state_end']),('lock',nom['receiver_lock_status']),('quality',quality)]:bins[dim,str(value)].append(pid)
  for (dim,value),ix in bins.items():
   for b in ARMS:strata.append(dict(scene=sc,dimension=dim,value=value,bank=b,**metrics([p[(sc,b),pid] for pid in ix])))
 checks=qualify(groups)
 if checks!=s['checks'] or all(checks.values())!=(s['status']=='PASS_FIXED_SESSION_SENSITIVITY_RATIONALE_ONLY'):raise ValueError('Independent frozen fixed criteria differ')
 for sc in SCENES:
  for b in ARMS:
   for k in ('accuracy','macro_F1','disagreement_vs_original'):
    if abs(groups[sc]['banks'][b][k]-s['groups'][sc]['banks'][b][k])>1e-10:raise ValueError('Independent full metrics differ from online')
 if s['NN']!=45362 or s['decisions']!=45360 or s['physics_records']!=10080 or s['native_process_calls']!=50400 or s['receiver_sessions']!=48 or not s['parameters_and_buffers_unchanged'] or not s['gradients_unset'] or any(s[k] for k in ('target_access','model_epochs','optimizer_updates','head_fit_updates','teacher_forwards','EMA_updates','automatic_training','performance_improvement_proved','goal_complete')):raise ValueError('Complete frozen source-only scope/cost required')
 log=(root/'dispatcher.log').read_text(encoding='utf8') if (root/'dispatcher.log').exists() else ''
 if log and (sum(x.startswith('BATCH ') for x in log.splitlines())!=40 or 'Traceback' in log or 'COMPLETE '+s['status'] not in log):raise ValueError('Full40batch stdout and natural completion required')
 summary=dict(status='ANALYZED',execution='VERIFIED',scientific_status=s['status'],qualified=all(checks.values()),checks=checks,groups=groups,clean=clean,paired=paired,fixed_clean_correct_count=len(correct),fixed_clean_correct_retention=retention,input_relative_RMS=inputstats,source_only=True,held_RX_used=False,model_seeds=[2026092701],RX_session_draws_not_modelseeds=True,NN=45362,physical_calls=50400,unique_L=5040,source_cells=72,physics_records=10080,predictions=45360,parent_epochs=20,parent_updates=4440,parent_NN=s['parent_NN'],new_model_epochs=0,new_optimizer_updates=0,head_fit_updates=0,wall_seconds=s['wall_seconds'],target_access=False,automatic_training=False,goal_complete=False,limits=['All observations are parent-exposed fit L,not held/target/newtraining evidence','Only3fixedadditional hardwarebanks;not universalhardware range or modelseed confirmation','Independent reader recomputesall decisions/CE/F1/margins/groups/criteria from saved records,not original IQ/hardware physics replay','Same underlying noise seed,processed noise/AGC may change with RX response','Passing only authorizes rationale for separate training,not proven Ag or accuracy gain'])
 write(root/'analysis_summary.json',summary);write(root/'source_strata.json',strata);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();s=analyze(a.input);print(json.dumps(dict(status=s['status'],scientific_status=s['scientific_status'],groups=s['groups'],NN=s['NN'])))
