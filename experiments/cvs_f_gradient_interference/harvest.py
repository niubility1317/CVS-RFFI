"""Independent stdlib reader of saved source scalars/decisions; no IQ/CP/model imports."""
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict,Counter
from pathlib import Path
from .core import measure,decision

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def lines(p):
 with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(s) for s in f]
def write_csv(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r));
 with Path(path).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def metrics(rows):
 cm=[[0]*6 for _ in range(6)]
 for r in rows:
  y,p=r['TX'],r['prediction']
  if not isinstance(y,int) or not isinstance(p,int) or y not in range(6) or p not in range(6) or not math.isfinite(r['margin_over30']):raise ValueError('Legal finite default-head L decision required')
  cm[y][p]+=1
 n=len(rows);correct=sum(cm[i][i] for i in range(6));f1=sum(2*cm[i][i]/max(1,sum(cm[i])+sum(row[i] for row in cm)) for i in range(6))/6
 return dict(count=n,correct=correct,accuracy=correct/n,macro_f1=f1,mean_margin_over30=statistics.mean(r['margin_over30'] for r in rows),confusion=cm)
def check_scalar(row):
 measured=measure(row['primary_squared'],row['secondary_squared'],row['dot'])
 for k,v in measured.items():
  a=row[k]
  if isinstance(v,bool) or v is None:
   if a is not v:raise ValueError('Saved scalar Boolean/undefined mismatch')
  elif not isinstance(a,(int,float)) or not math.isclose(a,v,rel_tol=1e-9,abs_tol=1e-12):raise ValueError('Saved scalar formula mismatch')
 return measured

def analyze(root,log,completion,output):
 root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);c=read(root/'resolved_config.json');summary=read(root/'summary.json');done=read(completion);smoke=read(root/'checkpoint_noquery_smoke.json');observation=read(root/'observation_audit.json')
 if c['schema']!='FcrossRX_cleanAs_gradient_diagnostic_v1' or c['target_access'] or c['optimizer_updates'] or c['V_values_used'] or c['U_values_used'] or c['head_fit_updates'] or c['source_L_count']!=5040 or c['source_RX']!=[3,4,6,8] or c['replicates']!=[0,1]:raise ValueError('Wrong registered source-only diagnosis')
 for r in (summary,done):
  if r['NN_packets']!=120964 or r['predictions']!=60480 or r['gradient_rows']!=1920 or r['optimizer_updates'] or r['EMA_updates'] or r['head_fit_updates'] or r['target_access'] or r['V_used'] or r['U_used'] or not r['parameters_and_buffers_unchanged']:raise ValueError('Wrong final source budget/state')
 if done['status']!='ARTIFACTS_COMPLETE' or smoke!=dict(status='PASS',real_checkpoints=2,synthetic_packets=4,IQ_reads_before_smoke=0,query_reads=0,optimizer_updates=0):raise ValueError('Real CP no-query smoke/terminal completion absent')
 obs=lines(root/'observations.jsonl.gz');grad=lines(root/'gradients.jsonl.gz');pred=lines(root/'predictions.jsonl.gz')
 if (len(obs),len(grad),len(pred))!=(80,1920,60480):raise ValueError('Incomplete actual records')
 lookup={};obs_ids={};cell=Counter()
 for o in obs:
  key=(o['rep'],o['batch']);expected_n=128 if o['batch']<39 else 48
  if key in lookup or len(o['ids'])!=expected_n or len(o['physics'])!=expected_n:raise ValueError('Observation key/count mismatch')
  lookup[key]=o
  for pid,physics in zip(o['ids'],o['physics']):
   if (o['rep'],pid) in obs_ids or physics['sample_id']!=pid or physics['ordinary_before_satellite'] or physics['channel_equalization_applied']:raise ValueError('Repeated/incorrect physical observation')
   obs_ids[o['rep'],pid]=physics
 if set(lookup)!={(r,b) for r in (0,1) for b in range(40)} or len(obs_ids)!=10080:raise ValueError('All source observations/replicates required')
 keys=set();paired=defaultdict(dict)
 for r in grad:
  key=(r['checkpoint'],r['rep'],r['batch'],r['secondary'],r['group']);check_scalar(r)
  if key in keys or r['checkpoint'] not in ('C0','C1') or (r['rep'],r['batch']) not in lookup or r['count']!=len(lookup[r['rep'],r['batch']]['ids']):raise ValueError('Gradient key/count mismatch')
  keys.add(key);paired[key[:-1]][r['group']]=r
 for key,groups in paired.items():
  if set(groups)!={'whole','default_head','SIG_TIME','Sinc_Hz','time_path','remaining'}:raise ValueError('Fixed whole/five groups missing')
  for k in ('primary_squared','secondary_squared','dot'):
   if not math.isclose(groups['whole'][k],sum(r[k] for n,r in groups.items() if n!='whole'),rel_tol=1e-9,abs_tol=1e-12):raise ValueError('Group/whole additive scalar conservation failed')
 result=decision(grad)
 if result!=summary['decision']:raise ValueError('Independent fixed evidence rule differs')
 seen=set();by=defaultdict(list);identity={};strata=defaultdict(list)
 for r in pred:
  key=(r['checkpoint'],r['rep'],r['view'],r['physical_id']);base=(r['rep'],r['physical_id'])
  if (r['rep'],r['batch']) not in lookup or r['physical_id'] not in lookup[r['rep'],r['batch']]['ids']:raise ValueError('Prediction batch/physical identity mismatch')
  if key in seen or base not in obs_ids or r['checkpoint'] not in ('C0','C1') or r['view'] not in ('primary_ordinary','same_ordinary','As') or r['RX'] not in (3,4,6,8) or r['day'] not in (1,2,3):raise ValueError('Prediction input role/key differs')
  seen.add(key);idmeta=(r['TX'],r['RX'],r['day'])
  if identity.setdefault(r['physical_id'],idmeta)!=idmeta:raise ValueError('Physical source label/RX/day changed')
  if r['view']=='As' and r['physics']!=obs_ids[base]:raise ValueError('Model-specific physics or selected redraw')
  prefix=(r['checkpoint'],r['rep'],r['view']);by[prefix].append(r)
  for dimension,value in [('RX',r['RX']),('TX',r['TX']),('day',r['day'])]+([(k,r['physics'][k]) for k in ('scene','state_end','receiver_lock_status')] if r['view']=='As' else []):strata[prefix+(dimension,str(value))].append(r)
 if len(identity)!=5040 or any(len(v)!=5040 for v in by.values()) or len(by)!=12:raise ValueError('Complete same physical decision support required')
 for pid,(tx,rx,day) in identity.items():cell[tx,rx,day]+=1
 if len(cell)!=72 or set(cell.values())!={70}:raise ValueError('Full legal source TX/RX/day coverage differs')
 raw_log=Path(log).read_text(encoding='utf-8');progress=[];other=[]
 for line in raw_log.splitlines():
  try:r=json.loads(line)
  except json.JSONDecodeError:other.append(line);continue
  if 'completed_batches' in r:progress.append(r)
  else:other.append(line)
 if len(progress)!=160 or progress[-1]['NN_packets']!=120964 or [(r['checkpoint'],r['rep'],r['batch']) for r in progress]!=[(a,r,b) for a in ('C0','C1') for r in (0,1) for b in range(40)]:raise ValueError('Full execution progress/cost log incomplete')
 if any(s in raw_log for s in ('Traceback (most recent call last)','FloatingPointError','nan','NaN')):raise ValueError('Critical execution error text')
 flat=[];gs=defaultdict(list)
 for r in grad:
  flat.append({k:v for k,v in r.items() if k not in ('primary_terms','secondary_terms')});gs[r['checkpoint'],r['rep'],r['secondary'],r['group']].append(r)
 gtable=[]
 for (model,rep,secondary,group),rr in gs.items():
  valid_ratio=[r['norm_ratio'] for r in rr if r['norm_ratio'] is not None];cos=[r['cosine'] for r in rr if r['cosine'] is not None]
  gtable.append(dict(checkpoint=model,rep=rep,secondary=secondary,group=group,batches=len(rr),median_norm_ratio=statistics.median(valid_ratio) if valid_ratio else None,median_cosine=statistics.median(cos) if cos else None,opposed_fraction=sum(r['opposed'] for r in rr)/len(rr),primary_reversal_fraction=sum(r['primary_direction_reversed'] for r in rr)/len(rr)))
 ptable=[dict(checkpoint=k[0],rep=k[1],view=k[2],**metrics(v)) for k,v in by.items()];stable=[dict(checkpoint=k[0],rep=k[1],view=k[2],dimension=k[3],value=k[4],**metrics(v)) for k,v in strata.items()]
 write_csv(output/'gradient_pairs.csv',flat);write_csv(output/'gradient_summary.csv',gtable);write_csv(output/'source_decisions.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in ptable]);write_csv(output/'source_strata.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in stable])
 final=dict(status='INDEPENDENTLY_VERIFIED_SOURCE_DIAGNOSTIC',decision=result,gradient_summary=gtable,source_decisions=ptable,complete_records=dict(observations=len(obs),gradients=len(grad),predictions=len(pred),whole_progress_lines=len(progress),other_log_lines=len(other),physical_L_unique=len(identity),cells=len(cell)),costs=dict(new_models=0,new_model_epochs=0,optimizer_updates=0,EMA_updates=0,head_fit_updates=0,NN_packets=120964,ordinary_generations=10080,full_As_generations=10080,wall_seconds=summary['wall_seconds'],reference_model_epochs=40,reference_updates=8880),source_only=True,target_access=False,model_seeds=1,observation_replicates=2,actual_AdamW_motion_measured=False,goal_completed=False)
 (output/'analysis_summary.json').write_text(json.dumps(final,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');return final

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--log',required=True);p.add_argument('--completion',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=analyze(a.input,a.log,a.completion,a.output);print(json.dumps({k:r[k] for k in ('status','decision','complete_records','costs')}))
