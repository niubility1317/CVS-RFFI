import json,math,time,re,gzip,sys,collections
from pathlib import Path
base=Path('/home/szu2070436088/2510044040/CV-SincNet');rid='20260928-phase1-ir-eg-source-screen-s392005-r01'
launch=json.loads((base/'runs'/rid/'launch_state.json').read_text())
skip={'source_ids','unlabeled_ids','selected_mask','groups','stable_observation_lengths','risk_crossfit_audit','used_divisors','field_divisors','field_components','normalization_scales','missing_reasons'}
def flatten(d,p=''):
 for k,v in d.items():
  if k in skip:continue
  key=p+'.'+k if p else k
  if isinstance(v,dict):yield from flatten(v,key)
  elif isinstance(v,(bool,int,float,str)) or v is None:yield key,v
def compact(d):
 return {k:v for k,v in flatten(d)}
def add(stats,k,v):
 if v is None:return
 if isinstance(v,(int,float)):
  if not math.isfinite(v):stats.setdefault(k,{'bad':0})['bad']=stats.get(k,{}).get('bad',0)+1;return
  a=stats.setdefault(k,dict(n=0,sum=0.,ss=0.,min=v,max=v,nonzero=0,first=v,last=v));a['n']+=1;a['sum']+=v;a['ss']+=v*v;a['min']=min(a['min'],v);a['max']=max(a['max'],v);a['last']=v;a['nonzero']+=v!=0
 elif isinstance(v,str):
  a=stats.setdefault(k,{'counts':{}});a['counts'][v]=a['counts'].get(v,0)+1
def finish(stats):
 for a in stats.values():
  if 'n' in a:
   a['mean']=a.pop('sum')/a['n'];a['sd']=math.sqrt(max(0,a.pop('ss')/a['n']-a['mean']**2))
 return stats
rows=[]
for lr in launch['rows']:
 root=Path(lr['output']);row=dict(row=lr['row_id'],files={},epochs={},records={},config=json.loads((root/'resolved_config.json').read_text()),dr_config=json.loads((root/'resolved_dr_config.json').read_text()))
 lasttime=0.;laststep=-1;idcounts=collections.Counter();seqerrors=[]
 for path in sorted(root.glob('*.jsonl')):
  name=path.name;size=path.stat().st_size;count=0;bad=[];partial=0;first=None;last=None;stats={};epochs={};small=[]
  with path.open(encoding='utf-8') as f:
   for ln,line in enumerate(f,1):
    if not line.endswith('\n'):partial+=1;continue
    try:d=json.loads(line)
    except ValueError as e:bad.append(dict(line=ln,error=str(e)));continue
    count+=1;first=first or compact(d);last=compact(d)
    if name=='actions.jsonl':
     ep=d['execution_epoch'];st=d['step']
     if st!=laststep+1:seqerrors.append([laststep,st])
     laststep=st;elapsed=d['elapsed_seconds'];delta=elapsed-lasttime;lasttime=elapsed
     values=dict(flatten(d));values['derived.step_wall_delta']=delta
     values['derived.clipped']=d['grad_norm']>5
     dr=d.get('daot_rc4',{});route=dr.get('route_funnel',{});groups=route.get('groups',{})
     domain=collections.Counter();tx=collections.Counter()
     for group,v in groups.items():
      parts=group.split(':');domain[':'.join(parts[1:])]+=v.get('total',0);tx[parts[0]]+=v.get('total',0)
     total=sum(domain.values())
     if total:values['derived.U_max_domain_fraction']=max(domain.values())/total;values['derived.U_max_pseudoTX_fraction']=max(tx.values())/total
     values['derived.P_starved']=route.get('before_budget',{}).get('P',{}).get('count',0)>0 and dr.get('partial_count',0)==0
     values['derived.HP_coverage']=(dr.get('hard_count',0)+dr.get('partial_count',0))/max(1,dr.get('u_samples',1))
     values['derived.identity_abs_below_1e-6']=abs(dr.get('weighted_identity',0))<1e-6
     tel=d.get('solver_telemetry') or {}
     disp=tel.get('parameter_displacement_norm',0)
     if disp:values['derived.response_over_full_update_norm']=tel.get('response_norm',0)/disp
     for key,v in values.items():add(stats,key,v);add(epochs.setdefault(ep,{}),key,v)
     for sid in d.get('unlabeled_ids',[]):idcounts[sid]+=1
    elif name=='ir_steps.jsonl':
     ep=d.get('epoch',0)
     for key,v in flatten(d):add(stats,key,v);add(epochs.setdefault(ep,{}),key,v)
    else:
     small.append(d)
     for key,v in flatten(d):add(stats,key,v)
  row['files'][name]=dict(bytes=size,records=count,invalid=bad,partial_lines=partial,first=first,last=last,stats=finish(stats))
  if epochs:row['epochs'][name]={ep:finish(s) for ep,s in epochs.items()}
  if small:row['records'][name]=small
 row['sequence_errors']=seqerrors
 row['U_ID_exposure']=dict(unique=len(idcounts),min=min(idcounts.values()) if idcounts else None,max=max(idcounts.values()) if idcounts else None)
 text=Path(lr['log']).read_text(errors='replace')
 row['stdout']=dict(lines=len(text.splitlines()),anomalies=[dict(line=i,text=s) for i,s in enumerate(text.splitlines(),1) if re.search(r'Traceback|ERROR|WARNING|\bnan\b|\binf\b|OOM|Killed|nonfinite',s,re.I)])
 row['stdout_full']=text
 rows.append(row)
 print(lr['row_id']+' parsed '+str({k:v['records'] for k,v in row['files'].items()}),file=sys.stderr,flush=True)
processes=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  try:
   args=(p/'cmdline').read_bytes().decode(errors='replace')
   if rid in args and ('train_ir' in args or 'train_response' in args):processes.append(dict(pid=int(p.name),argv=args.split('\x00')))
  except (OSError,PermissionError):pass
result=dict(run=rid,observed_at=time.time(),scope='All complete records of every JSONL and full stdout; partial final lines separately counted',active_training_processes=processes,rows=rows)
sys.stdout.buffer.write(gzip.compress(json.dumps(result,ensure_ascii=False,separators=(',',':')).encode(),compresslevel=3))
