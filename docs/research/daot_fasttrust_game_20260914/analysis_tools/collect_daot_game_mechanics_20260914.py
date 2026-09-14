import subprocess,json,gzip
from pathlib import Path
base=Path('E:/type10-7')
code=r'''
import json,math,time
from pathlib import Path
from collections import Counter,defaultdict
root=Path('/home/RESEARCH_USER/2510044040/CV-SincNet/runs/phase1_adv3b02_xuc_full_s392005_20260913_r1')
out={}
def read(p):return json.loads(p.read_text()) if p.exists() else None
for f in root.iterdir():
 if not f.is_dir() or not (f/'actions.jsonl').exists():continue
 d={'config':read(f/'resolved_config.json'),'dr_config':read(f/'resolved_dr_config.json'),'epochs':{},'probes':[],'windows':[],'curriculum':[]}
 epochs={}
 with (f/'actions.jsonl').open() as h:
  for line in h:
   try:a=json.loads(line)
   except json.JSONDecodeError:continue
   ep=a['origin_epoch'];z=epochs.setdefault(ep,{'n':0,'scenes':Counter(),'applied':Counter(),'actions':Counter(),'sums':Counter(),'finite_counts':Counter(),'min_lr':{},'max_lr':{}})
   z['n']+=1;z['scenes'][a['scenario']]+=1;z['applied'][a['scenario']]+=sum(a['selected_mask']);z['actions'][a['action']]+=1
   dr=a.get('daot_rc4',{})
   vals={'grad_norm':a['grad_norm'],'loss':a['loss'],**{'term_'+k:v for k,v in a['terms'].items()},**{'weighted_'+k:v for k,v in dr.get('weighted_components',{}).items()}}
   for k in ['hard_count','partial_count','negative_count','satellite_samples']:vals[k]=dr.get(k,0)
   for k,v in vals.items():
    if isinstance(v,(int,float)) and math.isfinite(v):z['sums'][k]+=v;z['finite_counts'][k]+=1
   for k,v in a.get('learning_rates',{}).items():z['min_lr'][k]=min(z['min_lr'].get(k,v),v);z['max_lr'][k]=max(z['max_lr'].get(k,v),v)
   if 'daot_grad_norm' in dr or 'xu_gradient_cosine' in a.get('fusion',{}):d['probes'].append({'step':a['step'],'epoch':ep,'fusion':a.get('fusion'), 'dr':dr})
 for ep,z in epochs.items():
  z['means']={k:v/z['finite_counts'][k] for k,v in z['sums'].items()};d['epochs'][ep]=z
 for name,key in [('ticket_windows.jsonl','windows'),('legacy_curriculum.jsonl','curriculum')]:
  p=f/name
  if p.exists():
   for line in p.open():
    try:v=json.loads(line)
    except json.JSONDecodeError:continue
    d[key].append({k:z for k,z in v.items() if k not in ['ticket_ids','consumed_ids']})
 out[f.name]=d
print(json.dumps({'read_at':time.time(),'rows':out}))
'''
compile(code,'read_only_mechanics','exec')
p=subprocess.run(['ssh','-F',str(base/'tools/n607_ssh_config'),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=code.encode(),capture_output=True,timeout=240)
if p.returncode:raise RuntimeError(p.stderr.decode('utf-8','replace'))
d=json.loads(p.stdout)
(base/'automation_reports/CV-SincNet/daot_fasttrust_game_comprehensive_20260914/full_mechanics.json.gz').write_bytes(gzip.compress(p.stdout))
print({k:len(v['epochs']) for k,v in d['rows'].items()})
