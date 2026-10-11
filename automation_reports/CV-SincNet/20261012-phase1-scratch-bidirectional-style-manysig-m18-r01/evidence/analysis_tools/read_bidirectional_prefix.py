from pathlib import Path
import sys,json
R=Path('E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011');sys.path.insert(0,str(R))
from experiments.cvs_scratch_bidirectional_style import design as d
from experiments.cvs_phase1_stack.publish import ssh
payload=r'''
import json,sys,time,zlib,math
from pathlib import Path
base=Path(BASE);release=Path(RELEASE);sys.path.insert(0,str(release))
from experiments.cvs_scratch_bidirectional_style import design as d
from experiments.cvs_scratch_bidirectional_style.analyze_source import _style_search
rows=[]
for row in d.rows():
 c=d.config(row);p=base/row['row_id']/'source/step_metrics.jsonl.gz'
 if not p.is_file():continue
 compressed=p.read_bytes();decoder=zlib.decompressobj(31);raw=decoder.decompress(compressed);last=raw.rfind(b'\n')
 lines=raw[:last+1].splitlines() if last>=0 else [];n=0;identity=0;styled=0;positive=0;negative=0;aux=0;changed=0
 for line in lines:
  item=json.loads(line);n+=1;assert item['step']==n
  detail=item['details'];active=n>444
  assert all(math.isfinite(v) for v in item['metrics'].values() if isinstance(v,(int,float)))
  assert bool(detail['mixstyle_steps'])==active
  if active:
   info=_style_search(detail['L_style'],c,n,detail['ramp'])
   identity+=int(info['identity_selected']);styled+=int(not info['identity_selected']);positive+=int(info['positive_selected']);negative+=int(info['negative_selected']);changed+=detail['L_style']['changed_packets']
  expected_aux=active and n%4==0;assert bool(detail['auxiliary_steps'])==expected_aux
  if expected_aux:
   assert detail['multilevel']['fishr_levels']==['time_down','t2.pool'];aux+=1
 rows.append(dict(row_id=row['row_id'],source_prefix_steps=n,identity_replays=identity,styled_replays=styled,positive_replays=positive,negative_replays=negative,changed_packet_layers=changed,auxiliary_steps=aux,gzip_finished=decoder.eof,captured_bytes=len(compressed)))
print(json.dumps(dict(status='VERIFIED_SOURCE_PREFIX',read_at=time.time(),run_id=d.RUN,source_only=True,target_access=False,scope='All complete JSON lines of each captured live source prefix; no final training or performance claim',rows=rows,total_steps=sum(r['source_prefix_steps'] for r in rows),identity_replays=sum(r['identity_replays'] for r in rows),styled_replays=sum(r['styled_replays'] for r in rows),auxiliary_steps=sum(r['auxiliary_steps'] for r in rows))))
'''.replace('BASE',repr(d.BASE.as_posix())).replace('RELEASE',repr(d.PROJECT+'/releases/'+d.RELEASE))
wrapper="import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(payload)+",text=True,capture_output=True)\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)\n"
v=json.loads(ssh(wrapper));folder=R/'automation_reports/CV-SincNet'/d.RUN
p=folder/'evidence/source_prefix_audit.json'
if p.exists():raise FileExistsError('Initial captured prefix retained; use a new evidence name')
d.write(p,v)
print(json.dumps(v))
