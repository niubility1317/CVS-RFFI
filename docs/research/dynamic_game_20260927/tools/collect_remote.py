"""Read-only N607 metadata/epoch-log snapshot; never loads IQ or checkpoints."""
from pathlib import Path
import subprocess,json,gzip
ROOT=Path('E:/type10-7'); OUT=ROOT/'analysis/dynamic_game_audit_20260927'
REMOTE=r'''from pathlib import Path
import json,datetime,gzip,sys,re,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
names=['phase1_native_response_matrix_20260914_r1','phase1_daot_rc4_pure_game_m3_20260917_r2','phase1_adv3b02_xuc_full_s392005_20260913_r1']
keep={'pipeline_state.json','effective_matrix.json','completion.json','initialization.json','resolved_config.json','resolved_dr_config.json','activation_status.json','logs.jsonl','metrics_epoch.jsonl','training_history.json','joint_probes.jsonl','rc4_calibration.jsonl','run_summary.json','score.json','training_state.json','phase1_terminal_status.json'}
out={'read_at':datetime.datetime.now().astimezone().isoformat(),'root':str(root),'runs':{},'processes':[]}
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  argv=(proc/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
  if any(n in argv for n in names) and 'python3 -' not in argv:out['processes'].append({'pid':int(proc.name),'argv':argv,'cwd':str((proc/'cwd').resolve())})
 except (OSError,PermissionError):pass
for n in names:
 p=root/'runs'/n
 d={'files':{},'artifacts':[],'stdout_scans':[]}
 for f in p.rglob('*'):
  if not f.is_file():continue
  rel=str(f.relative_to(p)); size=f.stat().st_size
  if f.suffix in ('.pth','.pt') or f.name.startswith('score') or 'prediction' in f.name:d['artifacts'].append({'path':rel,'bytes':size})
  if f.name not in keep:continue
  if size>50000000: d['files'][rel]={'omitted':'over 50 MB; inspect separately','bytes':size};continue
  txt=f.read_text(encoding='utf-8')
  if f.suffix=='.jsonl':
   rows=[];errors=[]
   for i,l in enumerate(txt.splitlines(),1):
    if l.strip():
     try:rows.append(json.loads(l))
     except Exception as e:errors.append({'line':i,'error':str(e)})
   d['files'][rel]={'bytes':size,'records':rows,'parse_errors':errors}
  else:
   try:d['files'][rel]={'bytes':size,'data':json.loads(txt)}
   except Exception as e:d['files'][rel]={'bytes':size,'parse_error':str(e),'text':txt}
 lp=root/'logs'/n
 if lp.exists():
  for f in lp.rglob('*.log'):
   size=f.stat().st_size;matches=[];lines=0
   with f.open(encoding='utf-8',errors='replace') as h:
    for lines,l in enumerate(h,1):
     if re.search(r'Traceback|Error:|Exception|out of memory|Killed|non.?finite|\bNaN\b|\bInf\b|warning',l,re.I):matches.append({'line':lines,'text':l.rstrip()[:2500]})
   d['stdout_scans'].append({'path':str(f),'bytes':size,'lines':lines,'matches':matches})
 out['runs'][n]=d
sys.stdout.buffer.write(gzip.compress(json.dumps(out,ensure_ascii=False).encode(),compresslevel=6))
'''
def main():
 p=subprocess.run(['ssh','-F',str(ROOT/'tools/n607_ssh_config'),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=REMOTE.encode('utf-8'),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
 if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
 d=json.loads(gzip.decompress(p.stdout)); path=OUT/'remote_evidence_20260927.json.gz';path.write_bytes(p.stdout)
 print('saved',str(path),'bytes',path.stat().st_size,'time',d['read_at'])
 print('matching_processes',d['processes'])
 for n,rd in d['runs'].items():
  complete=[k for k in rd['files'] if k.endswith('completion.json')]
  print(n,'complete',len(complete),'files',len(rd['files']),'stdout',len(rd['stdout_scans']),'artifacts',len(rd['artifacts']))
  print('completion',[(k,rd['files'][k].get('data')) for k in complete])
if __name__=='__main__':main()
