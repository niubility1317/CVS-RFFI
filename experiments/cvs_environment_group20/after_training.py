"""One-shot continuation of the registered source run;no training/IQ/CP/target."""
import argparse,io,json,os,subprocess,sys,tarfile,time
from pathlib import Path
from . import harvest
from .analyze import analyze
PY='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
def ssh_args():return ['C:/Program Files/Git/usr/bin/ssh.exe','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607',PY+' -']
def command(script,timeout=30):
 env=dict(os.environ,SystemRoot='C:/Windows',windir='C:/Windows',PATHEXT='.COM;.EXE;.BAT;.CMD');return subprocess.run(ssh_args(),input=script.encode('utf8'),capture_output=True,env=env,timeout=timeout)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
def observation(launch):
 return 'import json,os,time\nfrom pathlib import Path\nB=Path('+repr(harvest.REMOTE)+');initial='+repr(launch)+"\nr=dict(read_at=time.time(),completion=None,failure=None,handles=[])\nfor name in ('completion','failure'):\n p=B/(name+'.json')\n if p.exists():r[name]=json.loads(p.read_text())\nknown=[initial]\np=B/'launch.json'\nif p.exists():known+=json.loads(p.read_text())['rows']\nfor item in known:\n p=Path('/proc',str(item['pid']));h=dict(pid=item['pid'],exists=p.exists(),owned_live=False)\n if p.exists():\n  try:\n   actual=dict(start_ticks=int((p/'stat').read_text().split()[21]),cwd=os.readlink(p/'cwd'),argv=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode())\n   h.update(actual);h['owned_live']=actual['start_ticks']==item['start_ticks'] and actual['cwd']==item['cwd'] and all(s in actual['argv'] for s in item['argv'])\n  except OSError:pass\n r['handles'].append(h)\nprint(json.dumps(r))\n"
def plots(output,evidence):
 # The local bundle has no matplotlib. Use the already available project runtime,
 # and only frozen source analysis JSON plus this plotting program.
 remote=harvest.REMOTE+'/offline_analysis';payload={name:(output/name).read_text(encoding='utf8') for name in ('analysis_summary.json','training_curves.json','source_strata.json')};payload['plot.py']=Path(__file__).with_name('plot.py').read_text(encoding='utf8')
 src='import json,os,runpy,sys\nfrom pathlib import Path\nos.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")\nB=Path('+repr(harvest.REMOTE)+');O=Path('+repr(remote)+');payload='+repr(payload)+"\nif not (B/'completion.json').exists() or (B/'failure.json').exists():raise ValueError('Frozen source matrix required')\nO.mkdir()\nfor name,text in payload.items():(O/name).write_text(text,encoding='utf8')\nsys.argv=[str(O/'plot.py'),'--input',str(O)]\nrunpy.run_path(str(O/'plot.py'),run_name='__main__')\n"
 r=command(src,50);(evidence/'plot.stdout.log').write_bytes(r.stdout);(evidence/'plot.stderr.log').write_bytes(r.stderr)
 if r.returncode:raise RuntimeError('Offline plotting failed;preserve original source evidence and reconcile plot logs')
 script='import sys,tarfile\nfrom pathlib import Path\nO=Path('+repr(remote)+")\nwith tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:\n for p in sorted((O/'figures').iterdir()):\n  if p.is_file():t.add(p,arcname='figures/'+p.name,recursive=False)\n"
 r=command(script,30)
 if r.returncode:raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
 archive=evidence/'source_figures.tar.gz'
 with archive.open('xb') as f:f.write(r.stdout)
 with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:
  members=t.getmembers()
  if len(members)!=11 or any(not m.isfile() or not (output/m.name).resolve().is_relative_to(output.resolve()) or not m.name.startswith('figures/') for m in members):raise ValueError('Expected fivePNG/fivePDF/manifest from ownsource analysis')
  if (output/'figures').exists():raise FileExistsError('Preserve existing figures;reconcile before reuse')
  t.extractall(output)
 return dict(status='CREATED_AND_FETCHED',files=len(members),NN=0,IQ=0,CP=0,target_access=False)
def run(output,evidence,launch):
 if output.exists():raise FileExistsError('Reconcile own existing frozen artifact fetch before continuation')
 evidence.mkdir(parents=True,exist_ok=True);script=observation(launch);started=time.time()
 while True:
  try:
   r=command(script);(evidence/'auto_observation.stderr.log').write_bytes(r.stderr)
   if r.returncode:write(evidence/'auto_analysis_status.json',dict(status='UNKNOWN_OBSERVATION',returncode=r.returncode,no_restart=True));time.sleep(45);continue
   z=json.loads(r.stdout);write(evidence/'auto_observation.json',z)
  except (subprocess.TimeoutExpired,json.JSONDecodeError,OSError) as e:
   write(evidence/'auto_analysis_status.json',dict(status='UNKNOWN_OBSERVATION',error=repr(e),no_restart=True));time.sleep(45);continue
  if z['failure']:raise RuntimeError('Original run has preserved technicalfailure;no restart')
  if z['completion']:
   if z['completion'].get('status')!='ARTIFACTS_COMPLETE' or z['completion'].get('updates')!=17760 or not z['completion'].get('source_only') or z['completion'].get('target_access'):raise ValueError('Fixed source-only whole matrix completion required')
   break
  if not any(p['owned_live'] for p in z['handles']):raise RuntimeError('Original run handles gone without terminal completion;preserve and reconcile')
  write(evidence/'auto_analysis_status.json',dict(status='VERIFIED_WAIT',owned_live=[p['pid'] for p in z['handles'] if p['owned_live']],read_at=z['read_at'],no_restart=True));time.sleep(45)
 write(evidence/'auto_analysis_status.json',dict(status='FETCHING_FROZEN_SOURCE'))
 harvest.fetch(output);write(evidence/'auto_analysis_status.json',dict(status='INDEPENDENT_FULL_RECOUNT'))
 a=analyze(output);write(evidence/'auto_analysis_status.json',dict(status='PLOTTING',qualified=a['qualified'],source_only=True,target_access=False))
 p=plots(output,evidence);result=dict(status='ANALYZED_SOURCE_ONLY',run_id=harvest.RUN,qualified=a['qualified'],source_only=True,target_access=False,NN_analysis=0,IQ_analysis=0,CP_analysis=0,updates_analysis=0,plots=p,watch_wall_seconds=time.time()-started,goal_complete=False);write(evidence/'auto_analysis_status.json',result);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--evidence',required=True,type=Path);p.add_argument('--launch',required=True,type=Path);a=p.parse_args()
 try:run(a.output,a.evidence,json.loads(a.launch.read_text(encoding='utf8')))
 except Exception as e:write(a.evidence/'auto_analysis_status.json',dict(status='POSTPROCESS_FAILED',error=repr(e),no_restart=True,training_untouched=True));raise
