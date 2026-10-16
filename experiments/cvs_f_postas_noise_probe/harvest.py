"""Fetch only fixed source-run reports/predictions; never IQ,weights or target."""
import argparse,io,json,os,subprocess,tarfile
from pathlib import Path
RUN="20261017-diagnostic-FcrossRX-postAs-noise-control-fitL-s2026092701-m2-r01"
REMOTE='/home/szu2070436088/2510044040/CV-SincNet/runs/'+RUN
ARMS=('frozen-RAW10-s2026092701','frozen-ANTI-s2026092701')
FILES=('resolved_config.json','input_contract.json','source_labels.json','checkpoint_noquery_smoke.json','model_audit.json','physics.jsonl.gz','predictions.jsonl.gz','observation_audit.json','summary.json','progress.json')
def fetch(output,arm=None):
 output=Path(output)
 if output.exists():raise FileExistsError('Preserve existing fetched evidence;read cached artifacts')
 arms=(arm,) if arm else ARMS;names=[]
 if not arm:names+=['completion.json','request.json','launch.json','queue_state.json']
 for a in arms:
  row=a;names+=['logs/'+row+'.stdout.log']+[row+'/'+f for f in FILES]
 script='import sys,tarfile\nfrom pathlib import Path\nB=Path('+repr(REMOTE)+')\n'
 if not arm:script+="if not (B/'completion.json').is_file() or (B/'failure.json').exists():raise ValueError('Source matrix incomplete or technicallyfailed')\n"
 script+='names='+repr(names)+"\nwith tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:\n for n in names:\n  p=B/n\n  if not p.is_file():raise FileNotFoundError(p)\n  t.add(p,arcname=n,recursive=False)\n"
 compile(script,'source_evidence_fetch','exec');ssh='C:/Program Files/Git/usr/bin/ssh.exe' if os.name=='nt' else 'ssh'
 env=dict(os.environ)
 if os.name=='nt':env.update(SystemRoot='C:/Windows',windir='C:/Windows',PATHEXT='.COM;.EXE;.BAT;.CMD')
 partial=output.parent/(output.name+'.fetch.partial.tar.gz')
 with partial.open('xb') as f:result=subprocess.run([ssh,'-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -'],input=script.encode('utf8'),stdout=f,stderr=subprocess.PIPE,env=env,timeout=60)
 if result.returncode:raise RuntimeError(result.stderr.decode('utf8',errors='replace'))
 with tarfile.open(partial) as t:
  members=t.getmembers()
  if {m.name for m in members}!=set(names):raise ValueError('Incomplete fixed source artifacts')
  for m in members:
   if not m.isfile() or not (output/m.name).resolve().is_relative_to(output.resolve()):raise ValueError('Invalid evidence destination')
  output.mkdir(parents=True);t.extractall(output)
 partial.rename(output.parent/(output.name+'.source_evidence.tar.gz'))
 print(json.dumps(dict(status='FETCHED',source_only=True,rows=list(arms),target_access=False,files=len(names),IQ_or_weights_fetched=False)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();fetch(a.output)
