"""Fetch only fixed source-run reports/predictions; never IQ,weights or target."""
import argparse,io,json,os,subprocess,tarfile
from pathlib import Path
RUN="20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01"
REMOTE='/home/szu2070436088/2510044040/CV-SincNet/runs/'+RUN
SEED=2026092701
ARMS=('RAW','ALL','EARLY7','LATE7');VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
FILES=('completion.json','initialization.json','source_contract.json','source_frozen.json','resolved_config.json','actual_native_args.json','actual_modules.json','actual_channel.json','source_fold.json','satmix_result.json','ordinary_augmentation_actual.json','satellite_supervision_coverage.json','satmix_steps.jsonl.gz','unit_steps.jsonl.gz','training_sequence.jsonl.gz','epoch_metrics.jsonl','epoch_metrics.csv','metrics_epoch.csv','metrics_epoch.jsonl','step_metrics.jsonl.gz','timing_audit.json','timing_progress.json','timing_actual_modules.json')
def fetch(output,arm=None):
 output=Path(output)
 if output.exists():raise FileExistsError('Preserve existing fetched evidence;read cached artifacts')
 arms=(arm,) if arm else ARMS;names=[]
 if not arm:names+=['completion.json','result.json','request.json','launch.json','queue_state.json','real_checkpoint_smoke.json']
 for a in arms:
  row=a+'-RX1-s'+str(SEED);names+=['logs/'+row+'.stdout.log']+[row+'/source/'+f for f in FILES]+[row+'/source/source_readouts/'+v+'.json' for v in VIEWS]
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
