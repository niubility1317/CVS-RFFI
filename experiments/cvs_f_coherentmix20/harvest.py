"""Fetch only fixed source-run reports/predictions; never IQ,weights or target."""
import argparse,io,json,os,subprocess,tarfile
from pathlib import Path
RUN='20261016-phase1-FcrossRX-coherentmix-source-RX1-e20-s2026092702-m3-r02'
REMOTE='/home/szu2070436088/2510044040/CV-SincNet/runs/'+RUN
SEED=2026092702
ARMS=('SAME_RX','CROSS_RX');VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
FILES=('completion.json','initialization.json','source_contract.json','source_frozen.json','resolved_config.json','actual_native_args.json','actual_modules.json','actual_channel.json','source_fold.json','satmix_result.json','additive_audit.json','ordinary_augmentation_actual.json','satellite_supervision_coverage.json','satmix_steps.jsonl.gz','unit_steps.jsonl.gz','training_sequence.jsonl.gz','epoch_metrics.jsonl','epoch_metrics.csv','metrics_epoch.csv','metrics_epoch.jsonl','step_metrics.jsonl.gz','coherent_bank.json','coherent_pairs.jsonl.gz','coherent_progress.json')
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
def select_profile(name):
 global RUN,REMOTE,SEED,ARMS,FILES
 if name=='seed02_initial':return
 if name=='seed01_confirmation':RUN='20261017-phase1-FcrossRX-coherentmix-confirm-source-RX1-e20-s2026092701-m3-r01'
 elif name=='seed01_raw_reference':
  RUN='20261016-phase1-F-crossRX-additiveAs-sourceRX1-e20-s2026092701-m3-r01';ARMS=('D_raw',);FILES=tuple(n for n in FILES if not n.startswith('coherent_'))
 else:raise ValueError('Unregistered source artifact profile')
 SEED=2026092701;REMOTE='/home/szu2070436088/2510044040/CV-SincNet/runs/'+RUN
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--row',choices=('D_raw','SAME_RX','CROSS_RX'));p.add_argument('--profile',choices=('seed02_initial','seed01_confirmation','seed01_raw_reference'),default='seed02_initial');a=p.parse_args();select_profile(a.profile)
 if a.row is not None and a.row not in ARMS:raise ValueError('Row not in selected source artifact profile')
 fetch(a.output,a.row)
