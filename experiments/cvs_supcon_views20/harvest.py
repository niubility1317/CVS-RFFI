"""Read completed run-owned source artifacts once;no IQ/weights/inference or launch."""
import argparse,io,json,subprocess,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PRIVATE=ROOT/'local_artifacts/F_supcon_view_routing_E20_20261017';OUT=PRIVATE/'final';RUN='20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01'
ROWS={'Q_sat':'Q_sat-RX1-s2026092701','Q_mix':'Q_mix-RX1-s2026092701'}
NAMES=('actual_channel.json','actual_native_args.json','actual_modules.json','additive_audit.json','completion.json','epoch_metrics.csv','epoch_metrics.jsonl','initialization.json','metrics_epoch.csv','metrics_epoch.jsonl','ordinary_augmentation_actual.json','resolved_config.json','satellite_supervision_coverage.json','satmix_result.json','satmix_steps.jsonl.gz','scene_cycle_audit.json','scene_cycle_steps.jsonl.gz','scene_cycle_progress.json','supcon_views_audit.json','supcon_views_progress.json','source_contract.json','source_fold.json','source_frozen.json','step_metrics.jsonl.gz','training_sequence.jsonl.gz','unit_steps.jsonl.gz')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def fetch(arm):
 runtime=read(PRIVATE/'runtime.json');base=runtime['project']+'/runs/'+RUN;row=ROWS.get(arm);names=['completion.json','result.json','request.json','launch.json','queue_state.json','real_checkpoint_smoke.json','control_reuse.json'] if arm=='root' else ['logs/'+row+'.stdout.log']+[row+'/source/'+n for n in NAMES]+[row+'/source/source_readouts/'+v+'.json' for v in VIEWS]
 if arm!='root' and (OUT/row).exists():
  if all((OUT/n).is_file() for n in names) and read(OUT/row/'source/satmix_result.json')['status']=='SOURCE_SUPCON_VIEWS_E20_COMPLETE':return dict(status='CACHED_VERIFIED',row=arm,files=len(names),new_remote_reads=0,NN=0)
  raise FileExistsError('Partial local row preserved;reconcile instead of overwrite')
 if arm=='root' and (OUT/'completion.json').exists():
  if all((OUT/n).is_file() for n in names) and read(OUT/'completion.json')['status']=='ANALYZED':return dict(status='CACHED_VERIFIED',row=arm,files=len(names),new_remote_reads=0,NN=0)
  raise FileExistsError('Partial local root preserved;reconcile instead of overwrite')
 c=dict(base=base,row=row,names=names)
 script=r'''
import io,json,sys,tarfile
from pathlib import Path
c=CONFIG;base=Path(c['base']);row=c['row']
if row is None:
 q=json.loads((base/'completion.json').read_text())
 if q['status']!='ANALYZED' or q['logical_rows']!=4 or q['new_rows']!=2 or (base/'failure.json').exists():raise ValueError('Original source matrix not complete orfailed')
else:
 result=json.loads((base/row/'source/satmix_result.json').read_text());launch=json.loads((base/'launch.json').read_text());owned=next(z for z in launch['rows'] if z['row_id']==row)
 if result['status']!='SOURCE_SUPCON_VIEWS_E20_COMPLETE' or result['actual_steps']!=4440 or result['target_access'] or (Path('/proc')/str(owned['pid'])).exists():raise ValueError('Original row not terminal/complete;wait,no relaunch')
for name in c['names']:
 if not (base/name).is_file():raise FileNotFoundError(name)
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for name in c['names']:t.add(base/name,arcname=name,recursive=False)
sys.stdout.buffer.write(buf.getvalue())
'''.replace('CONFIG',repr(c))
 cmd=['ssh','-F',runtime['ssh_config'],'-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607',runtime['python']+' -'];r=subprocess.run(cmd,input=script.encode('utf8'),capture_output=True,timeout=45)
 if r.returncode:raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
 with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:
  members=t.getmembers()
  if {m.name for m in members}!=set(names) or any(not m.isfile() or not(OUT/m.name).resolve().is_relative_to(OUT.resolve()) for m in members):raise ValueError('Unexpected local evidence paths')
  OUT.mkdir(parents=True,exist_ok=True);t.extractall(OUT)
 if not all((OUT/n).is_file() for n in names):raise ValueError('Evidence retrieval incomplete;preserve')
 result=dict(status='VERIFIED',row=arm,files=len(names),compressed_bytes=len(r.stdout),real_IQ=0,checkpoint_loads=0,NN=0,updates=0,target_access=False,read_unix=time.time());(PRIVATE/('harvest_'+arm+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--row',required=True,choices=(*ROWS,'root'));a=p.parse_args();print(json.dumps(fetch(a.row)))
