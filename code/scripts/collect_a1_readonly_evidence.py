"""Read complete A1 text evidence via one bounded SSH transfer, without remote writes."""
import argparse
from pathlib import Path
import subprocess

REMOTE = r'''
import io,json,tarfile,datetime,sys,os
from pathlib import Path
p=Path('/home/szu2070436088/2510044040/CV-SincNet')
runs=sorted(x for x in (p/'runs').glob('a1_*') if x.is_dir())
runs.append(p/'runs/tweak_config2_portability_20260909_v5')
names={'pipeline_state.json','effective_matrix.json','launch_config.json','process.json',
       'score.json','evaluation_scope.json','final_metrics.json','summary.json',
       'user_periodic_reconfiguration_stop_result.json','selected_checkpoint.json'}
excluded={'target_inputs','target_truth','execution_check','periodic_execution_check'}
entries=[]
extra_only=EXTRA_ONLY
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
    for run in runs:
        roots=[run,p/'logs'/run.name]
        for root in roots:
            if not root.exists():continue
            for folder,dirs,files in os.walk(root):
                dirs[:]=[d for d in dirs if d not in excluded]
                for name in sorted(files):
                    path=Path(folder)/name
                    wanted=(name in names or name.endswith(('.jsonl','.csv','.log')) or
                            name.startswith('source_eval_epoch_') and name.endswith('.json'))
                    if extra_only:
                        wanted=(not wanted and name.endswith('.json') and 'truth' not in name.lower()
                            and name not in ('predictions.json','manifest.json') and path.stat().st_size<=10000000)
                    if not wanted:
                        continue
                    before=path.stat()
                    data=path.read_bytes()
                    relative=str(path.relative_to(p))
                    info=tarfile.TarInfo(relative.replace('\\','/'))
                    info.size=len(data);info.mtime=before.st_mtime
                    archive.addfile(info,io.BytesIO(data))
                    entries.append({'path':relative,'bytes':len(data),'mtime':before.st_mtime})
    inventory={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'runs':[r.name for r in runs],'files':entries}
    data=json.dumps(inventory,indent=2).encode()
    info=tarfile.TarInfo('inventory.json');info.size=len(data)
    archive.addfile(info,io.BytesIO(data))
'''

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extra-json',action='store_true')
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],
            input=REMOTE.replace('EXTRA_ONLY',str(args.extra_json)).encode(),stdout=stream,stderr=subprocess.PIPE,timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    print(f'COMPLETE_TEXT_EVIDENCE bytes={args.output.stat().st_size} path={args.output}')
