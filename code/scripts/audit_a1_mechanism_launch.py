"""Read-only N607 launch evidence through one bounded SSH session."""
import argparse
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import json, os, subprocess
from pathlib import Path
p=Path('/home/szu2070436088/2510044040/CV-SincNet')
run='a1_mechanism_screen_s392005_20260910_r2'
release=p/'releases/a1_mechanism_screen_75c07e46_r2'
state=json.loads((p/'runs'/run/'pipeline_state.json').read_text())
result={'state':state,'rows':{}}
for name,row in state['rows'].items():
    pid=row['pid']; proc=Path('/proc')/str(pid)
    item={'pid':pid,'exists':proc.exists()}
    if proc.exists():
        argv=(proc/'cmdline').read_bytes().decode().split('\0')[:-1]
        env=dict(x.split('=',1) for x in (proc/'environ').read_bytes().decode().split('\0') if '=' in x)
        item.update(argv=argv,cwd=os.readlink(proc/'cwd'),gpu=env.get('CUDA_VISIBLE_DEVICES'),
                    ppid=int(next(x.split()[1] for x in (proc/'status').read_text().splitlines() if x.startswith('PPid:'))))
        opts={key:argv[i+1] if i+1<len(argv) and not argv[i+1].startswith('--') else None
              for i,key in enumerate(argv) if key.startswith('--')}
        item['scratch_verified']=(opts.get('--from_scratch')=='true' and opts.get('--a1_scratch_only')=='true'
            and '--baseline_ckpt' not in argv and '--teacher_ckpt' not in argv)
        item['source_screen_only']=opts.get('--a1_source_screen_only')=='true'
    log=p/'logs'/run/(name+'.train.log')
    if log.exists():
        lines=log.read_text(errors='replace').splitlines()
        item['log_bytes']=log.stat().st_size
        item['log_tail']=lines[-8:]
        item['initialization_lines']=[x for x in lines if 'scratch' in x.lower() or 'random' in x.lower()][:8]
        item['epoch_lines']=[x for x in lines if '[E' in x][-3:]
    result['rows'][name]=item
check=p/'logs'/run/'execution_check.json'
result['execution_check']=json.loads(check.read_text()) if check.exists() else None
result['dispatcher_tail']=(release/'dispatcher.log').read_text(errors='replace').splitlines()[-10:]
result['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
result['gpu_map']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
print(json.dumps(result))
'''

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-id',default='a1_mechanism_screen_s392005_20260910_r2')
    parser.add_argument('--release-name',default='a1_mechanism_screen_75c07e46_r2')
    args=parser.parse_args()
    if not args.run_id.replace('_','').isalnum() or not args.release_name.replace('_','').isalnum():
        raise ValueError('Invalid run or release name')
    source=REMOTE.replace('a1_mechanism_screen_s392005_20260910_r2',args.run_id).replace(
        'a1_mechanism_screen_75c07e46_r2',args.release_name)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],
                          input=source,text=True,encoding='utf-8',capture_output=True,timeout=45,check=True)
    value=json.loads(result.stdout)
    args.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':value['state']['status'],'rows':len(value['rows']),
        'waiting':value['state']['waiting_rows'],'check_present':value['execution_check'] is not None,
        'dispatcher_tail':value['dispatcher_tail'][-2:]},ensure_ascii=False))
