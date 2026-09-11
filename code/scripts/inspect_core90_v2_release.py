"""Read-only post-state capture for the authorized N607 release."""
import argparse
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import json,os,subprocess,sys,time,math
from pathlib import Path
root=Path('/home/szu2070436088/2510044040/CV-SincNet')
run=sys.argv[1];logs=root/'logs'/run
status=json.loads((logs/'queue_status.json').read_text()) if (logs/'queue_status.json').exists() else {}
def proc(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return {'pid':pid,'exists':False}
 try:
  return dict(pid=pid,exists=True,cwd=os.readlink(p/'cwd'),argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1],
      ppid=next(s.split()[1] for s in (p/'status').read_text().splitlines() if s.startswith('PPid:')),
      cuda_visible=next((v for v in (p/'environ').read_bytes().decode().split('\0') if v.startswith('CUDA_VISIBLE_DEVICES=')),None))
 except FileNotFoundError:return {'pid':pid,'exists':False}
records=[]
for row in status.get('active',[])+status.get('failed',[])+status.get('completed',[]):
 out=Path(row['output_dir']);r=dict(row,process=proc(row['pid']))
 r['artifacts']={p.name:p.stat().st_size for p in out.glob('*') if p.is_file()}
 for name in ('resolved_config.json','backend_configuration.json'):
  if (out/name).exists():r[name]=json.loads((out/name).read_text())
 actions=out/'game_actions.jsonl'
 if actions.exists():
  lines=actions.read_text().splitlines();parsed=[json.loads(line) for line in lines]
  r['action_count']=len(lines);r['last_action']=parsed[-1] if parsed else None
  r['rejected_actions']=sum(a.get('accepted') is not True for a in parsed)
  r['nonfinite_losses']=sum(not math.isfinite(a.get('loss',float('nan'))) for a in parsed)
 log=Path(row['log']);r['stdout_tail']=log.read_text()[-2000:] if log.exists() else ''
 records.append(r)
smoke=logs/'scratch_smoke/result.json'
print(json.dumps(dict(captured_unix=time.time(),status=status,owner=proc(status['owner_pid']) if status else None,records=records,
 smoke=json.loads(smoke.read_text()) if smoke.exists() else None,
 dispatcher_log=(logs/'dispatcher.stdout.log').read_text()[-5000:],
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True))))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='core90_game_v2_20260911_r3');p.add_argument('--output',required=True)
    args=p.parse_args()
    result=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607',
        '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-',args.run],input=REMOTE,text=True,encoding='utf-8',capture_output=True,check=True,timeout=45)
    value=json.loads(result.stdout);out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(state=value['status'].get('state'),smoke=value['smoke']['status'] if value['smoke'] else None,
      active=len(value['status'].get('active',[])),pending=len(value['status'].get('pending',[])),failed=len(value['status'].get('failed',[])),
      rows=[dict(run=r['run_id'],pid=r['pid'],exists=r['process']['exists'],actions=r.get('action_count'),tail=r['stdout_tail'][-300:]) for r in value['records']]),indent=2))

if __name__=='__main__':main()
