"""Read-only N607 process/artifact verification; run on the experiment host."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root')
    a=p.parse_args()
    root=Path(a.root)
    def read(path):
        return json.loads(path.read_text()) if path.exists() else None
    queue=read(root/'queue_status.json')
    result=dict(root=str(root),queue=queue,smoke=read(root/'scratch_smoke/result.json'),processes=[])
    if queue:
        for record in queue['active']:
            proc=Path('/proc')/str(record['pid'])
            observed=dict(run_id=record['run_id'],pid=record['pid'],expected_gpu=record['gpu'])
            try:
                observed['cwd']=str((proc/'cwd').resolve(strict=True))
                observed['argv']=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
                env=dict(item.split('=',1) for item in (proc/'environ').read_bytes().decode().split('\0') if '=' in item)
                observed['cuda_visible_devices']=env.get('CUDA_VISIBLE_DEVICES')
                observed['ppid']=next(line.split(':',1)[1].strip() for line in (proc/'status').read_text().splitlines() if line.startswith('PPid:'))
                observed['binding_verified']=(observed['cuda_visible_devices']==record['gpu_uuid'] and observed['cwd']==queue['release'] and observed['argv']==record['argv'])
                out=Path(record['output_dir'])
                config=read(out/'resolved_config.json')
                if config: observed['initialization']=dict(from_scratch=config.get('from_scratch'),baseline_ckpt=config.get('baseline_ckpt'),resume=config.get('game_resume'),seed=config.get('seed'),epochs=config.get('epochs'))
                log=Path(record['log'])
                observed['stdout_bytes']=log.stat().st_size if log.exists() else 0
                observed['stdout_tail']=log.read_text(errors='replace').splitlines()[-4:] if log.exists() else []
                observed['completed_epochs']=sum(1 for _ in (out/'logs.jsonl').open()) if (out/'logs.jsonl').exists() else 0
                observed['action_rows']=sum(1 for _ in (out/'game_actions.jsonl').open()) if (out/'game_actions.jsonl').exists() else 0
            except (FileNotFoundError,ProcessLookupError,PermissionError) as error:
                observed['verification_error']=type(error).__name__
            result['processes'].append(observed)
    result['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    result['dispatcher_tail']=(root/'dispatcher.stdout.log').read_text(errors='replace').splitlines()[-12:]
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
