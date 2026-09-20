"""Deployed immutable actions. inspect is read-only; start is exclusive."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def identity(pid):
    p=Path('/proc')/str(pid)
    try:return dict(pid=pid,cwd=os.readlink(p/'cwd'),command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:return None


def main():
    if sys.argv[1]=='start':
        assert not (ROOT/'launch.lock').exists() and not (ROOT/'state.json').exists()
        m=json.loads((ROOT/'manifest.json').read_text())
        assert Path(m['remote_root']).resolve()==ROOT
        for r in m['rows']:assert not Path(r['output_root']).exists()
        for rx in range(6):
            for view in ['clean','sg']:assert (Path(m['data_root'])/f'rx_{rx}_{view}_x.npy').is_file()
        with (ROOT/'controller.log').open('xb') as f:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'dispatch.py')],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(submitted_pid=p.pid)))
    elif sys.argv[1]=='inspect':
        state=json.loads((ROOT/'state.json').read_text()) if (ROOT/'state.json').exists() else {}
        result=dict(time=time.time(),state=state,controller=identity(state.get('owner_pid',-1)),rows=[])
        for r in state.get('rows',[]):
            p=Path(r['output_root']);log=Path(r.get('log',ROOT/'absent'))
            epochs=p/'metrics_epoch.jsonl';batches=p/'batches.jsonl'
            item=dict(method=r['method'],status=r['status'],identity=identity(r.get('pid',-1)),
                log_size=log.stat().st_size if log.exists() else 0,
                log_tail=log.read_text(errors='replace')[-2500:] if log.exists() else '',
                resolved_config=(p/'resolved_config.json').exists(),complete=(p/'COMPLETE.json').exists(),
                epochs=sum(1 for _ in epochs.open()) if epochs.exists() else 0,
                batches=sum(1 for _ in batches.open()) if batches.exists() else 0)
            if batches.exists():
                with batches.open() as f:
                    last=None
                    for line in f:last=line
                item['last_batch']=json.loads(last) if last else None
            result['rows'].append(item)
        result['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
