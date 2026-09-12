"""Read-only per-run identity and progress; no process intervention."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
project=Path('/home/szu2070436088/2510044040/CV-SincNet');root=project/'runs'/a.run_id
state=json.loads((root/'pipeline_state.json').read_text());result={'time':time.time(),'state':state,'processes':{},'artifacts':{}}
for rid,entry in [('dispatcher',state),*state['rows'].items()]:
    pid=entry.get('pid')
    if pid:
        proc=Path('/proc')/str(pid)
        try:
            argv=(proc/'cmdline').read_bytes().decode().split('\0')
            env=dict(s.split('=',1) for s in (proc/'environ').read_bytes().decode().split('\0') if '=' in s)
            status=(proc/'status').read_text();ppid=next(x.split(':')[1].strip() for x in status.splitlines() if x.startswith('PPid:'))
            result['processes'][rid]=dict(pid=pid,ppid=ppid,cwd=os.readlink(proc/'cwd'),argv=argv,gpu=env.get('CUDA_VISIBLE_DEVICES'))
        except FileNotFoundError:result['processes'][rid]=dict(pid=pid,alive=False)
    folder=Path(entry.get('artifact_root',str(root/rid)))
    found={}
    for name in ('initialization.json','completion.json','logs.jsonl','metrics_epoch.jsonl','actions.jsonl','latest_ssdg.pth','final_ssdg.pth'):
        file=folder/name
        if not file.is_file():continue
        item=dict(bytes=file.stat().st_size,mtime=file.stat().st_mtime)
        if name in ('initialization.json','completion.json'):item['content']=json.loads(file.read_text())
        elif name in ('logs.jsonl','metrics_epoch.jsonl','actions.jsonl'):
            with file.open('rb') as f:f.seek(max(0,file.stat().st_size-20000));lines=f.read().splitlines()
            if lines:
                try:last=json.loads(lines[-1]);item['last']={k:last[k] for k in ('step','epoch','execution_epoch','origin_epoch','action','loss','mean_loss','field_evaluations','fusion') if k in last}
                except (ValueError,UnicodeError):item['last']='partial_write'
        found[name]=item
    log=project/'logs'/a.run_id/(rid+'.train.log')
    if log.is_file():
        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-5000));tail=f.read().decode(errors='replace')
        found['train.log']=dict(bytes=log.stat().st_size,mtime=log.stat().st_mtime,tail=tail)
    result['artifacts'][rid]=found
result['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
print(json.dumps(result,indent=2))
