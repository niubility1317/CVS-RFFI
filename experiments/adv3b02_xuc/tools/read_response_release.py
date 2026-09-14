"""Short read-only N607 state probe; persist independent launch evidence."""
import argparse,json,subprocess
from pathlib import Path

PROBE=r'''
import json,os,subprocess,time
from pathlib import Path
project=Path('/home/szu2070436088/2510044040/CV-SincNet');name=RUN_ID
run=project/'runs'/name;logs=project/'logs'/name
state=json.loads((run/'pipeline_state.json').read_text())
def process(pid):
    p=Path('/proc')/str(pid)
    try:return dict(pid=pid,cwd=str((p/'cwd').resolve(strict=True)),argv=(p/'cmdline').read_bytes().decode().split('\0'))
    except FileNotFoundError:return dict(pid=pid,absent=True)
result=dict(time=time.time(),state=state,dispatcher=process(state['pid']),workers={},
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True),
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True),
    smoke=json.loads((project/'logs'/(name+'_preflight')/'acceptance.json').read_text()))
for rid,entry in state['rows'].items():
    if not entry.get('pid'):continue
    log=logs/(rid+'.train.log');lines=log.read_text(errors='replace').splitlines() if log.exists() else []
    actions=run/rid/'actions.jsonl';last=None;count=0
    if actions.exists():
        with actions.open() as h:
            for line in h:
                if line.strip():last=line;count+=1
    cfg=run/rid/'resolved_config.json'
    result['workers'][rid]=dict(process=process(entry['pid']),gpu=entry['gpu'],log_size=log.stat().st_size if log.exists() else 0,
        log_tail=lines[-8:],errors=[x for x in lines if any(t in x for t in ('Traceback','RuntimeError','ValueError','CUDA out of memory'))],
        accepted_records=count,last_step=json.loads(last).get('step') if last else None,
        resolved=json.loads(cfg.read_text()) if cfg.exists() else None)
print(json.dumps(result))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    script=PROBE.replace('RUN_ID',repr(a.run_id))
    result=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607','python3 -'],input=script.encode(),capture_output=True,timeout=45)
    result.check_returncode();data=json.loads(result.stdout);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=data['state']['status'],dispatcher=data['dispatcher'],workers={k:{x:v[x] for x in ('gpu','log_size','accepted_records','errors')} for k,v in data['workers'].items()})))
if __name__=='__main__':main()
