"""Read-only inventory of registered native comparison artifacts and failures."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def inventory(spec_path):
    spec=read(spec_path); runtime=Path(spec['runtime_root']); state=read(runtime/'state.json')
    rows=[]
    for row in spec['rows']:
        out=Path(row['output_root']); entry={'row_id':row['row_id'],'stage':row['stage'],'method':row['method'],
            'output_root':str(out),'state':state['rows'][row['stage']+':'+row['row_id']]['status']}
        if row['stage']=='source':
            flag=out/'completion.json'; value=read(flag) if flag.exists() else {}
            entry.update(source_epoch=value.get('epoch'),source_status=value.get('status'),
                final_checkpoint_exists=(out/'last.pt').is_file())
        else:
            for stage in ('phase1','phase2'):
                path=out/(stage+'_complete.json'); value=read(path) if path.exists() else {}
                entry[stage+'_status']=value.get('status');entry[stage+'_records']=value.get('count',value.get('predictions'))
            entry['support_smoke_passed']=read(out/'real_checkpoint_smoke.json').get('passed') if (out/'real_checkpoint_smoke.json').exists() else None
            if entry['state']=='FAILED':
                log=Path(row['log']); last=[]
                with log.open(encoding='utf-8',errors='replace') as f:
                    for line in f:
                        last.append(line.rstrip());last=last[-24:]
                entry['failure_tail']=last
                entry['failure_signature']=next((line for line in reversed(last) if line.startswith(('TypeError:','ValueError:','RuntimeError:','FloatingPointError:'))),'UNKNOWN')
        rows.append(entry)
    source=[r for r in rows if r['stage']=='source'];pipeline=[r for r in rows if r['stage']=='phase12']
    if any(r['state']=='TRAINING_COMPLETE' and (r['source_epoch']!=200 or r['source_status']!='SOURCE_TRAINED' or not r['final_checkpoint_exists']) for r in source):
        raise ValueError('Source state/artifact mismatch')
    complete=[r for r in pipeline if r['phase2_status']=='PREDICTIONS_COMPLETE']
    return dict(captured_at=datetime.now(timezone.utc).isoformat(),read_only=True,dispatcher_state=state,
        counts=dict(Counter(r['stage']+':'+r['state'] for r in rows)),
        phase1_complete=sum(r['phase1_status']=='PREDICTIONS_COMPLETE' for r in pipeline),
        phase2_complete=len(complete),phase2_complete_prediction_records=sum(r['phase2_records'] for r in complete),
        failure_signatures=dict(Counter(r['failure_signature'] for r in pipeline if r['state']=='FAILED')),
        scoring=state['scoring'],rows=rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--remote');p.add_argument('--output',type=Path)
    p.add_argument('--ssh-executable',default='ssh');p.add_argument('--ssh-config',default='E:/type10-7/tools/n607_ssh_config')
    p.add_argument('--remote-python',default='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python')
    a=p.parse_args()
    if a.remote:
        # UTF-8 LF stdin, no remote write, credentials, truth or checkpoint load.
        command=shlex.join([a.remote_python,'-','--spec',a.spec])
        result=subprocess.run([a.ssh_executable,'-F',a.ssh_config,'-T','-o','BatchMode=yes','-o','ConnectTimeout=10',a.remote,command],
            input=Path(__file__).read_bytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        value=json.loads(result.stdout.decode('utf-8'))
    else:value=inventory(a.spec)
    if a.output:
        if a.output.exists():raise FileExistsError(a.output)
        a.output.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:value[k] for k in ['captured_at','counts','phase1_complete','phase2_complete','phase2_complete_prediction_records','failure_signatures','scoring']},ensure_ascii=False))
    else:print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
