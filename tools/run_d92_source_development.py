"""Sequential immutable source export and paired source diagnostic."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True,type=Path);p.add_argument('--commit',required=True)
    a=p.parse_args();spec=json.loads(a.spec.read_text(encoding='utf-8'))
    root=Path(spec['execution']['remote_run_root']);release=Path(__file__).resolve().parents[1]
    root.mkdir(parents=True,exist_ok=False)
    (root/'startup.json').write_text(json.dumps(dict(pid=os.getpid(),argv=sys.argv,cwd=os.getcwd(),python=sys.executable,
        commit=a.commit,launch_owner=spec['execution']['launch_owner'],spec=spec,time=time.time()),indent=2)+'\n',encoding='utf-8')
    for name in ('export_d92_source_development','evaluate_d92_source_development'):
        command=[sys.executable,str(release/'tools'/f'{name}.py'),'--spec',str(a.spec)]
        (root/'state.json').write_text(json.dumps(dict(status='RUNNING',stage=name,command=command))+'\n',encoding='utf-8')
        with (root/f'{name}.log').open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=release,stdout=stream,stderr=subprocess.STDOUT)
            (root/'child.json').write_text(json.dumps(dict(pid=child.pid,argv=command,cwd=str(release)))+'\n',encoding='utf-8')
            code=child.wait()
        if code:
            (root/'state.json').write_text(json.dumps(dict(status='TECHNICAL_FAILURE',stage=name,exit_code=code))+'\n',encoding='utf-8')
            raise RuntimeError(f'{name} failed; artifacts preserved; no automatic retry')
    complete=json.loads((Path(spec['development']['evaluation_output'])/'complete.json').read_text())
    if complete['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or complete['rows']!=180:raise ValueError('Incomplete diagnostics')
    (root/'state.json').write_text(json.dumps(dict(status='SOURCE_DIAGNOSTIC_COMPLETE',commit=a.commit,rows=180))+'\n',encoding='utf-8')


if __name__=='__main__':main()
