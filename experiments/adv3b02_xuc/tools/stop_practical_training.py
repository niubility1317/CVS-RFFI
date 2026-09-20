"""User-authorized stop of the old practical-four run only; preserve artifacts."""
import argparse,json,subprocess
from pathlib import Path
REMOTE=r'''
import json,os,signal,time
from pathlib import Path
run=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20260918-phase1-daot-rc4-practical4-manysig-s392005-r03')
result={'time':time.time(),'stop_requested':STOP,'rows':[]}
for receipt in sorted(run.glob('launch_*.json')):
    row=json.loads(receipt.read_text());pid=int(row['pid']);proc=Path('/proc')/str(pid)
    output=Path(row['output']);entry={'variant':row['variant'],'pid':pid,'gpu':row['gpu'],'output':str(output)}
    if proc.exists():
        argv=[s for s in (proc/'cmdline').read_bytes().decode().split('\0') if s]
        cwd=os.readlink(proc/'cwd')
        valid=(argv==row['argv'] and cwd==row['cwd'] and proc.stat().st_uid==os.getuid()
            and output.parent==run and '--run-id' in argv and argv[argv.index('--run-id')+1]==run.name)
        entry.update(argv=argv,cwd=cwd,verified_owner=valid)
        if not valid:raise RuntimeError('Process identity changed; no stop')
        group=os.getpgid(pid)
        if group!=pid:raise RuntimeError('Unexpected process group; no stop')
        members=[]
        for candidate in Path('/proc').iterdir():
            if not candidate.name.isdigit():continue
            try:
                if os.getpgid(int(candidate.name))==group:
                    if candidate.stat().st_uid!=os.getuid() or os.readlink(candidate/'cwd')!=cwd:
                        raise RuntimeError('Unexpected group member; no stop')
                    members.append(int(candidate.name))
            except (FileNotFoundError,ProcessLookupError):pass
        entry['group_members']=members
        if STOP:
            os.killpg(group,signal.SIGTERM)
            for _ in range(50):
                if not proc.exists() or not (proc/'cmdline').read_bytes():break
                time.sleep(.1)
    entry['alive_after']=proc.exists() and bool((proc/'cmdline').read_bytes())
    entry['checkpoints']=[{'name':p.name,'bytes':p.stat().st_size,'mtime':p.stat().st_mtime}
        for p in sorted(output.glob('*.pth'))]
    entry['initialization']=json.loads((output/'initialization.json').read_text())
    result['rows'].append(entry)
print(json.dumps(result))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--stop',action='store_true');p.add_argument('--output',required=True)
    a=p.parse_args();script=REMOTE.replace('STOP',repr(a.stop));compile(script,'remote','exec')
    result=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','N607','python3 -'],input=script.encode(),capture_output=True,check=True)
    data=json.loads(result.stdout);path=Path(a.output)
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)
    print(json.dumps(data,indent=2))

if __name__=='__main__':main()
