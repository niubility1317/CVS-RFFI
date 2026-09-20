"""Windows-native short-lived SSH transport; explicit actions, no retry loop."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

P=Path(__file__).resolve().parent
M=json.loads((P/'manifest.json').read_text(encoding='utf-8'))
REMOTE=M['remote_root']
PY='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
SSH=['ssh','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607']


def ssh(command,payload=None):
    r=subprocess.run(SSH+[command],input=payload,capture_output=True,timeout=90)
    if r.returncode:
        raise RuntimeError(r.stderr.decode('utf-8',errors='replace')+'\n'+r.stdout.decode('utf-8',errors='replace'))
    return r.stdout


def main():
    action=sys.argv[1]
    if action=='package':
        archive=P/'release.tar.gz'
        names=sorted([f.name for f in P.glob('*.py')]+['manifest.json','report.md','verification.json','review.json'])
        with tarfile.open(archive,'w:gz') as tar:
            for name in names:tar.add(P/name,arcname=name)
        (P/'release_sha256.txt').write_text(hashlib.sha256(archive.read_bytes()).hexdigest(),encoding='utf-8')
        print('PACKAGED',len(names))
    elif action=='upload':
        archive=P/'release.tar.gz';sha=(P/'release_sha256.txt').read_text()
        remote_archive='/tmp/'+P.name+'.tar.gz'
        # Unique path and no pre-existing release checked before SCP.
        pre='from pathlib import Path\nassert not Path('+repr(REMOTE)+').exists()\nassert not Path('+repr(remote_archive)+').exists()\nprint("ABSENT")\n'
        print(ssh('python3 -',pre.encode()).decode())
        subprocess.run(['scp','-F','E:/type10-7/tools/n607_ssh_config',str(archive),'N607:'+remote_archive],check=True,timeout=60)
        code='''import hashlib,tarfile,py_compile,json
from pathlib import Path
root=Path(REMOTE)
archive=Path(ARCHIVE)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==SHA
root.mkdir(exist_ok=False)
with tarfile.open(archive) as t:
 for member in t.getmembers():
  assert Path(member.name).name==member.name and member.isfile()
 t.extractall(root)
for f in root.glob('*.py'):py_compile.compile(str(f),doraise=True)
print(json.dumps(dict(status='LANDED',root=str(root),sha256=SHA)))
'''
        code=code.replace('REMOTE',repr(REMOTE)).replace('ARCHIVE',repr(remote_archive)).replace('SHA',repr(sha))
        output=ssh(PY+' -',code.encode())
        (P/'landing.json').write_bytes(output);print(output.decode())
    elif action=='smoke':
        import shlex
        cmd='CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 '+PY+' '+shlex.quote(REMOTE+'/verify.py')
        output=ssh(cmd);(P/'remote_smoke.json').write_bytes(output);print(output.decode())
    elif action in ['start','inspect']:
        import shlex
        output=ssh(PY+' '+shlex.quote(REMOTE+'/remote_action.py')+' '+action)
        (P/('remote_'+action+'.json')).write_bytes(output)
        if action=='start':print(output.decode())
        else:
            s=json.loads(output)
            print(json.dumps(dict(time=s['time'],controller=s['controller'],gpu=s['gpu'],rows=[
                {k:v for k,v in r.items() if k!='log_tail'} for r in s['rows']]),ensure_ascii=False))


if __name__=='__main__':main()
