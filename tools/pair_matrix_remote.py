"""Windows-native release and readback helper for the frozen pair24 manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'configs/phase1_adv3b02_pair24_manifest.json'
SSH=['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607']


def remote(source):
    compile(source,'remote_payload','exec')
    result=subprocess.run(SSH+['python3 -'],input=source.encode('utf-8'),capture_output=True,timeout=120)
    print(result.stdout.decode('utf-8',errors='replace'))
    print(result.stderr.decode('utf-8',errors='replace'))
    result.check_returncode()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['preflight','release','observe'])
    parser.add_argument('--manifest',type=Path,default=MANIFEST)
    args=parser.parse_args()
    m=json.loads(args.manifest.read_text(encoding='utf-8'))
    manifest_relative=str(args.manifest.resolve().relative_to(ROOT).as_posix())
    prefix='import json, pathlib, subprocess, os\nm='+repr(m)+'\nmanifest_relative='+repr(manifest_relative)+'\n'
    if args.action=='preflight':
        remote(prefix+"""
print(json.dumps({k:{'exists':pathlib.Path(m[k]).exists(),'path':m[k]} for k in ('python','checkpoint','dataset','code_root','output_root')}))
subprocess.run([m['python'],'-c','import sys,torch; print(sys.executable,torch.__version__,torch.cuda.is_available())'],check=True)
subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader'],check=True)
""")
    elif args.action=='release':
        if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip():
            raise RuntimeError('Release requires clean committed worktree')
        archive=ROOT.parent/(m['run_id']+'.tar')
        if archive.exists():
            raise FileExistsError(archive)
        subprocess.run(['git','archive','--format=tar','-o',str(archive),'HEAD'],cwd=ROOT,check=True)
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        remote(prefix+"p=pathlib.Path(m['code_root']); assert not p.exists(); assert not pathlib.Path(m['output_root']).exists(); p.parent.mkdir(parents=True,exist_ok=True); assert not pathlib.Path(str(p)+'.tar').exists()\n")
        target=m['code_root']+'.tar'
        subprocess.run(['scp','-F','E:/type10-7/tools/n607_ssh_config',str(archive),'N607:'+target],check=True)
        remote(prefix+'expected='+repr(digest)+'\n'+"""
import hashlib,tarfile
p=pathlib.Path(m['code_root']); archive=pathlib.Path(str(p)+'.tar')
actual=hashlib.sha256(archive.read_bytes()).hexdigest(); assert actual==expected
p.mkdir()
with tarfile.open(archive) as bundle:
    for member in bundle.getmembers():
        assert (p/member.name).resolve().is_relative_to(p.resolve())
        assert not member.issym() and not member.islnk()
    bundle.extractall(p)
files=['tools/pair_matrix_dispatch.py','tools/pair_matrix_start.py','tools/pair_matrix_manifest.py','code/scripts/smoke_adv3b02_pair_reform.py','code/SSDG/train_ssdg.py','code/cvsrffi/pair_reform.py','code/cvsrffi/pair_reform_runtime.py']
subprocess.run([m['python'],'-m','py_compile']+files,cwd=p,check=True)
saved=json.loads((p/manifest_relative).read_text()); assert saved==m
print(json.dumps({'release':'VERIFIED','sha256':actual,'rows':len(saved['rows'])}))
""")
    else:
        remote(prefix+"""
from collections import Counter
root=pathlib.Path(m['output_root']); state=json.loads((root/'status.json').read_text())
rows=[]
for row in state['rows']:
    r={k:row.get(k) for k in ('row_id','pid','gpu','status','exit_code')}
    pid=row.get('pid'); proc=pathlib.Path('/proc')/str(pid)
    r['alive']=proc.exists()
    r['cwd']=str((proc/'cwd').resolve()) if r['alive'] else None
    log=pathlib.Path(row['log']); r['log_bytes']=log.stat().st_size
    metrics=pathlib.Path(row['runroot'])/'metrics_epoch.jsonl'
    r['last_epoch']=None
    if metrics.exists():
        lines=metrics.read_text().splitlines()
        if lines: r['last_epoch']=json.loads(lines[-1]).get('epoch')
    if row['status']=='TECHNICAL_FAILURE': r['tail']=log.read_text(errors='replace')[-2500:]
    rows.append(r)
smoke=root/'checkpoint_smoke.json'
print(json.dumps({'status':state['status'],'dispatcher_pid':state['dispatcher_pid'],'dispatcher_alive':pathlib.Path('/proc',str(state['dispatcher_pid'])).exists(),'smoke':json.loads(smoke.read_text()) if smoke.exists() else state.get('smoke'),'counts':dict(Counter(r['status'] for r in rows)),'unlaunched':len(state['unlaunched_rows']),'rows':rows}))
subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],check=True)
""")


if __name__=='__main__':
    main()
