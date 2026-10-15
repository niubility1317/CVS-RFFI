"""One ordinary-account, CPU-only source calibration release."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cvs_urban_sigtime.operations import remote,SSH
ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261015-phase1-sigtime-joint160-calibration-manysig-s2026092701-r01'
RELEASE='cvs_joint160_source_calibration_20261015_r01'
SOURCE_ROW='20261015-phase1-sigtime-urban-ssl-e50-manysig-s2026092701-m4-r01/C3-s2026092701/source'

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
c=CONFIG;project=Path(c['project']);release=project/'releases'/c['release'];run=project/'runs'/c['run'];source=project/'runs'/c['source']
read=lambda p:json.loads(p.read_text())
done=read(source/'completion.json');frozen=read(source/'source_frozen.json')
if done['epoch']!=50 or done['optimizer_steps']!=11100 or done['target_access'] or done['head_fit_updates'] or frozen['epoch']!=50 or frozen['selection']!='final_only':raise ValueError('Need source-frozen final50 A; no target result is consumed')
if release.exists() or run.exists():raise FileExistsError('Reconcile existing calibration release/run before retry')
archive=project/'releases'/c['archive']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=c['sha256']:raise ValueError('Single archive transfer comparison failed')
if subprocess.check_output(['whoami'],text=True).strip()!='szu2070436088' or subprocess.check_output(['hostname'],text=True).strip()!='dell-DSS8440':raise ValueError('Ordinary account/host differs')
release.mkdir()
with tarfile.open(archive) as stream:
 for member in stream.getmembers():
  if not member.isfile() or not (release/member.name).resolve().is_relative_to(release.resolve()):raise ValueError('Unsafe archive destination')
 stream.extractall(release)
python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
env=dict(os.environ,PYTHONPATH=str(release)+os.pathsep+str(release/'code'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([python,'-m','compileall','-q',str(release/'experiments/cvs_urban_identity50')],cwd=release,env=env,check=True)
spec=read(release/'experiments/cvs_urban_identity50/registration_calibration.json')
if not Path(spec['data']['contract_ref']).is_file():raise FileNotFoundError('Existing source contract handle missing')
(release/'preflight.json').write_text(json.dumps(dict(status='VERIFIED',archive_transfer='VERIFIED',remote_compile='PASS',ordinary_account=True,cpu_threads=2,CUDA_VISIBLE_DEVICES='',source_ready=True,commit=c['commit'],read_at=time.time()),indent=2)+'\n')
command=[python,'-u','-m','experiments.cvs_urban_identity50.calibrate'];log=release/'calibration.log'
with log.open('x') as stream:child=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
submitted=dict(status='SUBMITTED',pid=child.pid,cwd=str(release),argv=command,log=str(log),run_root=str(run),commit=c['commit'])
(release/'submit.json').write_text(json.dumps(submitted,indent=2)+'\n');print(json.dumps(submitted))
'''

INSPECT=r'''
import json,os,time
from pathlib import Path
c=CONFIG;release=Path(c['project'])/'releases'/c['release'];run=Path(c['project'])/'runs'/c['run']
def read(p):return json.loads(p.read_text()) if p.is_file() else None
submit=read(release/'submit.json');process=None
if submit:
 try:process=dict(pid=submit['pid'],cwd=os.readlink(Path('/proc')/str(submit['pid'])/'cwd'),argv=(Path('/proc')/str(submit['pid'])/'cmdline').read_bytes().decode().strip('\0').split('\0'))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
result=read(run/'results.json');valid_live=bool(process and process['cwd']==str(release) and process['argv']==submit['argv'])
valid_done=bool(result and result['status']=='SOURCE_CALIBRATED' and result['source_only'] and not result['query_read'] and not result['added_training_updates'])
log=release/'calibration.log'
print(json.dumps(dict(status='VERIFIED' if valid_live or valid_done else 'FAILED' if submit and not process else 'UNKNOWN',
 submit=submit,process=process,preflight=read(release/'preflight.json'),result=result,
 log_bytes=log.stat().st_size if log.is_file() else 0,log_mtime=log.stat().st_mtime if log.is_file() else None,
 log_tail=log.read_text(errors='replace')[-2500:] if log.is_file() else None,read_at=time.time())))
'''


def config():return dict(project=PROJECT,release=RELEASE,run=RUN,source=SOURCE_ROW)


def inspect(output):
    value=remote(INSPECT.replace('CONFIG',repr(config())))
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    (path/'readback.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=value['status'],process=value['process'],result={k:value['result'].get(k) for k in ('status','coefficient','forward_packets','seconds')} if value['result'] else None,log_tail=value['log_tail']),ensure_ascii=False));return value


def publish(output):
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--','experiments','code','baselines','comparison_suite'],cwd=ROOT,text=True)
    if dirty.strip():raise ValueError('Commit intended code/config; preserve unrelated changes')
    names=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    names=[n for n in names if n.startswith(('experiments/','code/','baselines/','comparison_suite/')) and Path(n).suffix in ('.py','.json') and '/snapshots/' not in n and '/local_artifacts/' not in n and '/tests/' not in n]
    archive=path/(RELEASE+'.tar.gz')
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for name in names:tar.add(ROOT/name,arcname=name,recursive=False)
            blob=(commit+'\n').encode();entry=tarfile.TarInfo('release_commit.txt');entry.size=len(blob);tar.addfile(entry,io.BytesIO(blob))
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(['scp',*SSH,str(archive),'N607:'+PROJECT+'/releases/'+archive.name],check=True,timeout=60)
    result=remote(REMOTE.replace('CONFIG',repr(dict(config(),archive=archive.name,sha256=sha,commit=commit))))
    (path/'submission.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--inspect',action='store_true');args=parser.parse_args()
    inspect(args.output) if args.inspect else publish(args.output)
