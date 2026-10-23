"""Publish one committed archive; independent post-state is verified separately."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from .plan import RUN
from experiments.cric_dura_comparison.publish import ssh,CONNECTION

ROOT=Path(__file__).resolve().parents[2]

REMOTE=r'''
import hashlib,json,os,subprocess,tarfile
from pathlib import Path
p=PACKAGE;rt=p['runtime'];release=Path(rt['release_root'])
if subprocess.check_output(['whoami'],text=True).strip()!=rt['project'].split('/')[2]:raise ValueError('Ordinary account mismatch')
if release.exists() or Path(rt['run_root']).exists():raise FileExistsError('Reconcile existing release/run before retry')
for value in [*rt['frozen_sources'].values(),*rt['datasets'].values(),rt['old_jobs'],rt['reused_source_handles'],rt['ibrff_model_root'],rt['sdg_model_root']]:
 if not Path(value).exists():raise FileNotFoundError(value)
if hashlib.sha256(Path(p['archive_remote']).read_bytes()).hexdigest()!=p['sha256']:raise ValueError('Archive transfer mismatch')
release.mkdir()
with tarfile.open(p['archive_remote']) as tar:
 for member in tar.getmembers():
  if not member.isfile() or release.resolve() not in (release/member.name).resolve().parents:raise ValueError('Invalid release path')
 tar.extractall(release)
Path(rt['runtime_file']).write_text(json.dumps(rt,indent=2)+'\n')
env=dict(os.environ,PYTHONPATH=str(release),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([rt['python'],'-m','compileall','-q',str(release/'experiments/practical_full_2x2')],cwd=release,env=env,check=True)
cmd=[rt['python'],'-u','-m','experiments.practical_full_2x2.runner','dispatch','--config',rt['runtime_file']]
log=release/'controller.log'
with log.open('x') as stream:
 child=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(status='SUBMITTED_NOT_YET_VERIFIED',pid=child.pid,cwd=str(release),argv=cmd,log=str(log),execution_commit=rt['code_commit'])
(release/'submit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
'''

def publish(repo,output):
    repo=Path(repo);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    def git(*a):return subprocess.check_output(['git','-C',str(repo),*a])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Remote OID not independently verified')
    pre=json.loads((out/'release_preflight.json').read_text());old=pre['runtime']
    native=json.loads((ROOT/'local_artifacts/phase1_dg_sourcecount6_20261021/runtime.json').read_text())
    source=json.loads((ROOT/'local_artifacts/cric_dura_comparison_20261021/runtime.json').read_text())
    templates=json.loads((ROOT/'local_artifacts/external_DURA_readouts_20261023/preflight.json').read_text())['G_templates']
    project=old['project'];release=project+'/releases/'+RUN;run=project+'/runs/'+RUN
    rt=dict(run_id=RUN,project=project,release_root=release,run_root=run,code_commit=commit,
        matrix=release+'/experiment.json',runtime_file=release+'/runtime.json',python=old['python'],external_python=native['python'],
        frozen_sources=old['frozen_sources'],external_releases=[project+'/releases/20261019-phase1-external-dg-concat-pl-m219-r03',native['release_root']],
        ibrff_model_root=native['ibrff_model_root'],sdg_model_root=native['sdg_model_root'],datasets=source['datasets'],
        old_jobs=old['jobs'],reused_source_handles=source['reused_source_handles'],
        original_G_action_cache=old['original_G_action_cache'],original_G_coarse_cache=old['original_G_coarse_cache'],
        old_G_data=project+'/runs/20261021-phase1-T1-DURA-G-lineage-m35-r01/g_data',G_templates=templates,
        observation_unions={d:project+'/runs/20261019-phase1-external-dg-suite-m168-r01/data/observations/'+d for d in ('ManySig','ManyRx')},
        augmentation_cache=run+'/augmentation_cache',augmentation_cpu_workers=8,
        launch_owner='codex-practical-Full2x2-20261023',max_own_per_gpu=3,max_total_compute=15,
        allowed_gpus=[0,1,2,3,4],reserved_gpus=[5,6,7])
    archive=out/(RUN+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile archive/remote post-state before retry')
    paths=git('ls-files','-z','--','experiments','code','baselines').decode().split('\0')
    paths=[p for p in paths if p and Path(p).suffix in ('.py','.json','.txt','.yaml','.yml','.toml')]
    owned=[p for p in paths if p.startswith('experiments/practical_full_2x2/')]
    if not owned or git('status','--porcelain','--',*owned).strip():raise ValueError('Release adapter not clean/committed')
    document='automation_reports/CV-SincNet/'+RUN+'/experiment.json'
    source_archive=git('archive','--format=tar',commit,'--','experiments','code','baselines')
    with tarfile.open(fileobj=io.BytesIO(source_archive)) as prior,tarfile.open(archive,'w:gz') as tar:
        for member in prior:
            if member.name not in paths:continue
            value=prior.extractfile(member).read();name=member.name
            item=tarfile.TarInfo(name);item.size=len(value);item.mode=0o644;tar.addfile(item,io.BytesIO(value))
        value=git('show',commit+':'+document);item=tarfile.TarInfo('experiment.json');item.size=len(value);item.mode=0o644;tar.addfile(item,io.BytesIO(value))
        value=(commit+'\n').encode();item=tarfile.TarInfo('release_commit.txt');item.size=len(value);tar.addfile(item,io.BytesIO(value))
    package=dict(runtime=rt,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (out/'runtime.json').write_text(json.dumps(rt,indent=2)+'\n',encoding='utf-8')
    (out/'package.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    result=ssh(REMOTE.replace('PACKAGE',repr(package)))
    (out/'submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(result,archive_bytes=archive.stat().st_size)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();publish(a.repo,a.output)
