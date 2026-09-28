"""Run committed read-only support analysis after all lanes finish; download summaries."""
import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
from publish_d92_branch_support_probe import FLAGS

ROOT=Path(__file__).resolve().parents[1]
PATHS=['tools/summarize_d92_branch_support_probe.py','code/cvsrffi/__init__.py','code/cvsrffi/d92_branch_support_probe.py']
REMOTE=r'''
import json,os,subprocess,tarfile
from pathlib import Path
c=CONFIG
run=Path(c['run']);release=Path(c['release']);archive=Path(c['archive']);output=Path(c['output'])
done=json.loads((run/'complete.json').read_text())
if done['status']!='SUPPORT_PROBE_COMPLETE' or done['completed_rows']!=8 or done['episodes']!=4800:raise ValueError('Full support run incomplete')
if release.exists() or output.exists():raise FileExistsError('Analysis outputs exist; preserve and reconcile')
with tarfile.open(archive) as tar:
 for m in tar.getmembers():
  if not (release.parent/m.name).resolve().is_relative_to(release.resolve()) or m.issym() or m.islnk():raise ValueError('Unsafe analysis archive')
 tar.extractall(release.parent)
argv=[c['python'],'-u',str(release/'tools/summarize_d92_branch_support_probe.py'),'--spec',c['spec'],'--run-root',str(run),'--output',str(output)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
with (release/'analysis.log').open('x') as stream:
 p=subprocess.Popen(argv,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
 (release/'analysis_process.json').write_text(json.dumps(dict(pid=p.pid,argv=argv,cwd=str(release),commit=c['commit'])))
 rc=p.wait()
if rc:raise RuntimeError('Analysis failed; preserve log '+str(release/'analysis.log'))
summary=json.loads((output/'summary.json').read_text())
if summary['status']!='COMPLETE_SUPPORT_DIAGNOSTIC_VERIFIED':raise ValueError('Analysis artifact not complete')
(output/'analysis_execution.json').write_text(json.dumps(dict(commit=c['commit'],runtime_commit=done['commit'],argv=argv,release=str(release),status='VERIFIED'),indent=2))
print(json.dumps(dict(status=summary['status'],coverage=summary['coverage'],output=str(output))))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',default='configs/d92_branch_support_probe_20260929.json')
    p.add_argument('--analysis-release',default='d92_branch_support_probe_analysis_20260929_r01')
    a=p.parse_args();spec=json.loads((ROOT/a.spec).read_text(encoding='utf-8'))
    name=a.analysis_release
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):raise ValueError('Unsafe analysis release name')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=ROOT,text=True).split()[0]!=commit:raise ValueError('Analysis not pushed')
    if subprocess.check_output(['git','status','--porcelain','--',*PATHS],cwd=ROOT,text=True).strip():raise ValueError('Uncommitted analysis paths')
    base=PurePosixPath(spec['code']['cwd']).parent;release=str(base/name);archive_remote=str(base/(name+'.tar'))
    run=spec['execution']['remote_run_root'];output=run+'/results/support_summary'
    local=Path('E:/type10-7/automation_reports/CV-SincNet')/spec['run_id']/'results/support_summary'
    folder=Path('E:/type10-7/local_artifacts/d92_upgrade_20260928')/name
    if local.exists() or folder.exists():raise FileExistsError('Local analysis evidence exists; reconcile')
    probe='from pathlib import Path\nimport json\nassert not any(Path(p).exists() for p in '+repr([release,archive_remote,output])+')\nd=json.loads(Path('+repr(run+'/complete.json')+').read_text())\nassert d["status"]=="SUPPORT_PROBE_COMPLETE" and d["completed_rows"]==8 and d["episodes"]==4800\n'
    subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=probe.encode(),check=True)
    folder.mkdir(parents=True);archive=folder/(name+'.tar')
    subprocess.run(['git','archive','--format=tar','--prefix='+name+'/','--output='+str(archive),commit,*PATHS],cwd=ROOT,check=True)
    subprocess.run(['scp',*FLAGS,str(archive),'N607:'+archive_remote],check=True)
    cfg=dict(commit=commit,run=run,release=release,archive=archive_remote,output=output,python=spec['code']['environment'],spec=spec['code']['cwd']+'/'+a.spec)
    script=REMOTE.replace('CONFIG',repr(cfg));compile(script,'remote-analysis','exec')
    result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode(),capture_output=True)
    (folder/'analysis.stdout').write_bytes(result.stdout);(folder/'analysis.stderr').write_bytes(result.stderr)
    print(result.stdout.decode());print(result.stderr.decode());result.check_returncode()
    local.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['scp',*FLAGS,'-r','N607:'+output,str(local)],check=True)
    summary=json.loads((local/'summary.json').read_text(encoding='utf-8'))
    if summary['status']!='COMPLETE_SUPPORT_DIAGNOSTIC_VERIFIED' or summary['coverage']['episodes']!=4800:raise ValueError('Downloaded summary mismatch')
    print(json.dumps(dict(status='VERIFIED',summary=str(local/'summary.json'))))


if __name__=='__main__':main()
