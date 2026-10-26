"""Publish the fixed F-only three-seed confirmation, preserving existing outputs."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cric_dura_comparison.publish import ssh,CONNECTION
from . import clean_h_confirmation as c
from .clean_h_source import RUN as SOURCE_RUN
from .clean_h_evaluation import BASELINE_RUN
from .sig1_matched_baselines import prefix
from .publish_clean_h_source import REMOTE


def publish(repo,output,context):
    repo=Path(repo);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Committed code must match upstream')
    if git('status','--porcelain','--','experiments/pamf_channel_pl').strip():raise ValueError('Committed confirmation implementation required')
    ctx=json.loads(Path(context).read_text(encoding='utf-8'));c.validate_context(ctx);project=ctx['project'];release=project+'/releases/'+c.RELEASE
    rt=dict(run_id=c.RUN,project=project,python=ctx['python'],release_root=release,run_root=project+'/runs/'+c.RUN,
        context=release+'/context.json',runtime_file=release+'/runtime.json',code_commit=commit,frozen_sources=ctx['frozen_sources'],
        allowed_gpus=list(range(8)),launch_owner='codex-pamf-clean-h-confirm-20261027')
    c.validate_runtime(rt);archive=out/(c.RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile prior release before retry')
    names={p for p in git('ls-files','-z','--','experiments','code','baselines').decode().split('\0') if p and Path(p).suffix in ('.py','.json','.txt','.yaml','.yml','.toml')}
    data=git('archive','--format=tar',commit,'--','experiments','code','baselines')
    with tarfile.open(fileobj=io.BytesIO(data)) as src,tarfile.open(archive,'w:gz') as dst:
        for member in src:
            if member.name not in names:continue
            value=src.extractfile(member).read();item=tarfile.TarInfo(member.name);item.size=len(value);item.mode=0o644;dst.addfile(item,io.BytesIO(value))
        for name,value in {'context.json':Path(context).read_bytes(),'release_commit.txt':(commit+'\n').encode(),
            'experiment.json':git('show',commit+':automation_reports/CV-SincNet/'+c.RUN+'/experiment.json')}.items():
            item=tarfile.TarInfo(name);item.size=len(value);item.mode=0o644;dst.addfile(item,io.BytesIO(value))
    required=[*ctx['frozen_sources'].values(),project+'/runs/'+SOURCE_RUN+'/complete.json']
    required.extend(prefix(project,seed)+'/'+name for seed in c.SEEDS[1:] for name in ('complete.json','final_ssdg.pth'))
    for method in ('SIG1-Sup','SIG1-PL'):
        for seed in c.SEEDS:
            rid='F0-'+method+'-s'+str(seed);base=project+'/runs/'+BASELINE_RUN
            required.extend([base+'/F/'+rid+'/source/complete.json',base+'/F/'+rid+'/source/final_ssdg.pth',base+'/readouts/'+rid+'/scores/scores.json'])
    required.extend(v['views_root']+'/manifest.json' for v in ctx['tests'].values())
    package=dict(runtime=rt,required=required,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    for name,value in [('runtime.json',rt),('package.json',package)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    script=REMOTE.replace('experiments.pamf_channel_pl.clean_h_source','experiments.pamf_channel_pl.clean_h_confirmation')
    result=ssh(script.replace('PACKAGE',repr(package)))
    (out/'submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','pid','code_commit','archive_transfer','remote_compile','available_gpus')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args();publish(a.repo,a.output,a.context)
