"""Publish only after the fixed two-row source gate passes; no repeated controls."""
import argparse,hashlib,inspect,io,json,subprocess,tarfile
from pathlib import Path
from experiments.cric_dura_comparison.publish import ssh,CONNECTION
from . import receiver_balance_confirmation as c
from .publish_clean_h_source import REMOTE


def prerelease_source_check(p):
    rule=p['source_check'];s=json.loads(Path(rule['path']).read_text())
    if (s.get('status')!='RECEIVER_BALANCE_SOURCE_COMPLETE' or s.get('run_id')!=rule['run_id'] or s.get('model_seed')!=2026092701
        or s.get('target_reads')!=0 or s.get('U_truth_reads')!=0 or set(s.get('source_scores',{}))!=set(rule['methods'])
        or s.get('reference')!=dict(raw_correct=26552,H3_correct=62148)):raise ValueError('Source incomplete: no confirmation publication')
    passing=[]
    for method in rule['methods']:
        score=s['source_scores'][method]
        if set(score)!={'raw_correct','H3_correct'} or any(type(score[k]) is not int or not 0<=score[k]<=n for k,n in [('raw_correct',27000),('H3_correct',81000)]):
            raise ValueError('Complete source counts required')
        if score['raw_correct']>=26568 and score['H3_correct']>=62109:passing.append(method)
    winner=max(passing,key=lambda m:(s['source_scores'][m]['raw_correct'],s['source_scores'][m]['H3_correct'],-rule['methods'].index(m))) if passing else None
    if winner is None or s.get('source_selected')!=winner:raise ValueError('Source gate/ranking failed: no confirmation publication')
    return winner


def publish(repo,output,context):
    repo=Path(repo);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
    commit=git('rev-parse','HEAD').decode().strip();branch=git('branch','--show-current').decode().strip()
    if git('ls-remote','origin','refs/heads/'+branch).decode().split()[0]!=commit:raise ValueError('Committed actual code must match upstream')
    if git('status','--porcelain','--','experiments/pamf_channel_pl').strip():raise ValueError('Committed confirmation code required')
    ctx=json.loads(Path(context).read_text(encoding='utf-8'));c.previous.validate_context(ctx);project=ctx['project'];release=project+'/releases/'+c.RELEASE
    rt=dict(run_id=c.RUN,project=project,python=ctx['python'],release_root=release,run_root=project+'/runs/'+c.RUN,
        context=release+'/context.json',runtime_file=release+'/runtime.json',code_commit=commit,frozen_sources=ctx['frozen_sources'],allowed_gpus=list(range(8)),launch_owner=c.OWNER)
    c.validate_runtime(rt);archive=out/(c.RELEASE+'.tar.gz')
    if archive.exists():raise FileExistsError('Reconcile prior publication before retry')
    names={p for p in git('ls-files','-z','--','experiments','code','baselines').decode().split('\0') if p and Path(p).suffix in ('.py','.json','.txt','.yaml','.yml','.toml')}
    data=git('archive','--format=tar',commit,'--','experiments','code','baselines')
    with tarfile.open(fileobj=io.BytesIO(data)) as src,tarfile.open(archive,'w:gz') as dst:
        for member in src:
            if member.name not in names:continue
            value=src.extractfile(member).read();item=tarfile.TarInfo(member.name);item.size=len(value);item.mode=0o644;dst.addfile(item,io.BytesIO(value))
        for name,value in {'context.json':Path(context).read_bytes(),'release_commit.txt':(commit+'\n').encode(),
            'experiment.json':git('show',commit+':automation_reports/CV-SincNet/'+c.RUN+'/experiment.json')}.items():
            item=tarfile.TarInfo(name);item.size=len(value);item.mode=0o644;dst.addfile(item,io.BytesIO(value))
    required=[*ctx['frozen_sources'].values(),project+'/runs/'+c.source.RUN+'/complete.json']
    required.extend(c.prefix(project,seed)+'/'+name for seed in c.SEEDS[1:] for name in ('complete.json','final_ssdg.pth'))
    required.extend(path for method in ('SIG1-Sup','SIG1-PL') for seed in c.SEEDS for path in c.reference_score_paths(project,method,seed).values())
    required.extend(v['views_root']+'/manifest.json' for v in ctx['tests'].values())
    package=dict(runtime=rt,required=required,archive_remote=project+'/releases/'+archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),require_immediate_gpu=False,
        source_check=dict(path=project+'/runs/'+c.source.RUN+'/complete.json',run_id=c.source.RUN,methods=list(c.source.METHODS)))
    for name,value in [('runtime.json',rt),('package.json',package)]:(out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    subprocess.run(['scp',*CONNECTION,str(archive),'N607:'+package['archive_remote']],check=True,timeout=60)
    script=REMOTE.replace('experiments.pamf_channel_pl.clean_h_source','experiments.pamf_channel_pl.receiver_balance_confirmation')
    script=script.replace('p=PACKAGE;',inspect.getsource(prerelease_source_check)+'\np=PACKAGE;')
    script=script.replace('release.mkdir()',"winner=prerelease_source_check(p)\nrelease.mkdir()")
    result=ssh(script.replace('PACKAGE',repr(package)));(out/'submit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','pid','code_commit','archive_transfer','remote_compile','available_gpus')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);p.add_argument('--context',required=True)
    a=p.parse_args();publish(a.repo,a.output,a.context)
