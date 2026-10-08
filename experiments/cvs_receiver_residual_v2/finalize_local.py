"""One-shot completion sync for this run; no training intervention or tuning."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from experiments.cvs_receiver_residual_v2 import design as d
from experiments.cvs_receiver_residual_v2.publish import inspect,ssh

ARTIFACT=d.ROOT/'local_artifacts'/d.RELEASE
REPORT=d.ROOT/'automation_reports/CV-SincNet'/d.RUN
REGISTRY=Path('E:/type10-7/tools/experiment_registry.py')
BRANCH='codex/receiver-residual-dg-20261008'


def command(argv):
    return subprocess.run(argv,cwd=d.ROOT,check=True,capture_output=True,text=True,encoding='utf-8').stdout


def record(status,note):
    command([sys.executable,'-X','utf8',str(REGISTRY),'--root',str(d.ROOT),'record',d.RUN,
        '--status',status,'--evidence','local_artifacts/'+d.RELEASE+'/readback.json','--note',note])


def pull():
    code=r'''
import io,sys,tarfile
from pathlib import Path
base=Path(BASE)
names=['completion.json','failure.json','queue_state.json','dispatcher.json','source_matrix_frozen.json',
    'scoring_complete.json','scores.json','scores.csv','summary.json','summary.csv',
    'paired_results.json','resources.json','analysis.md']
for row in ROWS:
    names += [row+'/source/'+n for n in ('completion.json','initialization.json','resolved_config.json',
        'source_selection.json','resource_profile.json','epoch_metrics.jsonl','epoch_metrics.csv','centroid_diagnostics.jsonl')]
    names += [row+'/prediction/'+n for n in ('provenance.json','resolved_config.json','complete.json')]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
    for n in names:
        p=base/n
        if p.is_file():tar.add(p,arcname=n,recursive=False)
'''
    code=code.replace('BASE',repr(d.PROJECT+'/runs/'+d.RUN)).replace('ROWS',repr([d.row_id(c['arm'],c['model_seed']) for c in d.rows()]))
    blob=ssh(code);(ARTIFACT/'results.tar.gz').write_bytes(blob)
    folder=ARTIFACT/'results';folder.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for member in tar.getmembers():
            if not member.isfile() or not (folder/member.name).resolve().is_relative_to(folder.resolve()):
                raise ValueError('Unsafe result archive member')
        tar.extractall(folder)
    return folder


def mirror_root():
    target=Path('E:/type10-7/automation_reports/CV-SincNet')/d.RUN
    if (target/'experiment.json').exists():
        prior=d.read(target/'experiment.json');actual=d.read(REPORT/'experiment.json')
        if prior['run_id']!=actual['run_id'] or prior['execution']['launch_owner']!=actual['execution']['launch_owner']:
            raise ValueError('Canonical report belongs to another owner')
    target.mkdir(parents=True,exist_ok=True)
    for path in REPORT.rglob('*'):
        if path.is_file():
            output=target/path.relative_to(REPORT);output.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix=='.md':
                output.write_text(path.read_text(encoding='utf-8').replace('../../../experiments/',d.ROOT.joinpath('experiments').as_posix()+'/'),encoding='utf-8')
            else: shutil.copyfile(path,output)
    # Update only this entry, preserving all other canonical catalog entries/edits.
    row=next(json.loads(s) for s in (d.ROOT/'experiment_registry/catalog.jsonl').read_text(encoding='utf-8').splitlines()
        if json.loads(s).get('id')==d.RUN)
    canonical=Path('E:/type10-7/experiment_registry/catalog.jsonl')
    if canonical.exists():
        values=[json.loads(s) for s in canonical.read_text(encoding='utf-8').splitlines() if s.strip()]
        values=[r for r in values if r.get('id')!=d.RUN]+[row]
        canonical.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values),encoding='utf-8',newline='\n')
    csvpath=Path('E:/type10-7/experiment_registry/catalog.csv')
    if csvpath.exists():
        with (d.ROOT/'experiment_registry/catalog.csv').open(encoding='utf-8-sig',newline='') as f:
            new=next(r for r in csv.DictReader(f) if r['id']==d.RUN)
        with csvpath.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;old=list(reader)
        with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            writer.writerows([r for r in old if r['id']!=d.RUN]+[new])
    for method in row['tags']+row['methods']:
        path=d.ROOT/'experiment_registry/by_method'/(method+'.md')
        if path.exists():
            destination=Path('E:/type10-7/experiment_registry/by_method')/path.name
            text='\n- ['+row['name']+'](../../automation_reports/CV-SincNet/'+d.RUN+'/report.md) — '+row['status']+'\n'
            if destination.exists():
                existing=destination.read_text(encoding='utf-8')
                if d.RUN in existing:
                    lines=existing.splitlines()
                    destination.write_text('\n'.join(text.strip() if d.RUN in line else line for line in lines)+'\n',encoding='utf-8')
                else:
                    with destination.open('a',encoding='utf-8') as f:f.write(text)
            else: destination.write_text('# '+method+'\n'+text,encoding='utf-8')


def git_delivery():
    if command(['git','branch','--show-current']).strip()!=BRANCH: raise ValueError('Delivery branch changed')
    command(['git','status','-sb'])
    command(['git','-c','core.safecrlf=false','add','-f','--',str(REPORT),
        'experiment_registry','local_artifacts/'+d.RELEASE+'/readback.json'])
    command(['git','diff','--cached','--check'])
    if command(['git','diff','--cached','--name-only']).strip():
        # A watcher owns report/index only; never absorb a concurrently staged code edit.
        names=command(['git','diff','--cached','--name-only']).splitlines()
        if any(not n.startswith(('automation_reports/CV-SincNet/'+d.RUN+'/',
            'experiment_registry/','local_artifacts/'+d.RELEASE+'/readback.json')) for n in names):
            raise ValueError('Unrelated staged edits; preserve, do not commit them')
        command(['git','commit','-m','Record receiver residual R2 final independent results'])
    command(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','push','origin','HEAD'])
    head=command(['git','rev-parse','HEAD']).strip()
    remote=command(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/'+BRANCH]).split()[0]
    if head!=remote: raise ValueError('Remote delivery OID differs')
    d.write(ARTIFACT/'final_git_delivery.json',dict(status='VERIFIED',commit=head,remote_oid=remote))


def finalize(data):
    folder=pull();done=d.read(folder/'completion.json');scored=d.read(folder/'scoring_complete.json')
    if (done.get('status')!='ANALYZED' or done.get('rows')!=16 or done.get('views')!=7
        or scored.get('status')!='SCORED_COMPLETE' or scored.get('result_rows')!=1568
        or scored.get('independent_recount')!='VERIFIED' or scored.get('truth_last') is not True):
        raise ValueError('Independent final artifact coverage incomplete')
    summary=d.read(folder/'summary.json')['results'];pairs=d.read(folder/'paired_results.json')['comparisons']
    if len(summary)!=28 or len(pairs)!=28 or any(r['seed_count']!=4 for r in summary):
        raise ValueError('Summary seed/view coverage differs')
    results=REPORT/'results';results.mkdir(exist_ok=True)
    for name in ('summary.json','summary.csv','paired_results.json','resources.json','scoring_complete.json','completion.json','source_matrix_frozen.json'):
        shutil.copyfile(folder/name,results/name)
    evidence=REPORT/'evidence';evidence.mkdir(exist_ok=True)
    shutil.copyfile(ARTIFACT/'readback.json',evidence/'final_readback.json')
    path=REPORT/'report.md';text=path.read_text(encoding='utf-8')
    text += '\n## 冻结后独立测试结果\n\n状态ANALYZED，16/16源模型均E200×50=10000步；全部16×7预测完成后truth-last独立评分，1568条overall/RX/TX结果，独立重算VERIFIED。R1随机性混杂不作为此轮收益依据。\n\n'
    text += '|组|视图|Accuracy均值±样本SD|最差RX均值|\n|---|---|---:|---:|\n'
    for r in summary:
        text += f"|{r['arm']}|{r['view']}|{100*r['accuracy_mean']:.3f}±{100*r['accuracy_sd']:.3f}%|{100*r['worst_rx_mean']:.3f}%|\n"
    text += '\n配对差值、交互、Macro-F1和资源见results。测试性能不回流调参、选模、seed排除或重跑；所有负结果保留。Phase2/K/新类/H不适用。既有暴露benchmark，六场景是完整paired实用residual视图。\n'
    for r in pairs:
        if r['view']=='clean':text += f"\nclean {r['contrast']}：{r['mean_pp']:+.3f}个百分点（四seed样本SD {r['sd_pp']:.3f}）。\n"
    path.write_text(text,encoding='utf-8',newline='\n')
    spec=d.read(REPORT/'experiment.json');spec['code']['runtime_commit']=data['submit']['commit']
    spec['results_ref']='results/summary.json';d.write(REPORT/'experiment.json',spec)
    record('ANALYZED','VERIFIED all16 E200 source, all112 fullview predictions, independent truth-last scoring/recount; no target feedback')
    command([sys.executable,'-X','utf8',str(REGISTRY),'--root',str(d.ROOT),'build','--managed-only'])
    mirror_root();git_delivery()


def watch():
    ARTIFACT.mkdir(exist_ok=True)
    with (ARTIFACT/'completion_sync_owner.json').open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,run_id=d.RUN,started=time.time()),f)
    errors=0
    while True:
        try:
            data=inspect(ARTIFACT);errors=0
            if data['failure']:
                pull();record('FAILED','Remote technical failure; artifacts retained; no retry or training intervention')
                mirror_root();return
            if data['completion']:
                finalize(data);return
            if data['submit'] and not data['dispatcher_process']:
                d.write(ARTIFACT/'completion_sync_attention.json',dict(status='UNKNOWN',reason='Dispatcher absent without terminal artifact'))
                return
        except Exception as e:
            errors+=1
            d.write(ARTIFACT/'completion_sync_error.json',dict(status='UNKNOWN',error=repr(e),consecutive_errors=errors,time=time.time()))
            # Read-only connection failures may retry; mutations/partial finalization never replay.
            if errors>=3 or (ARTIFACT/'results/completion.json').exists(): raise
        time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--mirror',action='store_true');a=p.parse_args()
    if a.watch: watch()
    elif a.mirror: mirror_root()
    else: finalize(inspect(ARTIFACT))
