"""Read-only N607 observer and local result delivery; never changes a worker."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from experiments.cvs_dual_evidence import design as d
from experiments.cvs_phase1_stack.publish import ssh

ARTIFACT=d.ROOT/'local_artifacts'/d.RELEASE
REPORT=d.ROOT/'automation_reports/CV-SincNet'/d.RUN
REGISTRY='E:/type10-7/tools/experiment_registry.py'


def readback():
    script='''
import json,time,os
from pathlib import Path
p=Path(BASE);release=Path(RELEASE)
def read(q):return json.loads(q.read_text()) if q.is_file() else None
def proc(pid):
    q=Path('/proc')/str(pid)
    try:
        stat=(q/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,state=stat[0],start_ticks=int(stat[19]),cwd=str((q/'cwd').resolve()),argv=[s for s in (q/'cmdline').read_bytes().decode().split('\\0') if s])
    except (FileNotFoundError,ProcessLookupError):return None
receipt=read(release/'submit.json');rows=[]
for kind in ('source','predict'):
    launch=read(p/('launch_'+kind+'.json')) or dict(rows=[])
    for r in launch['rows']:
        root=p/r['row_id'];log=Path(r['log']);actual=proc(r['pid'])
        metrics=root/'source/epoch_metrics.jsonl';last=None
        if metrics.is_file():
            lines=metrics.read_text().splitlines()
            try:last=json.loads(lines[-1]) if lines else None
            except json.JSONDecodeError:last=None
        rows.append(dict(**r,actual=actual,log_bytes=log.stat().st_size if log.exists() else None,
            epoch=last.get('epoch') if last else None,
            resolved=read(root/'source/resolved_config.json'),
            completion=read(root/('source/completion.json' if kind=='source' else 'prediction/complete.json'))))
print(json.dumps(dict(at=time.time(),receipt=receipt,controller=proc(receipt['pid']) if receipt else None,
    queue=read(p/'queue_state.json'),failure=read(p/'failure.json'),completion=read(p/'completion.json'),rows=rows)))
'''.replace('BASE',repr(str(d.BASE))).replace('RELEASE',repr(d.PROJECT+'/releases/'+d.RELEASE))
    state=json.loads(ssh(script));d.write(ARTIFACT/'readback.json',state);return state


def pull():
    names=['completion.json','failure.json','queue_state.json','dispatcher.json','source_matrix_frozen.json',
        'scoring_complete.json','scores.json','scores.csv','day_scores.json','day_scores.csv','summary.json','summary.csv',
        'paired_results.json','resources.json','analysis.md']
    for r in d.rows():
        rid=r['row_id']
        names += [rid+'/source/'+n for n in ('completion.json','initialization.json','resolved_config.json','resolved_native_args.json','epoch_metrics.jsonl')]
        names += [rid+'/prediction/'+n for n in ('provenance.json','resolved_config.json','complete.json')]
        names += [rid+'/'+n for n in ('scores.json','scores.csv','day_scores.json','resources.json')]
    script="import sys,tarfile;from pathlib import Path\nbase=Path("+repr(str(d.BASE))+')\nnames='+repr(names)+"\nwith tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:\n for n in names:\n  p=base/n\n  if p.is_file():t.add(p,arcname=n,recursive=False)\n"
    blob=ssh(script);(ARTIFACT/'results.tar.gz').write_bytes(blob)
    folder=ARTIFACT/'results';folder.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as t:
        for m in t.getmembers():
            if not m.isfile() or not (folder/m.name).resolve().is_relative_to(folder.resolve()):raise ValueError('Unsafe artifact path')
        t.extractall(folder)
    return folder


def verify(folder):
    import numpy as np
    from experiments.cvs_dual_evidence.evaluate import VIEWS,SCENES,paired_results
    rows=d.read(folder/'scores.json')['results'];days=d.read(folder/'day_scores.json')['results']
    if len(rows)!=1568 or len(days)!=448:raise ValueError('Result matrix incomplete')
    for r in rows+days:
        cm=np.asarray(r['confusion']);den=cm.sum(0)+cm.sum(1)
        f1=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()
        if cm.shape!=(6,6) or cm.sum()!=r['query_count'] or abs(np.trace(cm)/cm.sum()-r['accuracy'])>1e-12 or abs(f1-r['macro_f1'])>1e-12:raise ValueError('Metric recount differs')
    for c in d.rows():
        for view in VIEWS:
            group=[r for r in rows+days if r['row_id']==c['row_id'] and r['view']==view]
            overall=next(r for r in group if r['dimension']=='overall')
            for dim,count in [('receiver',7),('transmitter',6),('day',4)]:
                part=[r for r in group if r['dimension']==dim]
                if len(part)!=count or not np.array_equal(np.sum([r['confusion'] for r in part],axis=0),overall['confusion']):raise ValueError('Strata do not sum to whole')
    summary=d.read(folder/'summary.json')['results']
    if len(summary)!=28:raise ValueError('Arm/view summary incomplete')
    for a in summary:
        rs=[r for r in rows if r['arm']==a['arm'] and r['view']==a['view'] and r['dimension']=='overall']
        if set(r['model_seed'] for r in rs)!=set(d.SEEDS):raise ValueError('Four-seed coverage differs')
        for key in ('accuracy','macro_f1'):
            vals=[r[key] for r in rs]
            if abs(statistics.mean(vals)-a[key+'_mean'])>1e-12 or abs(statistics.stdev(vals)-a[key+'_sd'])>1e-12:raise ValueError('Mean/SD differs')
    actual_pairs=d.read(folder/'paired_results.json')['comparisons'];expected_pairs=paired_results(rows)
    if len(actual_pairs)!=len(expected_pairs):raise ValueError('Paired coverage differs')
    for a,b in zip(actual_pairs,expected_pairs):
        for key in ('treatment','control','view','metric','seeds','positive_seeds'):
            if a[key]!=b[key]:raise ValueError('Paired identity differs')
        for key in ('mean_difference','sd_difference'):
            if abs(a[key]-b[key])>1e-12:raise ValueError('Paired statistic differs')
        if not np.allclose(a['differences'],b['differences'],rtol=0,atol=1e-12):raise ValueError('Paired seed differences differ')
    for c in d.rows():
        epochs=[json.loads(s) for s in (folder/c['row_id']/'source/epoch_metrics.jsonl').read_text().splitlines()]
        if [int(r['epoch']) for r in epochs]!=list(range(1,201)):raise ValueError('Full200 epoch telemetry missing')
        done=d.read(folder/c['row_id']/'source/completion.json');d.require_budget(done['logged_steps'],done['optimizer_steps'])
    return dict(status='VERIFIED',models=16,views=7,metric_records=len(rows),day_metric_records=len(days),full_training_epochs=3200,
        confusion_partition_summary_paired_recount='VERIFIED',target_feedback=False)


def command(args):return subprocess.check_output(args,cwd=d.ROOT,text=True,encoding='utf-8')


def mirror():
    target=Path('E:/type10-7/automation_reports/CV-SincNet')/d.RUN
    if (target/'experiment.json').exists() and d.read(target/'experiment.json')['execution']['launch_owner']!=d.OWNER:raise ValueError('Canonical report belongs to another owner')
    shutil.copytree(REPORT,target,dirs_exist_ok=True)
    command([sys.executable,'-X','utf8',REGISTRY,'--root','E:/type10-7','build','--managed-only'])


def finalize(state):
    folder=pull();complete=state.get('completion')
    status='ANALYZED' if complete and complete.get('status')=='ANALYZED' else 'FAILED'
    if status=='ANALYZED':
        check=verify(folder);d.write(REPORT/'evidence/final_verified.json',check)
        results=REPORT/'results';results.mkdir(exist_ok=True)
        for path in folder.rglob('*'):
            if path.is_file() and path.name!='epoch_metrics.jsonl':
                dest=results/path.relative_to(folder);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        audit=[]
        keys=['epoch','val_tx_acc','train_loss','train_loss_entropy','train_loss_unlabeled','train_reliable_ratio',
            'train_pseudo_selected','train_pseudo_total','train_loss_sat_cls_labeled','epoch_time_s','lr']
        for c in d.rows():
            epochs=[json.loads(s) for s in (folder/c['row_id']/'source/epoch_metrics.jsonl').read_text().splitlines()]
            audit.extend([dict(row_id=c['row_id'],**{k:v for k,v in r.items() if k in keys or k.startswith('dual_')}) for r in epochs])
        fields=sorted({k for r in audit for k in r})
        with (results/'all_epoch_diagnostics.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(audit)
        summary=d.read(results/'summary.json')['results'];pairs=d.read(results/'paired_results.json')['comparisons']
        lines=['# 双骨干完整测试结果','','16个独立scratch E200模型，4组×4seed；各视图168000个固定query，112组预测先于truth评分。',
            '旧batch_neighbor伪标签＋cosine学习率、相同44400更新预算。已有暴露代理基准；Phase2/K/新增类/H不适用。',
            '独立重算混淆矩阵、RX/TX/day分区、四seed均值与SD、paired差值通过。结果不得回流调参或选择性重跑。','',
            '|架构|视图|准确率均值±SD（%）|Macro-F1（%）|最差RX均值（%）|','|---|---|---:|---:|---:|']
        lines += [f"|{r['arm']}|{r['view']}|{100*r['accuracy_mean']:.3f} ± {100*r['accuracy_sd']:.3f}|{100*r['macro_f1_mean']:.3f}|{100*r['worst_rx_mean']:.3f}|" for r in summary]
        lines+=['','## 配对差值','','|处理−对照|视图|均值（百分点）|SD（百分点）|正增益seed|','|---|---|---:|---:|---:|']
        lines += [f"|{r['treatment']} − {r['control']}|{r['view']}|{100*r['mean_difference']:.3f}|{100*r['sd_difference']:.3f}|{r['positive_seeds']}/4|" for r in pairs if r['metric']=='accuracy']
        lines += ['','逐行/RX/TX见scores.csv，日期见day_scores.csv，混淆矩阵见scores.json与day_scores.json。',
            '3200个epoch的紧凑诊断见all_epoch_diagnostics.csv；实际训练/推理成本见resources.json，共享GPU计时不等同独占性能。',
            '四个model seed仅为描述性重复，未据此宣称统计显著性。']
        (results/'full_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        with (REPORT/'report.md').open('a',encoding='utf-8') as f:f.write('\n## 完整测试交付\n\nANALYZED / VERIFIED。完整结果见[结果表](results/full_results.md)、[逐RX/TX](results/scores.csv)、[逐日期](results/day_scores.csv)。所有负结果完整保留。\n')
    else:
        with (REPORT/'report.md').open('a',encoding='utf-8') as f:f.write('\n## 技术失败\n\nFAILED：'+json.dumps(state.get('failure'),ensure_ascii=False)+'。产物保留，无自动重跑；当前测试未完成。\n')
    d.write(REPORT/'evidence/final_remote_readback.json',state)
    spec=d.read(REPORT/'experiment.json');spec['status']=status
    spec['results_ref']='results/full_results.md' if status=='ANALYZED' else None;d.write(REPORT/'experiment.json',spec)
    command([sys.executable,'-X','utf8',REGISTRY,'--root',str(d.ROOT),'record',d.RUN,'--status',status,
        '--evidence','evidence/final_remote_readback.json','--note','Independent artifact readback; full fixed-matrix delivery, no worker interventions'])
    mirror()
    if command(['git','branch','--show-current']).strip()!='codex/receiver-residual-dg-20261008':raise ValueError('Delivery branch changed')
    paths=['automation_reports/CV-SincNet/'+d.RUN,'experiment_registry']
    command(['git','-c','core.safecrlf=false','add','-f','--',*paths]);command(['git','diff','--cached','--check','--',*paths])
    command(['git','-c','core.safecrlf=false','commit','--only','-m','Report dual evidence fixed-matrix experiment '+status,'--',*paths])
    command(['git','push','origin','HEAD'])
    head=command(['git','rev-parse','HEAD']).strip()
    remote=command(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/codex/receiver-residual-dg-20261008']).split()[0]
    if remote!=head:raise ValueError('Remote delivery OID mismatch')
    d.write(ARTIFACT/'final_delivery.json',dict(status='VERIFIED',experiment_status=status,commit=head,remote_oid=remote))


def watch():
    import msvcrt
    ARTIFACT.mkdir(parents=True,exist_ok=True)
    with (ARTIFACT/'observer.lock').open('a+b') as guard:
        guard.seek(0);guard.write(b'0');guard.flush();guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
        d.write(ARTIFACT/'observer.json',dict(pid=os.getpid(),cwd=os.getcwd(),run_id=d.RUN,role='read-only observer/local delivery',started=time.time()))
        while True:
            try:state=readback()
            except Exception as error:
                d.write(ARTIFACT/'observer_last_error.json',dict(status='UNKNOWN',error=repr(error),at=time.time()))
                time.sleep(120);continue
            if state['failure'] or state['completion']:finalize(state);return
            if not state.get('controller') or state['controller']['state'] in ('Z','X'):
                d.write(ARTIFACT/'observer_last_error.json',dict(status='UNKNOWN',reason='Controller absent without completion; no relaunch'));return
            time.sleep(120)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.watch:watch()
    else:
        s=readback();print(json.dumps(dict(controller=s['controller'],queue=s['queue'],failure=s['failure'],completion=s['completion'],rows=[dict(row_id=r['row_id'],kind=r['kind'],gpu=r['gpu'],pid=r['pid'],epoch=r['epoch'],log_bytes=r['log_bytes'],resolved=r['resolved'] is not None) for r in s['rows']]),ensure_ascii=False))
