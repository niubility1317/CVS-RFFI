"""Report already-scored Phase1 results and verified partial Phase2 status."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import shutil

FRESH={2026092701,2026092702,2026092703,2026092704}
NAMES={'protonet':'ProtoNet CDA','feature_separation':'Feature Separation','dadda':'DADDA','mrior':'MRIOR',
       'twostage':'TwoStage','csil':'CSIL','mopc_hr':'MoPC-HR','orthogonal':'Orthogonal Incremental SEI','radionet_ada':'RadioNet ADA专用骨干'}
VIEWS=('clean','satellite','practical_high','practical_mid','practical_low_urban')

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def verify(snapshot):
    flag=load(snapshot/'scoring_p1_complete.json');result=load(snapshot/'phase1_scored_results.json')
    inventory=load(snapshot/'artifact_inventory.json');summary=load(snapshot/'phase1_summary.json')['summary']
    rows=result['results']
    if flag['status']!='SCORED_COMPLETE' or result['status']!='SCORED' or flag['records']!=len(rows) or len(rows)!=1800:
        raise ValueError('Phase1 scored coverage mismatch')
    cells={}
    for row in rows:
        key=(row['method'],row['model_seed'],row['view'],row['receiver'])
        if key in cells:raise ValueError('Repeated scored cell')
        cells[key]=row
        cm=row['confusion'];count=sum(sum(v) for v in cm)
        if count!=row['query_count'] or not math.isclose(sum(cm[i][i] for i in range(len(cm)))/count,row['accuracy'],abs_tol=1e-12):
            raise ValueError('Scored confusion/count/accuracy mismatch')
    for method in NAMES:
        for seed in FRESH|{392005}:
            if any((method,seed,v,'ALL') not in cells for v in VIEWS):raise ValueError('Missing Phase1 view/seed')
            if cells[method,seed,'clean','ALL']['query_count']!=168000 or cells[method,seed,'satellite','ALL']['query_count']!=168000:
                raise ValueError('Phase1 target count mismatch')
            if sum(cells[method,seed,v,'ALL']['query_count'] for v in VIEWS[2:])!=168000:raise ValueError('Scene count mismatch')
            for view in VIEWS:
                entries=[r for r in rows if r['method']==method and r['model_seed']==seed and r['view']==view and r['receiver']!='ALL']
                if len(entries)!=7 or sum(r['query_count'] for r in entries)!=cells[method,seed,view,'ALL']['query_count']:
                    raise ValueError('Receiver coverage mismatch')
    for s in summary:
        if s['receiver']!='ALL':continue
        seeds=FRESH if s['seed_scope']=='fresh_four' else {392005} if s['seed_scope']=='historical_392005' else FRESH|{392005}
        values=[cells[s['method'],seed,s['view'],'ALL']['accuracy'] for seed in seeds]
        if not math.isclose(statistics.mean(values),s['accuracy_mean'],abs_tol=1e-12):raise ValueError('Summary mean mismatch')
        if len(values)>1 and not math.isclose(statistics.stdev(values),s['accuracy_seed_sd'],abs_tol=1e-12):raise ValueError('Summary seed SD mismatch')
    return rows,summary,inventory

def build(snapshot,output):
    snapshot=Path(snapshot);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows,summary,inv=verify(snapshot)
    def selected(m,v,scope='fresh_four'):
        return next(r for r in summary if r['method']==m and r['view']==v and r['receiver']=='ALL' and r['seed_scope']==scope)
    def value(m,v):
        row=selected(m,v);return f"{100*row['accuracy_mean']:.2f} ± {100*row['accuracy_seed_sd']:.2f}"
    order=sorted(NAMES,key=lambda m:selected(m,'satellite')['accuracy_mean'],reverse=True)
    lines=['# 原生对比实验进展与Phase1结果（2026-10-01）','',
        '**进程已全部结束，整批未全部成功。**45个源模型均完成200轮；45个Phase1目标评估已固定并独立评分。Phase2完成35条完整预测流水线，CSIL与MoPC-HR各5条在新类注册时失败，整批状态为PARTIAL。N607核查时调度器PID98355已退出，GPU无训练进程。','',
        f"产物清单采集时间（UTC）：`{inv['captured_at']}`。本次只读核查、回收和整理结果，没有停止、重新训练、重启、修改远端配置或打开Phase2 truth。",'',
        '## Phase1目标域准确率','',
        '单位为%；均值±样本标准差来自预先固定的4个新模型seed（2026092701至2026092704）。历史优化seed392005单列在CSV，不混入主统计；全5seed另列敏感性统计。所有结果使用第200轮最后权重。','',
        '|方法|clean|星地总体|high|mid|low urban|','|---|---:|---:|---:|---:|---:|']
    for method in order:lines.append('|'+NAMES[method]+'|'+'|'.join(value(method,v) for v in VIEWS)+'|')
    lines+=['','每个模型seed：clean与星地总体各168000例；high55998例、mid56174例、low urban55828例，三个场景的物理样本互斥。星地总体按实际样本数汇总，不能把三个场景均值直接当成总体。RadioNet行是ADA专用骨干在Phase1的零目标域适应测试，并不是ADA后k-NN的准确率。','',
        '本轮新增9个源模型中，CSIL星地总体均值最高（67.10%），其次为ProtoNet CDA（65.04%）。这是均值比较，未据此宣称统计显著。Feature Separation的星地总体高于clean；报告保留原值，不据此反向选择或修改参数。','',
        '## 与既有CVS结果的关系','',
        '既有matched批次CVS的clean/high/mid/low urban准确率分别为80.33±0.80、79.37±1.70、77.26±1.57、52.46±0.62%。因此CVS在clean、high、mid仍高于本轮新增方法；low urban不是最优，CSIL的58.54%高出6.08个百分点。不能写成CVS在所有星地场景均领先。','',
        '该CVS行来自既有Phase1 DAOT A1＋FastTrust-RC4，Phase2对应D92 E0去RF32；不是本轮新跑的CVS。参见[原批次结果](../cvs_comparison_20260928/完整对比实验报告.md)。输入capsule与主统计seed口径相同，源训练机制、无标签数据使用和样本暴露量不同，论文必须披露这些差异。','',
        '## Phase2完成情况','',
        '|方法|完整预测row/计划row|新类注册结果状态|','|---|---:|---|']
    for method in NAMES:
        records=[r for r in inv['rows'] if r['stage']=='phase12' and r['method']==method]
        complete=sum(r['phase2_status']=='PREDICTIONS_COMPLETE' for r in records)
        note='注册时报兼容性错误；缺失完整结果' if method in {'csil','mopc_hr'} else '原生闭集方法，C记N/A' if method in {'dadda','mrior','twostage','feature_separation'} else '完整预测已固定，整批独立评分未启动'
        lines.append(f'|{NAMES[method]}|{complete}/{len(records)}|{note}|')
    lines+=['',f"35条成功流水线共固定{inv['phase2_complete_prediction_records']}条预测记录，覆盖7个方法的5个模型seed。CSIL/MoPC-HR失败输出及日志保留，不能以其他方法的结果补齐。当前没有本批已独立评分的Phase2旧类适应准确率、注册后新旧类准确率或H，这些项记N/A；35条预测完成不等于35条已经评分。",'',
        '失败签名（10条一致）：','',"```text\nTypeError: kaiming_uniform_() got an unexpected keyword argument 'generator'\n```",'',
        '错误发生于CSIL/MoPC-HR扩展分类器的Kaiming初始化；N607的PyTorch API与本地验证环境不一致。它是程序兼容性失败，不能解释为方法准确率低或优化未收敛。本地验证未覆盖这一实际N607版本边界，是本轮实现的验证遗漏。','',
        '当前scorer要求所有45条预登记Phase2预测完整后才连接truth，10条失败使其保持WAITING。下一步应修复该确定性兼容错误，保持方法、数据、预算和全部seed不变，在新run保留原失败产物、补齐失败阶段，然后对完整矩阵独立评分；不能按Phase1目标成绩选seed或调参。','',
        '## 版本与证据边界','',
        '训练和预测代码：`577825849c5c69edfbec9e258948b5621e4c643a`。信道为25MHz residual_noeq（post_sync残差、无额外均衡），Phase2复用capsule `residual-noeq-ba667eee4fb061055e4c08b5`。RadioNet执行ADA→冻结编码器→cosine距离加权k-NN（k=K），该批1000次更新与作者默认10000次不同，仍需披露。CSIL经验Fisher/掩码修正、CSIL/MoPC的IQ编码器扩展按原设计披露。','',
        '本报告校核了1800条Phase1评分记录的混淆矩阵、样本数、场景/RX/seed覆盖及跨seed汇总；没有重算预测或训练模型。完整分层数据及历史seed/全5seed统计见CSV。','',
        '[Phase1统计CSV](phase1_summary.csv) · [当前读回证据](../../../automation_reports/CV-SincNet/20260930-phase12-native-baselines-practical-m5-r01/local_readback/20261001T1053HKT/artifact_inventory.json)']
    (output/'实验进展与Phase1结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fields=list(summary[0])
    with (output/'phase1_summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(summary)
    compact=[{k:v for k,v in r.items() if k!='confusion'} for r in rows]
    with (output/'phase1_all_rows.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
    meta={'phase1_scored_rows':len(rows),'phase1_summary_rows':len(summary),'read_only_remote':True,
        'phase2_complete':inv['phase2_complete'],'phase2_failed':sum(r['state']=='FAILED' for r in inv['rows'] if r['stage']=='phase12'),
        'phase2_scored':False,'verified_at_utc':inv['captured_at'],'code_commit':inv['dispatcher_state']['commit']}
    (output/'validation.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workspace-root',type=Path);p.add_argument('--mirror-to',type=Path)
    a=p.parse_args();print(json.dumps(build(a.snapshot,a.output)))
    if a.mirror_to:
        if not a.workspace_root:raise ValueError('--workspace-root required for mirroring')
        runs=('20260930-phase1-native-baselines-practical-manysig-m5-r01','20260930-phase12-native-baselines-practical-m5-r01')
        for run in runs:
            relative=Path('automation_reports/CV-SincNet')/run
            report=a.workspace_root/relative/'report.md'
            header='## 2026-10-01 independent completion readback'
            current=report.read_text(encoding='utf-8')
            if header not in current:
                current+='\n'+header+'\n\n45 final-epoch200 source models verified; Phase1 scored45/45, 1800 stratified records. Phase2 predictions35/45, failed10/45 (CSIL and MoPC-HR generator initialization API incompatibility); Phase2 scoring not complete.\n\n[Phase1 results and failures](../../../docs/results/cvs_native_comparison_20261001/实验进展与Phase1结果.md). Readback: local_readback/20261001T1053HKT/artifact_inventory.json in the Phase12 run.\n'
                report.write_text(current,encoding='utf-8')
            for name in ('experiment.json','report.md','events.jsonl'):
                shutil.copy2(a.workspace_root/relative/name,a.mirror_to/relative/name)
        for source in (a.snapshot,a.output):
            shutil.copytree(source,a.mirror_to/source.relative_to(a.workspace_root),dirs_exist_ok=True)
        for name in ('inspect_native_comparison_artifacts.py','build_native_comparison_status_report.py'):
            shutil.copy2(Path(__file__).parent/name,a.workspace_root/'tools'/name)
