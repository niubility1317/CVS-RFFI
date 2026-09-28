"""Validate all180 diagnostic rows and produce compact source-only evidence."""
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=Path('E:/type10-7')
RUN='20260928-diagnostic-d92-scv-source-s2026092701-r01'


def main():
    source=WORKSPACE/'local_artifacts/d92_upgrade_20260928/source_r01/metrics.jsonl'
    rows=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    keys=[(r['receiver'],r['scene'],r['k'],r['method']) for r in rows]
    methods=['frozen_dg','D92','D92-SCV-v1'];ks=[1,5,10,20]
    if len(rows)!=180 or len(set(keys))!=180:raise ValueError('Incomplete/duplicated source matrix')
    if any(r['validation_count']!=300 or r['new_accuracy'] is not None or r['harmonic'] is not None for r in rows):raise ValueError('Metric scope drift')
    report=WORKSPACE/'automation_reports/CV-SincNet'/RUN
    output=report/'results';output.mkdir(exist_ok=True)
    compact=[{k:v for k,v in r.items() if k!='audit'} for r in rows]
    (output/'compact.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in compact),encoding='utf-8')
    with (output/'per_cell.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
    summary=[]
    for k in ks:
        means={m:statistics.mean(r['accuracy'] for r in rows if r['k']==k and r['method']==m) for m in methods}
        pair=[]
        for r in rows:
            if r['k']==k and r['method']=='D92-SCV-v1':
                b=next(t for t in rows if (t['receiver'],t['scene'],t['k'],t['method'])==(r['receiver'],r['scene'],k,'D92'))
                pair.append(r['accuracy']-b['accuracy'])
        summary.append(dict(k=k,**means,delta_vs_d92=means['D92-SCV-v1']-means['D92'],
            delta_vs_dg=means['D92-SCV-v1']-means['frozen_dg'],worst_cell_delta=min(pair),
            cells_below_d92=sum(d<0 for d in pair)))
    evidence=dict(status='VERIFIED_SOURCE_DIAGNOSTIC',row_count=len(rows),summary=summary,
        model_seed=2026092701,source_only=True,new_class_h_verified=False,
        fit_seconds={m:sum(r['elapsed_seconds'] for r in rows if r['method']==m) for m in methods},
        full_trace_path=str(source),target_access=False)
    (output/'summary.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    table=['|K|冻结DG|原D92|D92-SCV-v1|相对D92（百分点）|相对DG（百分点）|',
           '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        table.append(f"|{r['k']}|{r['frozen_dg']*100:.2f}%|{r['D92']*100:.2f}%|{r['D92-SCV-v1']*100:.2f}%|{r['delta_vs_d92']*100:+.2f}|{r['delta_vs_dg']*100:+.2f}|")
    text='\n## 第一轮源域结果（VERIFIED）\n\n远端state及complete均显示180条完整；dispatcher/worker均已退出。33300个源域物理记录导出完整，原L6300、单一V27000；checkpoint严格加载，缺失/多余key均为0，未更新encoder、未访问target。\n\n'+'\n'.join(table)+'\n\n每档K为15个RX×scene切片的均值，单一model seed、单一support draw；V每类50条。候选相对原D92改善明显，但主要通过避免破坏已有Phase1判决。尚不能证明比零适应更强，更不能宣称新类H改善或目标域泛化改善。逐切片与完整选择证据见results/和本地full_trace_path。\n\n后续：联合注册predictor已实现并通过11项输入边界负测，但尚未发布/运行。确认集范围待用户选择：优先独立未暴露样本，或在本批数据上做冻结后的一次描述性对比。元数据核实19-1、8-14、8-7三个接收机未进入最近一批source/target集合，均覆盖现有全部26类、每类至少200条；历史所有实验是否暴露尚未证明，不能直接称严格未暴露确认集。\n'
    path=report/'report.md';body=path.read_text(encoding='utf-8')
    if '## 第一轮源域结果' in body:raise ValueError('Results already recorded; do not append twice')
    path.write_text(body+text,encoding='utf-8')
    (WORKSPACE/'docs/D92_SUPPORT_UPGRADE_20260928.md').write_text((ROOT/'docs/D92_SUPPORT_UPGRADE_20260928.md').read_text(encoding='utf-8')+text,encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
