"""Summarize all frozen scored records; never select or modify experiments."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('E:/type10-7/local_artifacts/comparison_results_20260927')
METHODS=['cvcnn_ce-ce','cvcnn_ce-pl','riei_fd-ce','riei_fd-pl','drift-ce','drift-pl','poster-ce','radionet-ce']
VIEWS=['clean','practical_high','practical_mid','practical_low_urban']


def summarize():
    p1=json.loads((DATA/'phase1_results.json').read_text(encoding='utf-8'))
    p2=json.loads((DATA/'phase2_results.json').read_text(encoding='utf-8'))
    assert p1['status']==p2['status']=='SCORED'
    assert len(p1['results'])==1280 and len(p2['results'])==105840
    groups=defaultdict(lambda:defaultdict(list))
    for r in p1['results']:
        if r['receiver']!='ALL':
            continue
        method,seed=r['row_id'].rsplit('-s',1)
        assert sum(map(sum,r['confusion']))==r['count']
        assert abs(sum(r['confusion'][i][i] for i in range(6))/r['count']-r['accuracy'])<1e-12
        for metric in ('accuracy','macro_f1'):
            groups[('phase1',method,r['view'],metric)][int(seed)].append(r[metric])
    seen=set()
    for r in p2['results']:
        key=(r['row_id'],r['split_id'],r['mode'])
        assert key not in seen
        seen.add(key)
        method,seed=r['row_id'].rsplit('-s',1)
        k=0 if r['mode']=='frozen_dg' else r['k']
        assert r['query_count']==30*r['class_count']
        for metric in ('accuracy','old_accuracy','new_accuracy','harmonic_mean'):
            if r[metric] is not None:
                groups[('phase2',method,r['mode'],r['class_count'],k,metric)][int(seed)].append(r[metric])
    summary={}
    for key,values in groups.items():
        assert set(values)=={392005,2026092701,2026092702,2026092703,2026092704}
        expected=1 if key[0]=='phase1' else (21 if key[2]=='frozen_dg' else 105)
        assert all(len(v)==expected for v in values.values()),(key,{s:len(v) for s,v in values.items()})
        byseed={str(s):mean(v)*100 for s,v in values.items()}
        fresh=[v for s,v in byseed.items() if s!='392005']
        summary['|'.join(map(str,key))]=dict(fresh_mean=mean(fresh),fresh_sd=stdev(fresh),historical=byseed['392005'],
            per_seed=byseed,records_per_seed=expected)
    def cell(key,history=False):
        r=summary['|'.join(map(str,key))]
        return f"{r['historical']:.2f}" if history else f"{r['fresh_mean']:.2f} ± {r['fresh_sd']:.2f}"
    lines=['# 对比实验结果：2026-09-27','',
        'N607读回核实：40/40源模型完成200轮；Phase1最终clean/星地测试已评分；Phase2的40/40预测任务完成，scorer正常结束，共105840条评分记录。',
        '', '数值单位为%。主表为4个新模型种子2026092701–2026092704的均值±样本标准差。历史优化种子392005单列。Phase2先在每个模型种子内平均7接收机×3场景×5support抽样，再计算跨模型种子统计；这些抽样不作为105个独立模型seed。',
        '', '所有结果使用固定第200轮last.pt；不选择最好轮次、最好seed或最好接收机。星地场景为residual/post_sync/noeq。当前没有本批匹配CVS结果，不能据此宣称CVS优于对比方法。',
        '', '## Phase1目标域准确率','', '| 方法 | Clean | High | Mid | Low urban |','|---|---:|---:|---:|---:|']
    for m in METHODS:
        lines.append('| '+m+' | '+' | '.join(cell(('phase1',m,v,'accuracy')) for v in VIEWS)+' |')
    lines+=['','测试共168000条ManySig目标物理记录；每条提供clean对照及一个固定星地场景观测。3种星地场景的物理样本集合互不重叠。','', '## Phase1 Macro-F1','', '| 方法 | Clean | High | Mid | Low urban |','|---|---:|---:|---:|---:|']
    for m in METHODS:
        lines.append('| '+m+' | '+' | '.join(cell(('phase1',m,v,'macro_f1')) for v in VIEWS)+' |')
    lines+=['','## Phase2：6旧类+20新类，总准确率','',
        '每列固定相同K、相同注册类、相同received IQ和support/query。NCM是统一冻结表征注册扩展；POSTER与RadioNet FT为其作者微调路线。不同路线分别列出。','',
        '| 方法/路线 | K=1 | K=5 | K=10 | K=20 |','|---|---:|---:|---:|---:|']
    routes=[(m,'support_ncm') for m in METHODS]+[(m,'author_finetune') for m in ['poster-ce','radionet-ce']]
    for m,mode in routes:
        lines.append('| '+m+'/'+mode+' | '+' | '.join(cell(('phase2',m,mode,26,k,'accuracy')) for k in (1,5,10,20))+' |')
    lines+=['','## Phase2：K=5、6旧类+20新类分项','', '| 方法/路线 | 旧类 | 新类 | 调和均值 |','|---|---:|---:|---:|']
    for m,mode in routes:
        lines.append('| '+m+'/'+mode+' | '+' | '.join(cell(('phase2',m,mode,26,5,metric)) for metric in ('old_accuracy','new_accuracy','harmonic_mean'))+' |')
    lines+=['','## Phase2旧类域泛化/适应：仅6旧类','', '| 方法 | 冻结DG | K=5 NCM | K=5作者FT |','|---|---:|---:|---:|']
    for m in METHODS:
        lines.append('| '+m+' | '+cell(('phase2',m,'frozen_dg',6,0,'accuracy'))+' | '+cell(('phase2',m,'support_ncm',6,5,'accuracy'))+' | '+(cell(('phase2',m,'author_finetune',6,5,'accuracy')) if m in ('poster-ce','radionet-ce') else 'N/A')+' |')
    lines+=['','## 历史优化种子392005：单列，不并入主表','', '| 方法 | Clean | High | Mid | Low urban |','|---|---:|---:|---:|---:|']
    for m in METHODS:
        lines.append('| '+m+' | '+' | '.join(cell(('phase1',m,v,'accuracy'),True) for v in VIEWS)+' |')
    lines+=['','## 范围与复现','',
        '- Phase1源训练ManySig：6TX、5源RX、L/U/V=0.07/0.63/0.30；7目标RX与源RX分离。',
        '- Phase2目标数据ManyTx：6旧类+最多20新类、7目标RX、3场景、4档K、5次support抽样。不可直接与Phase1 ManySig准确率作方法增益差值。',
        '- POSTER源预算200轮为本项目匹配预算，微调10轮；RadioNet为DF路线，微调30轮，不代表ADA/triplet全部变体。',
        '- 本表不含CVS、其余尚未运行的原生DA/注册方法。低分保留，不触发测试集调参或选择性重跑。',
        '- 完整逐接收机结果见本机local_artifacts/comparison_results_20260927/phase1_results.json；完整Phase2逐split结果见同目录phase2_results.json。',
        '- 汇总前校验全部记录数、重复row/split/mode、5个模型seed覆盖、每seed切片数量，以及Phase1混淆矩阵计数和准确率一致性。']
    report=ROOT/'docs/COMPARISON_RESULTS_20260927.md'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (DATA/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n'.join(lines[:19]))
    print('\n'.join(lines[32:47]))
    return report


if __name__=='__main__':
    summarize()
