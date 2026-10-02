"""Recompute all scored confusion matrices; report fixed paired comparisons."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from experiments.cvs_selected_concat.prepare import RUN,SEEDS

VIEWS=['clean','satellite','practical_high','practical_mid','practical_low_urban']
LABELS=['Clean','星地总体','High','Mid','Low urban']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def metrics(cm):
    if len(cm)!=6 or any(len(x)!=6 or any(type(y)!=int or y<0 for y in x) for x in cm):raise ValueError('Invalid confusion matrix')
    n=sum(map(sum,cm))
    if not n:raise ValueError('Empty confusion matrix')
    f1=[]
    for i in range(6):
        den=sum(cm[i])+sum(x[i] for x in cm);f1.append(2*cm[i][i]/den if den else 0.)
    return dict(accuracy=sum(cm[i][i] for i in range(6))/n,macro_f1=sum(f1)/6,query_count=n)
def verify(row):
    m=metrics(row['confusion'])
    if m['query_count']!=row.get('query_count',row.get('count')):raise ValueError('Count mismatch')
    for k in ['accuracy','macro_f1']:
        if not math.isclose(m[k],row[k],abs_tol=1e-12):raise ValueError('Metric mismatch: '+k)
    return row
def merged(rows):return [[sum(x['confusion'][i][j] for x in rows) for j in range(6)] for i in range(6)]
def avg(v):return statistics.mean(v),statistics.stdev(v)
def fmt(v):
    mean,sd=avg(v);return f'{100*mean:.2f} ± {100*sd:.2f}'
def csvwrite(p,rows):
    fields=list(rows[0])
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def analyze(root,workspace):
    evidence=root/'evidence/final';audit=read(evidence/'completion_audit.json');new=[verify(x) for x in read(evidence/'test_scores.json')['results']]
    if len(new)!=160 or len({(x['model_seed'],x['view'],x['receiver']) for x in new})!=160:raise ValueError('Expected exact 4x5x8 score records')
    if sorted({x['model_seed'] for x in new})!=SEEDS or set(x['view'] for x in new)!=set(VIEWS):raise ValueError('Unregistered score matrix')
    receivers=sorted({x['receiver'] for x in new if x['receiver']!='ALL'})
    if len(receivers)!=7:raise ValueError('Expected seven receiver strata')
    for seed in SEEDS:
        for view in VIEWS:
            allrow=next(x for x in new if x['model_seed']==seed and x['view']==view and x['receiver']=='ALL')
            pieces=[x for x in new if x['model_seed']==seed and x['view']==view and x['receiver']!='ALL']
            if len(pieces)!=7 or merged(pieces)!=allrow['confusion']:raise ValueError('RX/overall confusion mismatch')
        sat=next(x for x in new if x['model_seed']==seed and x['view']=='satellite' and x['receiver']=='ALL')
        scenes=[x for x in new if x['model_seed']==seed and x['view'] in VIEWS[2:] and x['receiver']=='ALL']
        if merged(scenes)!=sat['confusion']:raise ValueError('Scene/overall confusion mismatch')
    oldpath=workspace/'automation_reports/CV-SincNet/20261001-phase1-cvs-identity-ce-practical-manysig-m5-r02/evidence/test_scores_readback.json'
    old=[verify(x) for x in read(oldpath)['files']['phase1_scored_results.json']['results'] if x['model_seed'] in SEEDS]
    cv=[verify(x) for x in read(workspace/'local_artifacts/comparison_results_20260927/phase1_results.json')['results'] if x['row_id'].startswith('cvcnn_ce-ce-s') and int(x['row_id'].split('-s')[-1]) in SEEDS]
    pure=read(root/'evidence/pure_clean_baseline_reference.json')['rows']
    def get(data,seed,view,rx='ALL'):return next(x for x in data if x['model_seed']==seed and x['view']==view and x['receiver']==rx)
    comparisons=[]
    for view in VIEWS:
        for seed in SEEDS:
            a=get(new,seed,view);b=get(old,seed,view)
            if view=='satellite':
                rows=[x for x in cv if x['row_id']=='cvcnn_ce-ce-s'+str(seed) and x['receiver']=='ALL' and x['view'] in VIEWS[2:]]
                if len(rows)!=3:raise ValueError('Expected three CVCNN scene matrices')
                c=metrics(merged(rows))
            else:
                c=next(x for x in cv if x['row_id']=='cvcnn_ce-ce-s'+str(seed) and x['receiver']=='ALL' and x['view']==view)
            if a['query_count']!=b['query_count'] or a['query_count']!=c.get('query_count',c.get('count')):raise ValueError('Comparison counts differ')
            comparisons.append(dict(seed=seed,view=view,new_accuracy=a['accuracy'],new_macro_f1=a['macro_f1'],old_native_CVS_CE_accuracy=b['accuracy'],cvcnn_CE_accuracy=c['accuracy'],
                delta_native_CVS_CE_pp=100*(a['accuracy']-b['accuracy']),delta_CVCNN_CE_pp=100*(a['accuracy']-c['accuracy']),
                pure_clean_residual_accuracy=next(x['accuracy'] for x in pure if x['seed']==seed) if view=='clean' else None,
                delta_pure_clean_residual_pp=100*(a['accuracy']-next(x['accuracy'] for x in pure if x['seed']==seed)) if view=='clean' else None))
    csvwrite(evidence/'test_metrics.csv',[{k:v for k,v in x.items() if k!='confusion'} for x in new])
    csvwrite(evidence/'paired_comparisons.csv',comparisons)
    text='\n\n## 最终测试：VERIFIED\n\n4/4 从零训练完成 E200，固定末轮权重完成 clean／satellite 预测，全部4行固定后独立 truth-last 评分完成160条指标记录。全40000步、800轮文本／JSONL／CSV一致性与预算／增强状态核验通过；160个总体和RX混淆矩阵的accuracy／Macro-F1独立重算通过，RX合并、三场景合并与总体完全一致。\n\n'
    text+='以下为四个固定seed的均值±样本标准差，单位为%。**具体识别性能由测试集判断；源V不作为最终性能结论。**\n\n| 测试场景 | 本次残差CVS Accuracy | 本次 Macro-F1 | 历史原生 CVS-CE Accuracy | 历史 CVCNN-CE Accuracy |\n|---|---:|---:|---:|---:|\n'
    summary=[]
    for view,label in zip(VIEWS,LABELS):
        rows=[x for x in comparisons if x['view']==view]
        accuracy=[x['new_accuracy'] for x in rows];f1=[x['new_macro_f1'] for x in rows]
        text+=f'| {label} | {fmt(accuracy)} | {fmt(f1)} | {fmt([x["old_native_CVS_CE_accuracy"] for x in rows])} | {fmt([x["cvcnn_CE_accuracy"] for x in rows])} |\n'
        summary.append(dict(view=view,accuracy_mean=avg(accuracy)[0],accuracy_seed_sd=avg(accuracy)[1],macro_f1_mean=avg(f1)[0],macro_f1_seed_sd=avg(f1)[1],
            delta_native_CVS_CE_mean_pp=statistics.mean(x['delta_native_CVS_CE_pp'] for x in rows),delta_CVCNN_CE_mean_pp=statistics.mean(x['delta_CVCNN_CE_pp'] for x in rows)))
    clean=[x for x in comparisons if x['view']=='clean'];delta=[x['delta_pure_clean_residual_pp'] for x in clean]
    text+=f'\n纯clean训练残差CVS基准为 **78.4543% ± 0.8436%**（Macro-F1 78.2067% ± 0.9517%）。本次相对其clean准确率配对差值为 **{statistics.mean(delta):+.4f} ± {statistics.stdev(delta):.4f}个百分点**，{sum(x>0 for x in delta)}/4种子为正。该基准已完成，无需补跑；它没有历史卫星测试，本次不追溯追加。\n\n'
    text+='| 模型seed | Clean | 星地总体 | High | Mid | Low urban |\n|---:|---:|---:|---:|---:|---:|\n'
    for seed in SEEDS:text+='| '+str(seed)+' | '+' | '.join(f'{100*get(new,seed,v)["accuracy"]:.2f}%' for v in VIEWS)+' |\n'
    text+='\n| 接收机物理ID | Clean | 星地总体 | High | Mid | Low urban |\n|---|---:|---:|---:|---:|---:|\n'
    for rx in receivers:text+='| '+str(rx)+' | '+' | '.join(fmt([get(new,s,v,rx)['accuracy'] for s in SEEDS]) for v in VIEWS)+' |\n'
    text+='\n各RX的Macro-F1、完整混淆矩阵及逐seed配对差值均保存在附件，不只报告总体。训练只用mid／low urban，High是本次未用于训练增强的场景；卫星总体按样本合并，并非三场景指标的简单平均。\n\n'
    profiles=[x['profile'] for x in audit['rows']]
    text+='| 实测成本 | 四seed均值 |\n|---|---:|\n'
    for key,label in [('total_parameters','总参数'),('gradient_used_parameters','CE实际使用参数'),('conv_linear_macs_per_sample','Conv／Linear MAC/样本'),('resident_state_bytes','模型常驻字节'),('training_batch128_ms','合成训练batch128 ms'),('inference_batch1_ms','推理batch1 ms'),('inference_batch128_ms','推理batch128 ms')]:
        text+=f'| {label} | {statistics.mean(x[key] for x in profiles):.3f} |\n'
    text+='\n硬件RTX3090、Torch2.1.0+cu121、完整FP32；副本resource profile不回流模型。MAC不包含FFT／归一化／物理特征等，实测训练batch128是可丢弃副本上的合成普通CE；正式E80后拼接batch256及信道生成成本由完整源日志耗时记录，不能把副本单batch成本当作正式增强训练成本。星载设备／新增传输字节N/A。\n\n'
    text+='本次是用户指定的既有测试较优结构＋固定两场景增强实验，历史benchmark已曝光；不是新的盲测或目标无接触的架构筛选。本轮测试不回流参数、结构、权重或种子选择，全部种子及负结果保留。历史CVS／CVCNN使用三场景增强，网络、显式loader和backend口径还存在差异；无增强残差基准也存在backend／loader口径差异。因此以上差值评价整套固定配置，不单独证明“两场景增强”或残差头的因果收益。无Phase2，适应／新类／H为N/A。\n\n'
    text+='[完整训练与产物审计](evidence/final/completion_audit.json) · [完整测试混淆矩阵](evidence/final/test_scores.json) · [所有分层指标CSV](evidence/final/test_metrics.csv) · [逐种子配对比较](evidence/final/paired_comparisons.csv)。\n'
    with (root/'report.md').open('a',encoding='utf-8') as f:f.write(text)
    result=dict(status='VERIFIED',summary=summary,pure_clean_delta_mean_pp=statistics.mean(delta),pure_clean_delta_sd_pp=statistics.stdev(delta),all_cm_recomputed=160,
        full_step_records=40000,epoch_records=800,release_commit=audit['release_commit'],target_feedback=False,claim_scope='BENCHMARK_INFORMED_FIXED_DESIGN')
    (evidence/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report-root',type=Path,required=True);p.add_argument('--workspace',type=Path,default=Path('E:/type10-7'));a=p.parse_args();analyze(a.report_root,a.workspace)
