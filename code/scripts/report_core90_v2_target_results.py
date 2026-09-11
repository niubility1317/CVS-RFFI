"""Detailed multi-seed target-domain report, separate from source validation."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st

def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])+'\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();d=json.loads(Path(a.input).read_text(encoding='utf-8'));out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    m=d['manifest'];scores=d['scores'];scenes=m['scenes'];rows=m['rows'];assert d['complete']['complete'] and set(rows)==set(scores)
    records=[];summary=[];cms=[];pairs=[];aggregates=[]
    for row in rows:
        score=scores[row];method=row.rsplit('_seed',1)[0];seed=m['training_seeds'][row]
        assert score['target_evaluated'] and not score['source_only'] and score['prediction_count']==672000
        s=dict(run_id=row,method=method,seed=seed,**{scene+'_accuracy_pct':score['scenes'][scene]['accuracy']*100 for scene in scenes},
            **{scene+'_macro_f1_pct':score['scenes'][scene]['macro_f1']*100 for scene in scenes},
            leo_mean_accuracy_pct=score['leo_mean_accuracy']*100,leo_mean_macro_f1_pct=st.mean(score['scenes'][s]['macro_f1'] for s in scenes[1:])*100,
            worst_leo_rx_accuracy_pct=min(score['scenes'][s]['worst_rx_accuracy'] for s in scenes[1:])*100)
        summary.append(s)
        for scene in scenes:
            metric=score['scenes'][scene]
            for scope,groups in [('overall',{'all':metric}),('receiver',metric['per_rx']),('day',metric['per_day'])]:
                for group,g in groups.items():
                    records.append(dict(run_id=row,method=method,seed=seed,scene=scene,scope=scope,group=group,tx='',count=g['count'],
                        accuracy_pct=g['accuracy']*100,macro_f1_pct=g['macro_f1']*100,macro_recall_pct=g['macro_recall']*100))
                    for tx,v in g['per_tx_accuracy'].items():
                        records.append(dict(run_id=row,method=method,seed=seed,scene=scene,scope=scope+'_tx',group=group,tx=tx,
                            count=g['per_tx_count'][tx],accuracy_pct=v*100,macro_f1_pct='',macro_recall_pct=''))
            for y,line in enumerate(d['confusion_matrices'][row][scene]):
                for prediction,count in enumerate(line):cms.append(dict(run_id=row,scene=scene,truth=y,prediction=prediction,count=count))
    for method in sorted({r['method'] for r in summary}):
        group=[r for r in summary if r['method']==method]
        for field in [*[s+'_accuracy_pct' for s in scenes],'leo_mean_accuracy_pct','leo_mean_macro_f1_pct']:
            values=[r[field] for r in group]
            aggregates.append(dict(method=method,metric=field,n_seeds=len(group),mean_pct=st.mean(values),sample_std_pct=st.stdev(values) if len(values)>1 else None))
    for pair in d['paired_predictions']:
        baseline=pair['baseline'];variant=pair['variant']
        for scene,counts in pair['scenes'].items():
            pairs.append(dict(baseline=baseline,variant=variant,scene=scene,accuracy_delta_pp=(scores[variant]['scenes'][scene]['accuracy']-scores[baseline]['scenes'][scene]['accuracy'])*100,**counts))
    for filename,values in [('target_summary.csv',summary),('target_detailed_metrics.csv',records),('target_confusion_matrices.csv',cms),('target_seed_aggregates.csv',aggregates),('target_paired_predictions.csv',pairs)]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    p=lambda v:f'{v:.3f}'
    lines=['# CORE90 V2目标域测试集详细结果','',
        f"**本报告为{len(rows)}个固定E200模型的目标接收机测试，非source验证集。**目标RX={m['target_rxs']}，day={m['target_days']}；每场景{m['samples_per_scene']:,}条，四场景共{m['samples_per_scene']*4:,}条/模型。",'',
        '6类闭集全部注册类argmax，无support适配、无测试拟合、无选模或参数更新。本目标集已被历史研究接触，口径为previously_exposed_benchmark_recheck；不是全新未见确认集，也不是Phase2少样本适配/未知拒识结果。训练seed分别列出，信道增强seed固定392002。','',
        '## 1. 目标域Accuracy（%）','',
        'A=ordinary、adv=0；B=ordinary、adv=0.35；C=B8-D1、adv=0；D=B8-D1、adv=0.35。','',
        table(['模型','seed','clean','clear','low_elev','rain','LEO均值','最弱LEO RX'],[[r['method'],r['seed'],*[p(r[s+'_accuracy_pct']) for s in scenes],p(r['leo_mean_accuracy_pct']),p(r['worst_leo_rx_accuracy_pct'])] for r in summary]),
        '## 2. 目标域Macro-F1（%）','',
        table(['模型','seed',*scenes,'LEO均值'],[[r['method'],r['seed'],*[p(r[s+'_macro_f1_pct']) for s in scenes],p(r['leo_mean_macro_f1_pct'])] for r in summary]),
        '## 3. 跨seed描述性汇总','',
        '均值±样本标准差；单seed不报告标准差。不将同一采集组内窗口当作独立seed来估计置信区间。','',
        table(['方法','指标','seed数','均值±SD（%）'],[[r['method'],r['metric'],r['n_seeds'],p(r['mean_pct'])+(' ± '+p(r['sample_std_pct']) if r['sample_std_pct'] is not None else '（单seed）')] for r in aggregates]),
        '## 4. 同seed配对差值','',
        table(['比较','seed','clean Δpp','clear Δpp','low_elev Δpp','rain Δpp','LEO均值 Δpp'],[[pair['variant'].rsplit('_seed',1)[0]+' − '+pair['baseline'].rsplit('_seed',1)[0],m['training_seeds'][pair['variant']],*[f"{(scores[pair['variant']]['scenes'][s]['accuracy']-scores[pair['baseline']]['scenes'][s]['accuracy'])*100:+.3f}" for s in scenes],f"{(scores[pair['variant']]['leo_mean_accuracy']-scores[pair['baseline']]['leo_mean_accuracy'])*100:+.3f}"] for pair in d['paired_predictions']]),
        '逐样本对错翻转、预测不一致数和最大置信度差见target_paired_predictions.csv。C−A是关闭对抗项的负对照；D−B才对应相同adv=0.35下的B8-D1效应。','',
        '## 5. 同模型source与target落差','',
        table(['模型','source clean','target clean','差值pp','source LEO均值','target LEO均值','差值pp'],[[row,p(d['source_scores'][row]['scenes']['clean']['accuracy']*100),p(scores[row]['scenes']['clean']['accuracy']*100),p((scores[row]['scenes']['clean']['accuracy']-d['source_scores'][row]['scenes']['clean']['accuracy'])*100),p(d['source_scores'][row]['leo_mean_accuracy']*100),p(scores[row]['leo_mean_accuracy']*100),p((scores[row]['leo_mean_accuracy']-d['source_scores'][row]['leo_mean_accuracy'])*100)] for row in rows]),
        'source与target的接收机和样本组成不同，上述差值仅描述泛化落差。','']
    for scope,key,label in [('per_rx','target_rxs','RX'),('per_day','target_days','day'),('per_tx_accuracy',None,'TX标签')]:
        for scene in scenes:
            groups=m[key] if key else range(m['num_classes'])
            lines += [f'## {scene}逐{label}Accuracy（%）','',table(['模型',*[str(g) for g in groups]],[[row,*[p((scores[row]['scenes'][scene][scope][str(g)]['accuracy'] if key else scores[row]['scenes'][scene][scope][str(g)])*100) for g in groups]] for row in rows]),'']
    lines+=['## 完整性与原始数据','',
        f"- 全部{len(rows)}模型×4场景先完成预测封存，再统一连接truth评分；总计{sum(v['predictions'] for v in d['verification'].values()):,}条预测。",
        f"- 独立遍历全部预测重新计算混淆矩阵及Accuracy/Macro-F1，最大绝对评分误差={max(v['max_metric_error'] for v in d['verification'].values())}。",
        '- 每场景每RX24000、每day42000、每TX标签28000条。四种view共享168000条基础IQ，不能当作672000条独立物理样本。',
        f"- 目标预测及评分耗时{d['complete']['total_seconds']/60:.2f}分钟，共享GPU负载下记录，不是隔离吞吐测量。",
        '- 其余V2模型未纳入本次固定10模型批次，不用缺失结果作完整矩阵结论。',
        '- [总体指标CSV](target_summary.csv)','- [逐RX/TX/day及组内TX指标CSV](target_detailed_metrics.csv)',
        '- [全部混淆矩阵CSV](target_confusion_matrices.csv)','- [跨seed汇总CSV](target_seed_aggregates.csv)',
        '- [逐样本配对翻转汇总](target_paired_predictions.csv)','- [完整评分与闭合证据](target_results.json)','',
        '状态：TARGET_ARTIFACTS_COMPLETE / VERIFIED；科学口径：已接触目标基准的固定模型闭集复核，无自动科学晋级。']
    (out/'target_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(models=len(rows),detail_rows=len(records),confusion_cells=len(cms),summary=summary),indent=2))

if __name__=='__main__':main()
