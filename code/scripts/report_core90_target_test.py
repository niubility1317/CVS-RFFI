"""Create detailed target test tables from the frozen artifact scores."""
import argparse,csv,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);ap.add_argument('--source-report')
    a=ap.parse_args();root=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'frozen_manifest.json').read_text(encoding='utf-8'))
    complete=json.loads((root/'complete.json').read_text(encoding='utf-8'))
    assert complete['complete'] and complete['rows']==manifest['rows']
    scores={r:json.loads((root/(r+'_target_scores.json')).read_text(encoding='utf-8')) for r in manifest['rows']}
    rows=[]
    for r,s in scores.items():
        assert s['complete'] and s['target_evaluated'] and not s['source_only']
        assert s['samples_per_scene']==manifest['samples_per_scene']
        for scene,m in s['scenes'].items():
            for scope,values in [('overall',{'all':m}),('receiver',m['per_rx']),('day',m['per_day'])]:
                for key, v in values.items():
                    rows.append(dict(row=r,scene=scene,scope=scope,group=key,tx='',count=v['count'],
                        accuracy_pct=v['accuracy']*100,macro_f1_pct=v['macro_f1']*100,macro_recall_pct=v['macro_recall']*100))
                    for tx,acc in v['per_tx_accuracy'].items():
                        rows.append(dict(row=r,scene=scene,scope=scope+'_tx',group=key,tx=tx,
                            count=v['per_tx_count'][tx],accuracy_pct=acc*100,macro_f1_pct='',macro_recall_pct=''))
    stem='core90_target15_392005_20260911'
    with (out/(stem+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/(stem+'.json')).write_text(json.dumps(dict(manifest=manifest,complete=complete,scores=scores),ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 15行E200模型目标测试集详细结果','','本报告是未参与训练的目标接收机测试结果；全部15行及四场景预测固定后统一评分。百分数保留两位，CSV/JSON保留完整精度。单seed闭集诊断不支持统计显著性、未知类能力或科学晋级结论。',
        '',f"测试集每场景{manifest['samples_per_scene']:,}条，RX={manifest['target_rxs']}，day={manifest['target_days']}，注册类数={manifest['num_classes']}。四场景共用基础IQ，不能视为四份独立样本；模型训练seed392005，LEO扰动seed{manifest['augmentation_seed']}。",'',
        '## 总体准确率','','|模型|clean|clear|low-elev|rain|LEO均值|最弱LEO|最弱LEO接收机|','|---|---:|---:|---:|---:|---:|---:|---:|']
    scenes=manifest['scenes']
    for r,s in scores.items():
        nums=[s['scenes'][c]['accuracy']*100 for c in scenes]+[s['leo_mean_accuracy']*100,s['leo_worst_accuracy']*100,min(s['scenes'][c]['worst_rx_accuracy'] for c in scenes[1:])*100]
        lines.append('|'+r+'|'+'|'.join(f'{v:.2f}' for v in nums)+'|')
    if a.source_report:
        previous=json.loads(Path(a.source_report).read_text(encoding='utf-8'))
        source={r['row']:r['source_performance'] for r in previous['complete_runs']}
        lines+=['','## 同模型源域V与目标测试集差异','','差值为目标测试减源域V，单位百分点。两者接收机及样本组成不同，差值仅描述泛化落差。','','|模型|源域clean|目标clean|clean差值|源域LEO均值|目标LEO均值|LEO差值|','|---|---:|---:|---:|---:|---:|---:|']
        for r,s in scores.items():
            v=source[r];vc=v['scenes']['clean']['accuracy'];tc=s['scenes']['clean']['accuracy'];vl=v['leo_mean_accuracy'];tl=s['leo_mean_accuracy']
            lines.append('|'+r+'|'+'|'.join(f'{x*100:.2f}' for x in (vc,tc,tc-vc,vl,tl,tl-vl))+'|')
    first=next(iter(scores.values()))['scenes']['clean']
    lines+=['','## 实际测试样本分布','','|分组|样本数/场景|','|---|---:|']
    for rx,v in first['per_rx'].items():lines.append(f"|RX{rx}|{v['count']}|")
    for day,v in first['per_day'].items():lines.append(f"|day{day}|{v['count']}|")
    for tx,n in first['per_tx_count'].items():lines.append(f'|TX{tx}|{n}|')
    lines+=['','## 各场景Macro-F1','','|模型|'+'|'.join(scenes)+'|','|---|'+'---:|'*4]
    for r,s in scores.items():lines.append('|'+r+'|'+'|'.join(f"{s['scenes'][c]['macro_f1']*100:.2f}" for c in scenes)+'|')
    for c in scenes:
        lines+=['',f'## {c}逐接收机准确率','','|模型|'+'|'.join('RX'+str(rx) for rx in manifest['target_rxs'])+'|','|---|'+'---:|'*len(manifest['target_rxs'])]
        for r,s in scores.items():lines.append('|'+r+'|'+'|'.join(f"{s['scenes'][c]['per_rx'][str(rx)]['accuracy']*100:.2f}" for rx in manifest['target_rxs'])+'|')
    for scope,key,label in [('per_day','target_days','日期'),('per_tx_accuracy',None,'发射机')]:
        for c in scenes:
            groups=manifest[key] if key else range(manifest['num_classes'])
            lines+=['',f'## {c}逐{label}准确率','','|模型|'+'|'.join(str(x) for x in groups)+'|','|---|'+'---:|'*len(groups)]
            for r,s in scores.items():
                vals=[s['scenes'][c][scope][str(x)]['accuracy'] if key else s['scenes'][c][scope][str(x)] for x in groups]
                lines.append('|'+r+'|'+'|'.join(f'{v*100:.2f}' for v in vals)+'|')
    lines+=['','## 数据与闭合证据','',f"- 15模型共{manifest['samples_per_scene']*len(scenes)*len(scores):,}条预测，评分完成；耗时{complete['total_seconds']/60:.2f}分钟。",
        f'- CSV共{len(rows)}行，含总体、接收机、日期及各分组内逐TX指标。',
        '- 所有模型面向全部注册类逐样本argmax；不更新参数或buffer，不使用测试集选模。',
        '- 样本共享且按采集分组相关，不能把这些样本当作独立seed计算方法置信度。']
    (out/(stem+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(rows=len(rows),models=len(scores),output=str(out))))

if __name__=='__main__':main()
