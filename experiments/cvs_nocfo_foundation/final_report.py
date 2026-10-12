"""Automatic complete source-log and independent scored-result reporting."""
import statistics as st,time,json
from . import design as d

def build(result,source):
    ids={r['row_id'] for r in d.rows()}
    if result['status']!='COMPLETE' or set(result['completed_rows'])!=ids or not all(result['checks'].values()):raise ValueError('Incomplete scoring')
    if source['status']!='COMPLETE' or source['errors'] or source['pending'] or source['total_steps']!=13320 or {r['row_id'] for r in source['rows']}!=ids:raise ValueError('Incomplete source evidence')
    for row in source['rows']:
        ex=row['resolved_execution']
        if row['stdout_errors'] or ex['explicit_CFO_identity_input'] or ex['U_iterator'] or ex['teacher_source']!='none' or ex['EMA'] or ex['satellite_generator'] or ex['auxiliary_neural_networks']:raise ValueError('Unexpected execution')
    rows=[r for r in result['records'] if r['dimension']=='overall'];summary=next(s for s in result['summary'] if s['arm']=='CE')
    if {r['model_seed'] for r in rows}!=set(d.SEEDS) or not summary['complete_three_seeds']:raise ValueError('Missing seed')
    lines=['# 无显式 CFO 纯 CE E20：完整测试结果','',f"3/3 行、13320 次源更新、504000 次 clean 预测独立 truth-last 评分完成。三种子均值 {summary['accuracy_mean']:.4%}；84%目标"+('达到。' if summary['accuracy_mean']>=.84 else '未达到。'),'',
        '原生旧门控随机权重；显式 CFO 坐标27恒零，内部逐包补偿保留。只用合法L6300，无U/教师/EMA/卫星增强/辅助网络。固定20 epoch与final20；V只作健康评估。原始IQ可能仍包含残余频率信息。','',
        '|seed|clean准确率|Macro-F1|相对历史P0纠错／破坏／净变化|','|---|---:|---:|---:|']
    for r in sorted(rows,key=lambda x:x['model_seed']):lines.append(f"|{r['model_seed']}|{r['accuracy']:.4%}|{r['macro_f1']:.4%}|{r['parent_fix_count']}/{r['parent_break_count']}/{r['net_correct_gain']:+d}|")
    lines+=['',f"SD {100*summary['accuracy_sd']:.4f} 个百分点。历史P0输入及训练器不同，只作附带公开参考；差异不能单独归因于CFO屏蔽。",'']
    for dimension,label,records in [('receiver','RX',result['records']),('transmitter','TX',result['records']),('day','day',result['day_records'])]:
        lines+=['|'+label+'|seed2701|seed2702|seed2703|均值|','|---|---:|---:|---:|---:|']
        for stratum in sorted({r['stratum'] for r in records if r['dimension']==dimension}):
            cells={r['model_seed']:r['accuracy'] for r in records if r['dimension']==dimension and r['stratum']==stratum}
            if set(cells)!=set(d.SEEDS):raise ValueError('Missing stratum')
            lines.append('|'+str(stratum)+'|'+'|'.join(f'{cells[s]:.4%}' for s in d.SEEDS)+f'|{st.mean(cells.values()):.4%}|')
        lines.append('')
    lines+=['|seed|训练和源诊断分钟|clean推理秒|峰值已分配显存MiB|','|---|---:|---:|---:|']
    for r in result['resources']:
        resource=r['source']['measured_training_resources'];peak=resource['peak_cuda_memory_bytes'];peak='N/A' if peak is None else f'{peak/1024**2:.2f}'
        lines.append(f"|{r['row_id']}|{resource['identity_training_and_source_diagnostics_seconds']/60:.2f}|{r['prediction']['inference_seconds']['clean']:.2f}|{peak}|")
    lines+=['','全部逐seed混淆矩阵、NLL、分层、资源保留在evidence/test_results_analysis.json；完整13320步源日志汇总独立保存在evidence/source_full_analysis.json。已暴露开发基准，非盲测。所有三行源教师固定用于后续匹配探索，不按本次评分筛选或改变配置。其余24行及关系／作用网络尚未发布。']
    return '\n'.join(lines)+'\n'

def finish():
    from .collect_results import analyze,markdown
    from .analyze_source import analyze as source_analyze,report
    from .source import expected_counts
    configs=[d.config(r) for r in d.rows()]
    source=source_analyze(d.BASE.as_posix(),configs,{c['row_id']:expected_counts(c) for c in configs})
    value=dict(completion={'status':'EXPERIMENT_COMPLETE'},read_at=time.time(),teacher_cache=None,rows=[],aggregate={'scores.json':d.read(d.BASE/'scores.json')})
    for c in configs:
        root=d.BASE/c['row_id'];score=d.BASE/'row_scoring'/c['row_id']
        value['rows'].append(dict(row_id=c['row_id'],completion=d.read(root/'completion.json'),source_completion=d.read(root/'source/completion.json'),freeze=d.read(root/'source_frozen.json'),artifacts={name:d.read(score/name) for name in ['scores.json','day_scores.json','parent_scores.json','resources.json']}))
    result=analyze(value);text=build(result,source)
    d.write(d.BASE/'evidence/source_full_analysis.json',source);d.write(d.BASE/'evidence/test_results_analysis.json',result)
    (d.BASE/'source_trajectory.md').write_text(report(source),encoding='utf-8');(d.BASE/'test_results.md').write_text(markdown(result),encoding='utf-8');(d.BASE/'final_analysis.md').write_text(text,encoding='utf-8')
    spec=d.read(d.BASE/'experiment.json');spec['status']='ANALYZED'
    for row in spec['rows']:row['status']='TRAINED_AND_SCORED'
    d.write(d.BASE/'experiment.json',spec)
    with (d.BASE/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='ANALYZED',time=time.time(),rows=3,source_steps=13320,clean_predictions=504000,target_feedback=False))+'\n')
    (d.BASE/'report.md').write_text(text,encoding='utf-8')
    return dict(status='VERIFIED',rows=3,source_steps=13320)

if __name__=='__main__':print(json.dumps(finish()))
