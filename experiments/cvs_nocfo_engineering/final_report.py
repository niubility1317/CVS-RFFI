"""All fixed arms/seeds and resources; no candidate selection from scores."""
import statistics as st,time,json
from . import design as d

def build(result,source):
    ids={r['row_id'] for r in d.rows()}
    if result['status']!='COMPLETE' or set(result['completed_rows'])!=ids or not all(result['checks'].values()):raise ValueError('Incomplete scoring')
    if source['status']!='COMPLETE' or source['errors'] or source['pending'] or source['total_steps']!=106560 or {r['row_id'] for r in source['rows']}!=ids:raise ValueError('Incomplete source logs')
    for row in source['rows']:
        ex=row['resolved_execution']
        if row['stdout_errors'] or ex['explicit_CFO_identity_input'] or not ex['U_iterator'] or ex['teacher_source']!='peer' or ex['EMA'] or ex['satellite_generator'] or ex['auxiliary_neural_networks']:raise ValueError('Unexpected execution')
    primary=next(s for s in result['summary'] if s['arm']==d.PRIMARY)
    lines=['# 无显式 CFO 的匹配 SSL／工程探索：完整测试','',f"24/24 行完成、106560 次源更新、4032000 次 clean 固定预测独立truth-last评分。主方案 {d.PRIMARY} 三种子均值 {primary['accuracy_mean']:.4%}，84%目标"+('达到。' if primary['accuracy_mean']>=.84 else '未达到。'),'',
        '全部随机原生身份学生20 epoch，固定final20；同输入/代码路径CE对照来自先发布三行无CFO基线。除CE外各学生使用另外两seed固定源CE老师ALL soft KL，444步L预热、888步ramp；没有EMA、卫星或Fishr。CE对照没有U读出，SSL对照没有工程目标；U及额外计算成本单列。原始IQ仍可能含残余频率。','',
        '|方案|seed2701|seed2702|seed2703|均值|SD百分点|','|---|---:|---:|---:|---:|---:|']
    overall={(r['arm'],r['model_seed']):r for r in result['records']+result['parent_records'] if r['dimension']=='overall'}
    for arm in ('CE',*d.ARMS):
        rows=[overall[arm,s] for s in d.SEEDS];values=[r['accuracy'] for r in rows]
        lines.append('|'+arm+'|'+'|'.join(f'{v:.4%}' for v in values)+f'|{st.mean(values):.4%}|{100*st.stdev(values):.4f}|')
    lines+=['','|同seed比较|seed2701差|seed2702差|seed2703差|均差（百分点）|','|---|---:|---:|---:|---:|']
    for a,b in d.PAIRS:
        values=[overall[a,s]['accuracy']-overall[b,s]['accuracy'] for s in d.SEEDS]
        lines.append('|'+a+'−'+b+'|'+'|'.join(f'{v*100:+.4f}' for v in values)+f'|{100*st.mean(values):+.4f}|')
    for dim,label,records in [('receiver','RX',result['records']),('transmitter','TX',result['records']),('day','day',result['day_records'])]:
        lines+=['','|'+label+'|'+'|'.join(d.ARMS)+'|','|---|'+'|'.join(['---:']*len(d.ARMS))+'|']
        for group in sorted({r['stratum'] for r in records if r['dimension']==dim}):
            values=[]
            for arm in d.ARMS:
                subset=[r for r in records if r['dimension']==dim and r['stratum']==group and r['arm']==arm]
                if {r['model_seed'] for r in subset}!=set(d.SEEDS):raise ValueError('Missing stratum seeds')
                values.append(f"{st.mean(r['accuracy'] for r in subset):.4%}")
            lines.append('|'+str(group)+'|'+'|'.join(values)+'|')
    lines+=['','|方案|训练含源诊断分钟|推理秒|最大已分配显存MiB|额外SAM前向|隐藏混合／clean replay前向|','|---|---:|---:|---:|---:|---:|']
    for arm in d.ARMS:
        rows=[r for r in result['resources'] if r['row_id'].startswith(arm+'-s')]
        if len(rows)!=3:raise ValueError('Missing resource seed')
        resources=[r['source']['measured_training_resources'] for r in rows];p=resources[0]
        lines.append(f"|{arm}|{st.mean(r['identity_training_and_source_diagnostics_seconds'] for r in resources)/60:.2f}|{st.mean(r['prediction']['inference_seconds']['clean'] for r in rows):.2f}|{max(r['peak_cuda_memory_bytes'] for r in resources)/1024**2:.2f}|{p['sam_second_forwards']}|{p['hidden_mix_steps']}/{p['clean_replay_steps']}|")
    cache=result['teacher_cache']
    if not cache or cache['packet_occurrences']!=170100 or cache['status']!='COMPLETE':raise ValueError('Incomplete fixed teacher cache')
    lines+=['',f"共享三源教师U缓存：170100模型样本读出，{cache['total_seconds']:.2f}秒，{cache['output_bytes']}字节；生成成本只计一次。每学生依赖其他两源教师，学生20＋教师40＝60历史模型轮，推理只用学生。",'','L_REPLAY和hidden mix都额外3996次L128前向／反向，ramp、权重和clean dropout重放匹配。隐藏混合额外供体/Beta/层hook成本实测在墙钟中，不能称为完全相同计算。SAM两个完整目标前反向、一个正式更新，rho.02，全部可训练身份参数参与。固定工程超参沿用新报告前设计，不由目标分数改变。','',
        '所有逐seed RX/TX/day、confusion、NLL、CE相对纠错/破坏、资源见evidence/test_results_analysis.json；全部106560步源证据另列。全部方案与负结果保留。已暴露开发benchmark，非盲测；不选最佳seed或回流调参。关系与有限作用网络属于后续独立阶段，未混入本批。']
    return '\n'.join(lines)+'\n'

def finish():
    from .collect_results import analyze,markdown
    from .analyze_source import analyze as source_analyze,report
    from .source import expected_counts
    configs=[d.config(r) for r in d.rows()];source=source_analyze(d.BASE.as_posix(),configs,{c['row_id']:expected_counts(c) for c in configs})
    value=dict(completion={'status':'EXPERIMENT_COMPLETE'},read_at=time.time(),teacher_cache=d.read(d.BASE/'peer_cache/completion.json'),rows=[],aggregate={'scores.json':d.read(d.BASE/'scores.json')})
    for c in configs:
        root=d.BASE/c['row_id'];score=d.BASE/'row_scoring'/c['row_id']
        value['rows'].append(dict(row_id=c['row_id'],completion=d.read(root/'completion.json'),source_completion=d.read(root/'source/completion.json'),freeze=d.read(root/'source_frozen.json'),artifacts={name:d.read(score/name) for name in ['scores.json','day_scores.json','parent_scores.json','resources.json']}))
    result=analyze(value);text=build(result,source)
    d.write(d.BASE/'evidence/source_full_analysis.json',source);d.write(d.BASE/'evidence/test_results_analysis.json',result)
    (d.BASE/'source_trajectory.md').write_text(report(source),encoding='utf-8');(d.BASE/'test_results.md').write_text(markdown(result),encoding='utf-8');(d.BASE/'final_analysis.md').write_text(text,encoding='utf-8');(d.BASE/'report.md').write_text(text,encoding='utf-8')
    spec=d.read(d.BASE/'experiment.json');spec['status']='ANALYZED'
    for row in spec['rows']:row['status']='TRAINED_AND_SCORED'
    d.write(d.BASE/'experiment.json',spec)
    with (d.BASE/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='ANALYZED',time=time.time(),rows=24,source_steps=106560,target_feedback=False))+'\n')
    return dict(status='VERIFIED',rows=24,source_steps=106560)

if __name__=='__main__':print(json.dumps(finish()))
