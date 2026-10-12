"""All fixed arms/seeds and resources; no candidate selection from scores."""
import statistics as st,time,json
from . import design as d

def build(result,source):
    ids={r['row_id'] for r in d.rows()}
    if result['status']!='COMPLETE' or set(result['completed_rows'])!=ids or not all(result['checks'].values()):raise ValueError('Incomplete scoring')
    if source['status']!='COMPLETE' or source['errors'] or source['pending'] or source['total_steps']!=66600 or {r['row_id'] for r in source['rows']}!=ids:raise ValueError('Incomplete source logs')
    for row in source['rows']:
        ex=row['resolved_execution']
        if row['stdout_errors'] or ex['explicit_CFO_identity_input'] or not ex['U_iterator'] or ex['teacher_source']!='peer' or ex['EMA'] or ex['satellite_generator'] or ex['auxiliary_neural_networks']:raise ValueError('Unexpected execution')
    primary=next(s for s in result['summary'] if s['arm']==d.PRIMARY)
    lines=['# 跨 RX 身份关系约束：完整测试','',f"15/15行、66600完整源更新、2520000次clean固定预测独立truth-last评分。主方案{d.PRIMARY}三种子均值{primary['accuracy_mean']:.4%}，84%目标"+('达到。' if primary['accuracy_mean']>=.84 else '未达到。'),'', '原生旧门控＋cos随机学生20轮，显式CFO slot27恒零。每行保留身份CE和同两固定源教师ALL软KL，另外每4次主更新读出30个独立源L。合并/分RX原型×最终/浅中最终三层固定四方案；L30_CE以同额外样本预算直接训练CE。999次源参考，总29970L读出。固定final20自动clean，不使用EMA/卫星/Fishr/辅助头。', '', '|方案|seed2701|seed2702|seed2703|均值|SD百分点|','|---|---:|---:|---:|---:|---:|']
    overall={(r['arm'],r['model_seed']):r for r in result['records']+result['parent_records'] if r['dimension']=='overall'}
    for arm in ('CE','SSL',*d.ARMS):
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
    lines+=['','|方案|训练含源诊断分钟|推理秒|最大已分配显存MiB|额外参考L/前向|参考有梯度前向|','|---|---:|---:|---:|---:|---:|']
    for arm in d.ARMS:
        rows=[r for r in result['resources'] if r['row_id'].startswith(arm+'-s')]
        if len(rows)!=3:raise ValueError('Missing resource seed')
        resources=[r['source']['measured_training_resources'] for r in rows];p=resources[0]
        lines.append(f"|{arm}|{st.mean(r['identity_training_and_source_diagnostics_seconds'] for r in resources)/60:.2f}|{st.mean(r['prediction']['inference_seconds']['clean'] for r in rows):.2f}|{max(r['peak_cuda_memory_bytes'] for r in resources)/1024**2:.2f}|{p['relation_L']}/{p['relation_forwards']}|{p['relation_reference_gradient_forwards']}|")
    cache=result['teacher_cache']
    if not cache or cache['packet_occurrences']!=170100 or cache['status']!='COMPLETE':raise ValueError('Incomplete fixed teacher cache')
    lines+=['',f"固定共用源教师U缓存来自上一批（170100样本模型读出，{cache['total_seconds']:.2f}秒，{cache['output_bytes']}字节）；本批不重新生成，不按本批每行重复计。学生20＋两源老师40＝60历史模型轮；目标推理只用学生。",'', '关系梯度经query实际浅time_down/中t2.pool均值和population std及final_joint回到身份网络；原型eval/no_grad、无需辅助头，源L支持与query PID完全独立且每query仅比较其他4RX。参考30个L覆盖6TX×5RX，循环day，tau.1；每层所有6TX竞争，组合等层平均、weight1乘固定ramp。身份CE始终保留。L30_CE控制样本读出数量匹配，但训练CE参考有梯度/普通dropout，关系支持eval/no_grad，因此不称完全同算力。所有额外资源实测单列。','', '源关系准确率/margin与身份头准确率/margin分开见source_trajectory.md，不以关系loss下降宣称泛化有效。target每seed/RX/TX/day、混淆/NLL与相对CE纠错/破坏见完整JSON；与同输入SSL及L30_CE控制的配对差列于上表。既有开发基准已暴露，非盲测；全部方案保留，不挑最佳seed、不回流调参。有限作用模型/新U结构任务属于下一独立阶段，尚未用于身份训练。']
    return '\n'.join(lines)+'\n'

def finish():
    from .collect_results import analyze,markdown
    from .analyze_source import analyze as source_analyze,report
    from .source import expected_counts
    configs=[d.config(r) for r in d.rows()];source=source_analyze(d.BASE.as_posix(),configs,{c['row_id']:expected_counts(c) for c in configs})
    value=dict(completion={'status':'EXPERIMENT_COMPLETE'},read_at=time.time(),teacher_cache=d.read(d.e.BASE/'peer_cache/completion.json'),rows=[],aggregate={'scores.json':d.read(d.BASE/'scores.json')})
    for c in configs:
        root=d.BASE/c['row_id'];score=d.BASE/'row_scoring'/c['row_id']
        value['rows'].append(dict(row_id=c['row_id'],completion=d.read(root/'completion.json'),source_completion=d.read(root/'source/completion.json'),freeze=d.read(root/'source_frozen.json'),artifacts={name:d.read(score/name) for name in ['scores.json','day_scores.json','parent_scores.json','resources.json']}))
    result=analyze(value);text=build(result,source)
    d.write(d.BASE/'evidence/source_full_analysis.json',source);d.write(d.BASE/'evidence/test_results_analysis.json',result)
    (d.BASE/'source_trajectory.md').write_text(report(source),encoding='utf-8');(d.BASE/'test_results.md').write_text(markdown(result),encoding='utf-8');(d.BASE/'final_analysis.md').write_text(text,encoding='utf-8');(d.BASE/'report.md').write_text(text,encoding='utf-8')
    spec=d.read(d.BASE/'experiment.json');spec['status']='ANALYZED'
    for row in spec['rows']:row['status']='TRAINED_AND_SCORED'
    d.write(d.BASE/'experiment.json',spec)
    with (d.BASE/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='ANALYZED',time=time.time(),rows=15,source_steps=66600,target_feedback=False))+'\n')
    return dict(status='VERIFIED',rows=15,source_steps=66600)

if __name__=='__main__':print(json.dumps(finish()))
