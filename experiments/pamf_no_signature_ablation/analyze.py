from collections import defaultdict
from itertools import product
from pathlib import Path
import csv
import statistics
from experiments.cric_dura_comparison.io import read,write
from experiments.cric_dura_comparison.analyze_main import metrics
from .method import ARMS,SEEDS,READOUTS,TASKS,PARAMETERS


def csvfile(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(rt):
    sources=read(Path(rt['run_root'])/'sources.json');values={};per=[];cost=[];serialized=[]
    for rid,entry in sources.items():
        row=entry['row'];s=read(Path(entry['scores'])/'scores.json')
        if not s['truth_last'] or not s['target_feedback_forbidden'] or s['model_seed']!=row['model_seed']:raise ValueError('Scoring independence differs')
        serialized.append(dict(row=row,score=s))
        result=[r for r in s['results'] if r['readout'] in READOUTS]
        for rd,view in product(READOUTS,rt['views']):
            a=[r for r in result if r['readout']==rd and r['view']==view and r['dimension']=='overall']
            if {r['test'] for r in a}!=set(TASKS) or len(a)!=3:raise ValueError('Incomplete three-population matrix')
            cm=[[sum(r['confusion'][i][j] for r in a) for j in range(6)] for i in range(6)]
            count=sum(r['count'] for r in a)
            f1=[2*cm[i][i]/(sum(cm[i])+sum(r[i] for r in cm)) if sum(cm[i])+sum(r[i] for r in cm) else 0. for i in range(6)]
            result.append(dict(test='pooled',readout=rd,view=view,dimension='overall',stratum='ALL',
                               count=count,confusion=cm,accuracy=sum(cm[i][i] for i in range(6))/count,macro_f1=sum(f1)/6))
        for r in result:
            for metric,value in metrics(r,6).items():
                k=(row['arm'],row['model_seed'],r['test'],r['readout'],r['view'],r['dimension'],r['stratum'],metric)
                values[k]=100*value
        bases={(t,rd,dim,st,m) for arm,seed,t,rd,v,dim,st,m in values if arm==row['arm'] and seed==row['model_seed']}
        for t,rd,dim,st,m in bases:
            values[row['arm'],row['model_seed'],t,rd,'six_practical_mean',dim,st,m]=statistics.mean(
                values[row['arm'],row['model_seed'],t,rd,v,dim,st,m] for v in rt['views'] if v!='clean')
        d=read(Path(entry['source_output'])/'complete.json')
        cost.append(dict(row_id=rid,arm=row['arm'],seed=row['model_seed'],reused=row['reuse'],epochs=d['epoch'],updates=d['updates'],
                         worker_seconds=sum(v['seconds'] for v in d['segments']),parameters=PARAMETERS[row['arm']],
                         system_parameters=PARAMETERS[row['arm']]+266951))
    groups=defaultdict(dict)
    for (arm,seed,t,rd,v,dim,st,m),value in sorted(values.items()):
        per.append(dict(arm=arm,seed=seed,test=t,readout=rd,view=v,dimension=dim,stratum=st,metric=m,percent=value))
        groups[arm,t,rd,v,dim,st,m][seed]=value
    summary=[];paired=[]
    for (arm,t,rd,v,dim,st,m),a in sorted(groups.items()):
        if set(a)!=set(SEEDS):raise ValueError('Incomplete three-seed summary')
        common=dict(arm=arm,test=t,readout=rd,view=v,dimension=dim,stratum=st,metric=m)
        summary.append(dict(common,mean=statistics.mean(a.values()),sd=statistics.stdev(a.values())))
        if arm!='BASE':
            delta=[a[s]-groups['BASE',t,rd,v,dim,st,m][s] for s in SEEDS]
            paired.append(dict(common,mean=statistics.mean(delta),sd=statistics.stdev(delta),seed_values=';'.join(map(str,delta))))
    interaction=[]
    for t,rd,v,m in product((*TASKS,'pooled'),READOUTS,(*rt['views'],'six_practical_mean'),('accuracy','macro_f1')):
        a=[values['BASE',s,t,rd,v,'overall','ALL',m]-values['NO_PA',s,t,rd,v,'overall','ALL',m]
           -values['NO_REFERENCE',s,t,rd,v,'overall','ALL',m]+values['NO_PA_REFERENCE',s,t,rd,v,'overall','ALL',m] for s in SEEDS]
        interaction.append(dict(test=t,readout=rd,view=v,metric=m,mean=statistics.mean(a),sd=statistics.stdev(a),seed_values=';'.join(map(str,a))))
    out=Path(rt['run_root'])/'analysis';out.mkdir()
    for name,data in [('per_seed',per),('summary',summary),('paired_vs_baseline',paired),('PA_reference_interaction',interaction),('costs',cost)]:csvfile(out/(name+'.csv'),data)
    write(out/'score_inputs.json',serialized)
    write(out/'complete.json',dict(status='ANALYZED',models=15,new_models=12,reused_models=3,readouts=READOUTS,
          tests=[*TASKS,'pooled'],pooled_packets_per_view=198000,macro_F1_from_pooled_CM=True,
          scope='Exposed engineering benchmark; three F seeds conditional on one frozen DURA-L; no target feedback'))
    names=dict(BASE='去 SIG-TIME 基底',NO_PA='再去功放支路',NO_REFERENCE='再去参考响应支路',
               NO_PA_REFERENCE='同时去功放和参考响应支路',WRONG_REFERENCE='打乱参考模板相位')
    text=['# 监督去 SIG-TIME PAMF 消融结果','',
          '全部 15 个模型单元已完成。12 个新增模型各从零训练 50 轮／11100 更新，基底三种子复用既有冻结产物。',
          '固定同一 DURA-L；三种子为识别器重复，不是三次独立恢复器训练。综合每景 198000 包，宏平均 F1 从合并混淆矩阵重算。','',
          '| 结构 | 直接识别 clean | 直接识别六景 | DURA clean | DURA 六景 |','| --- | --- | --- | --- | --- |']
    for arm in ARMS:
        cells=[]
        for rd,v in product(READOUTS,('clean','six_practical_mean')):
            a=next(x for x in summary if x['arm']==arm and x['test']=='pooled' and x['readout']==rd and x['view']==v and x['dimension']=='overall' and x['metric']=='accuracy')
            cells.append(f"{a['mean']:.3f}±{a['sd']:.3f}%")
        text.append('| '+names[arm]+' | '+' | '.join(cells)+' |')
    text.extend(['','各群体、逐场景、逐 TX/RX/day、三种子配对差和交互项见同目录 CSV。净纠错及原始混淆矩阵见 score_inputs.json。',
                 '参数量、训练成本见 costs.csv；实际时延见 ../timing。删支同时改变容量和计算，不能据此宣称真实硬件参数的因果可辨识性。'])
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
