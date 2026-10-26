"""Add pooled disjoint-population statistics to the PAMF branch IEEE report."""
import argparse,gzip,json,statistics,sys
from pathlib import Path
ROOT=Path('E:/type10-7');sys.path.insert(0,str(ROOT))
from tools.reports import pamf_branch_ieee_focus as focus
OUT=ROOT/'exports/reports/20261026-PAMF-branch-IEEE-r23'
FROZEN=ROOT/'exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/frozen_statistics/analysis/score_inputs.jsonl.gz'
TASKS=('cross_rx','cross_both');COUNTS=dict(zip(TASKS,(126000,42000)))
TN=dict(zip(TASKS,('跨接收机','跨日跨接收机')))
ARMS,NAMES,EN,SEEDS,SCENES=focus.ARMS,focus.NAMES,focus.EN,focus.SEEDS,focus.SCENES
focus.OUT=OUT

def metrics(cm):
    n=sum(map(sum,cm));accuracy=100*sum(cm[i][i] for i in range(6))/n
    f=[]
    for i in range(6):
        denominator=sum(cm[i])+sum(cm[j][i] for j in range(6));f.append(2*cm[i][i]/denominator if denominator else 0.)
    return {'accuracy':accuracy,'macro_f1':100*statistics.mean(f)}

def prepare():
    focus.extract();records=[];seen=set();matrices={}
    with gzip.open(FROZEN,'rt',encoding='utf-8') as f:
        for line in f:
            if 'F0-CE_' not in line:continue
            item=json.loads(line);row=item['row']
            if row['protocol_id']!='F0' or row['method'] not in {'CE_'+a for a in ARMS}:continue
            a=row['method'][3:];seed=int(row['model_seed']);assert (a,seed) not in seen;seen.add((a,seed))
            assert item['score']['status']=='SCORED' and item['score']['truth_last'] is True
            for result in item['score']['results']:
                if result['readout']!='raw' or result['dimension']!='overall' or result['stratum']!='ALL' or result['test'] not in TASKS:continue
                key=(a,seed,result['test'],result['view']);assert key not in matrices
                cm=result['confusion'];assert len(cm)==6 and all(len(r)==6 for r in cm)
                assert result['count']==sum(map(sum,cm))==COUNTS[result['test']]
                m=metrics(cm)
                assert abs(m['accuracy']-100*result['accuracy'])<1e-10 and abs(m['macro_f1']-100*result['macro_f1'])<1e-10
                record=dict(structure=a,seed=seed,population=result['test'],view=result['view'],reader='PAMF_direct',count=result['count'],confusion=cm,frozen_accuracy=result['accuracy']*100,frozen_macro_f1=result['macro_f1']*100)
                records.append(record);matrices[key]=cm
            if len(seen)==18:break
    assert len(records)==252 and len(seen)==18
    with gzip.open(OUT/'inputs/population_confusions.jsonl.gz','wt',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    population=[];pooled=[];pooled_cm=[]
    for a in ARMS:
        for seed in SEEDS:
            all_values={}
            for v in SCENES:
                cm=[[sum(matrices[a,seed,t,v][i][j] for t in TASKS) for j in range(6)] for i in range(6)]
                assert sum(map(sum,cm))==168000
                pooled_cm.append(dict(structure=a,seed=seed,view=v,count=168000,confusion=cm));all_values[v]=metrics(cm)
                for m,x in all_values[v].items():pooled.append(dict(structure=a,display_name=NAMES[a],seed=seed,view=v,metric=m,percent=x,packets_per_scene=168000))
            for m in focus.METRICS:pooled.append(dict(structure=a,display_name=NAMES[a],seed=seed,view='six_practical_mean',metric=m,percent=statistics.mean(all_values[v][m] for v in SCENES[1:]),packets_per_scene=168000))
            for t in TASKS:
                values={v:metrics(matrices[a,seed,t,v]) for v in SCENES}
                for v in (*SCENES,'six_practical_mean'):
                    for m in focus.METRICS:population.append(dict(structure=a,seed=seed,population=t,view=v,metric=m,percent=statistics.mean(values[s][m] for s in SCENES[1:]) if v=='six_practical_mean' else values[v][m],packets_per_scene=COUNTS[t]))
    with gzip.open(OUT/'inputs/pooled_confusions.jsonl.gz','wt',encoding='utf-8') as f:
        for r in pooled_cm:f.write(json.dumps(r)+'\n')
    focus.write(OUT/'tables/population_per_seed.csv',population);focus.write(OUT/'tables/pooled_per_seed.csv',pooled)
    ix={(r['structure'],r['view'],r['metric'],r['seed']):r['percent'] for r in pooled};summary=[];paired=[]
    for a in ARMS:
        for v in focus.VIEWS:
            for m in focus.METRICS:
                summary.append(dict(structure=a,display_name=NAMES[a],view=v,metric=m,**focus.stats([ix[a,v,m,s] for s in SEEDS])))
                if a!='full':paired.append(dict(structure=a,display_name=NAMES[a],view=v,metric=m,definition='variant_minus_full',**focus.stats([ix[a,v,m,s]-ix['full',v,m,s] for s in SEEDS])))
    focus.write(OUT/'tables/pooled_summary.csv',summary);focus.write(OUT/'tables/pooled_variant_minus_full.csv',paired)
    figure=[]
    for a in ARMS:
        for v in ('clean','six_practical_mean'):
            rr=next(r for r in summary if r['structure']==a and r['view']==v and r['metric']=='accuracy')
            diff=[ix[a,v,'accuracy',s]-ix['full',v,'accuracy',s] for s in SEEDS]
            figure.append(dict(structure=a,display_name=NAMES[a],view=v,mean=rr['mean'],sd=rr['sd'],seed_values=rr['seed_values'],delta_mean=statistics.mean(diff),delta_sd=statistics.stdev(diff),delta_seed_values=';'.join(map(str,diff))))
    focus.write(OUT/'tables/figure_composite.csv',figure)
    focus.write(OUT/'tables/population_weights.csv',[dict(population=t,display_name=TN[t],packets_per_scene=COUNTS[t],weight=COUNTS[t]/168000) for t in TASKS])
    meta=json.loads((OUT/'inputs/scope.json').read_text(encoding='utf-8'))
    meta.update(scope='PAMF direct: joint-domain primary results plus pooled two disjoint populations',additional_populations=list(TASKS),population_counts=COUNTS,pooled_packets_per_scene=168000,pooled_accuracy='Trace of summed population confusion matrices / 168000',pooled_macro_f1='Recomputed from summed six-class confusion matrices; not an average of population F1',aggregation_order='Per seed, sum populations within each scene; compute metrics; mean six practical scenes; then three-seed mean and sample SD',no_additional_models=True)
    (OUT/'inputs/scope.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(population_CMs=252,pooled_CMs=126,pooled_samples_per_scene=168000,new_training=0,new_predictions=0)),flush=True)

def figures():
    focus.figures()
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    rr=list(focus.rows(OUT/'tables/figure_composite.csv'))
    fig,axes=plt.subplots(1,2,sharey=True,figsize=(7.16,2.85));fig.subplots_adjust(left=.265,right=.985,bottom=.19,top=.82,wspace=.20)
    for i,a in enumerate(ARMS):
        for offset,view,marker,filled in [(-.13,'clean','o',False),(.13,'six_practical_mean','s',True)]:
            r=next(r for r in rr if r['structure']==a and r['view']==view)
            color='#3D5B73' if view=='clean' else '#7C9EA8'
            axes[0].errorbar(float(r['mean']),i+offset,xerr=float(r['sd']),fmt=marker,ms=4,mfc=color if filled else 'white',mec=color,ecolor=color,elinewidth=.8,capsize=2)
            if a=='full':continue
            mean=float(r['delta_mean']);color=focus.LOSS if mean<0 else focus.GAIN if mean>0 else focus.NEUTRAL
            axes[1].errorbar(mean,i+offset,xerr=float(r['delta_sd']),fmt=marker,ms=4,mfc=color if filled else 'white',mec=color,ecolor=color,elinewidth=.8,capsize=2)
    axes[0].set(yticks=range(6),yticklabels=[EN[a] for a in ARMS],xlabel='Pooled accuracy (%)');axes[0].invert_yaxis()
    axes[1].scatter([0],[0],marker='D',color=focus.NEUTRAL,s=12,zorder=3);axes[1].axvline(0,color='#46515C',ls='--',lw=.75);axes[1].set_xlabel('Variant - PAMF (pp)')
    for ax in axes:ax.grid(axis='x',alpha=.15,lw=.5);ax.xaxis.set_major_locator(MaxNLocator(4))
    legend=[Line2D([],[],marker='o',color=focus.NEUTRAL,ls='',mfc='white',label='Clean'),Line2D([],[],marker='s',color=focus.NEUTRAL,ls='',label='Six-scene mean'),Line2D([],[],color=focus.LOSS,lw=2,label='Negative: degradation'),Line2D([],[],color=focus.GAIN,lw=2,label='Positive: improvement')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.58,1.0),ncol=2,frameon=False,columnspacing=1.6,handlelength=1.8)
    for ext in ('pdf','svg','png'):fig.savefig(OUT/'figures'/('composite_testing.'+ext),dpi=320)
    plt.close(fig)
    manifest=json.loads((OUT/'figures/manifest.json').read_text(encoding='utf-8'))
    manifest.append(dict(name='composite_testing',caption='两类互斥测试综合：每景合并168000条样本。左为综合准确率，右为同种子“变体减完整PAMF”；完整模型记作0参考。每点为3个训练种子均值；误差线左为指标标准差，右为配对差标准差。clean与六景均值分列。',source_csv='tables/figure_composite.csv',style='IEEE double-column; editable Times New Roman text'))
    (OUT/'figures/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(figures=4)),flush=True)

def document():
    focus.document()
    summary=list(focus.rows(OUT/'tables/pooled_summary.csv'));paired=list(focus.rows(OUT/'tables/pooled_variant_minus_full.csv'))
    def get(a,v='six_practical_mean',m='accuracy'):return next(r for r in summary if (r['structure'],r['view'],r['metric'])==(a,v,m))
    def pair(a,v='six_practical_mean',m='accuracy'):return next(r for r in paired if (r['structure'],r['view'],r['metric'])==(a,v,m))
    def fmt(r,signed=False):return f"{float(r['mean']):+.2f}±{float(r['sd']):.2f}" if signed else f"{float(r['mean']):.2f}±{float(r['sd']):.2f}"
    definition='综合测试仅合并两类互斥群体：跨接收机126000包、跨日跨接收机42000包，每个场景共168000包。准确率按实际样本合并计算，两类权重分别为75%和25%；不是两个准确率的简单平均。宏平均F1先合并两类六分类混淆矩阵，再按六类重算，不能直接加权或平均已有F1。'
    order='合并在每个模型种子、每个场景内独立完成。随后对六个信道场景等权平均，最后对三个模型种子计算均值与样本标准差；相对变化仍先按种子计算“变体减完整PAMF”。同一18个模型同时支持分群与综合统计，不增加模型重复数。'
    finding='完整PAMF的综合clean准确率为'+fmt(get('full','clean'))+'%，综合六景准确率为'+fmt(get('full'))+'%，综合六景宏平均F1为'+fmt(get('full',m='macro_f1'))+'%。去除功放相关支路的综合六景准确率变化为'+fmt(pair('NO_PA'),True)+'个百分点；去除路径描述支路的变化为'+fmt(pair('NO_SIGNATURE'),True)+'个百分点。后者与跨日跨接收机分群中的+0.85±0.25个百分点方向不同，说明分群与综合结论应分别保留。'
    limitation='综合结果反映既有样本分布，跨接收机群体占75%，对综合值影响较大；它不能代替各群体结果。综合样本全部来自目标接收机，日期包含训练中已见日期和另一较早未见日期。所有综合结果均来自既有冻结混淆矩阵，未新增预测、真值访问或模型训练。'
    scope_old='本报告只使用日期与接收机同时未见的测试群体，以及PAMF直接识别的冻结结果。'
    scope_new='本报告主分析采用日期与接收机同时未见的测试群体，并补充两类互斥测试群体的综合统计；全部结果均为PAMF直接识别。'
    limits_old='结论只适用于当前六类划分、监督分类目标和跨日跨接收机测试群体；不能外推到其他群体或训练目标。'
    limits_new='结论只适用于当前六类划分、监督分类目标、已列出的分群结果和既有样本权重下的综合测试；不能外推到其他训练目标。'
    md=(OUT/'report.md').read_text(encoding='utf-8');assert scope_old in md and limits_old in md
    md=md.replace(scope_old,scope_new).replace(limits_old,limits_new)
    parts=['## 两类测试综合统计','',definition,'',order,'','| 网络结构 | 综合clean准确率（%） | 综合clean宏平均F1（%） | 综合六景准确率（%） | 综合六景宏平均F1（%） |','|---|---|---|---|---|']
    combos=[('clean','accuracy'),('clean','macro_f1'),('six_practical_mean','accuracy'),('six_practical_mean','macro_f1')]
    paper=[]
    for a in ARMS:
        values=[NAMES[a],*[fmt(get(a,v,m)) for v,m in combos]];parts.append('| '+' | '.join(values)+' |');paper.append(dict(zip(('网络结构','综合clean准确率','综合clean宏平均F1','综合六景准确率','综合六景宏平均F1'),values)))
    parts.extend(['','![两类测试综合图](figures/composite_testing.png)','',finding,'',limitation,''])
    focus.write(OUT/'tables/论文表_综合结果.csv',paper)
    md=md.replace('## 结果解读','\n'.join(parts)+'\n## 结果解读')
    (OUT/'report.md').write_text(md,encoding='utf-8')
    tex=(OUT/'ieee/main.tex').read_text(encoding='utf-8');assert focus.tex(scope_old) in tex
    tex=tex.replace(focus.tex(scope_old),focus.tex(scope_new)).replace(focus.tex(limits_old),focus.tex(limits_new))
    abstract_start=tex.index(r'\begin{abstract}')+len(r'\begin{abstract}');abstract_end=tex.index(r'\end{abstract}')
    abstract='本报告呈现PAMF直接识别在跨日跨接收机群体上的五项网络分支消融，并增加跨接收机126000包与跨日跨接收机42000包两类互斥群体的综合统计。每组含三个模型种子。综合指标先按种子和场景合并168000条样本对应的混淆矩阵，再计算准确率与宏平均F1；相对变化为“变体减完整网络”，负值表示下降。分群与综合结果均保留种子离散度和反向结果。'
    tex=tex[:abstract_start]+'\n'+focus.tex(abstract)+'\n'+tex[abstract_end:]
    section=[r'\section{两类测试的综合统计}',focus.tex(definition),focus.tex(order),r'\begin{equation}C_{s,v}^{\mathrm{all}}=\sum_{g=1}^{2}C_{s,v}^{(g)},\quad N=168000.\end{equation}',r'\begin{equation}\operatorname{MacroF1}(C)=\frac{1}{6}\sum_{k=1}^{6}\frac{2C_{kk}}{\sum_j C_{kj}+\sum_i C_{ik}}.\end{equation}',focus.tex(finding),focus.tex(limitation),r'表~\ref{tab:pooled}~和图~\ref{fig:composite_testing}~给出综合结果。',r'\begin{table*}[!t]',r'\caption{两类测试合并统计（\%，每景168000包，三种子均值±样本标准差）。宏平均F1由合并混淆矩阵重算。}',r'\label{tab:pooled}',r'\centering\small',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}',r'\toprule',r'网络结构 & \multicolumn{2}{c}{综合clean} & \multicolumn{2}{c}{综合六景均值} \\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',r' & 准确率 & 宏平均F1 & 准确率 & 宏平均F1 \\',r'\midrule']
    for a in ARMS:
        values=[focus.tex(fmt(get(a,v,m))) for v,m in combos]
        if a=='full':values=[r'\textbf{'+v+'}' for v in values]
        section.append(focus.tex(NAMES[a])+' & '+' & '.join(values)+r' \\')
    section.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table*}'])
    caption=next(r['caption'] for r in json.loads((OUT/'figures/manifest.json').read_text(encoding='utf-8')) if r['name']=='composite_testing')
    section.extend([r'\begin{figure*}[!t]',r'\centering',r'\includegraphics[width=\textwidth]{../figures/composite_testing.pdf}',r'\caption{'+focus.tex(caption)+'}',r'\label{fig:composite_testing}',r'\end{figure*}'])
    tex=tex.replace(r'\section{结论与适用边界}','\n\n'.join(section)+'\n\n'+r'\section{结论与适用边界}')
    (OUT/'ieee/main.tex').write_text(tex,encoding='utf-8')
    (OUT/'README.md').write_text('''# PAMF 网络分支消融：含两类综合测试的 IEEE 版

保留仅PAMF直接识别、五项消融、负值表示下降、无效率内容的要求。主分析仍为跨日跨接收机；综合图仅合并跨接收机126000、跨日跨接收机42000包，每景168000包。

综合准确率来自合并混淆矩阵的正确数／总数，两类权重为75%和25%；宏平均F1从合并矩阵重算。每种子、每景先合并群体，再求六景均值，最后统计三种子均值与标准差。不是简单平均两个群体的准确率或F1。

report.pdf为IEEE双栏报告，ieee/main.tex为可编辑源码，figures含4幅PDF/SVG矢量图和PNG预览。inputs保存原18模型的筛选后冻结指标及252个总体混淆矩阵，不含IQ、权重、逐包预测或真值。pooled_confusions保存126个合并矩阵。tables提供综合与分群逐种子指标、综合配对差及作图数据。

解包可用纯标准库复算：

```text
python -X utf8 -S reproduce/verify_report.py --root .
```

同一18模型复用，新增训练、预测、目标真值访问均为0。新不可覆盖包r23，旧包保留。
''',encoding='utf-8')
    print(json.dumps(dict(status='COMPOSITE_IEEE_SOURCE_WRITTEN',tables=3,figures=4)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','figures','document']);a=p.parse_args();{'prepare':prepare,'figures':figures,'document':document}[a.phase]()
