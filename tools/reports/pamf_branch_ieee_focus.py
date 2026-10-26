"""PAMF-only joint receiver/date generalization; fixed supervised branch subset."""
import argparse,csv,gzip,json,math,statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path('E:/type10-7')
SOURCE=ROOT/'exports/reports/20261025-PAMF-branch-ablation-r20'
OUT=ROOT/'exports/reports/20261026-PAMF-branch-IEEE-r21'
SEEDS=(2026092701,2026092702,2026092703)
ARMS=('full','NO_TIME','NO_SIGNATURE','NO_PA','NO_REFERENCE','WRONG_REFERENCE')
NAMES=dict(zip(ARMS,('完整 PAMF','去除显式时间坐标','去除路径描述支路','去除功放相关支路','去除参考响应支路','打乱参考模板相位')))
EN=dict(zip(ARMS,('PAMF','w/o time coordinate','w/o path descriptor','w/o PA features','w/o reference response','Phase-scrambled template')))
SCENES=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
VIEWS=(*SCENES,'six_practical_mean')
SCENE_NAMES=('clean','高郊区','中郊区','低郊区','高城市','中城市','低城市')
METRICS=('accuracy','macro_f1')
LOSS='#B6574E';GAIN='#307F76';NEUTRAL='#707983'

def rows(path):
    with (gzip.open if path.suffix=='.gz' else open)(path,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)

def write(path,data):
    data=list(data);assert data;path.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open if path.suffix=='.gz' else open)(path,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def stats(a):
    return dict(mean=statistics.mean(a),sd=statistics.stdev(a),seed_values=';'.join(map(str,a)),negative_seeds=sum(x<0 for x in a),positive_seeds=sum(x>0 for x in a))

def extract():
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'inputs').mkdir();(OUT/'tables').mkdir();(OUT/'figures').mkdir();(OUT/'ieee').mkdir()
    keep={'CE_'+a for a in ARMS}
    selected=[r for r in rows(SOURCE/'tables/per_seed_overall.csv') if r['method'] in keep and r['test']=='cross_both' and r['readout']=='raw']
    assert len(selected)==288 and all(int(r['packets_per_view'])==42000 for r in selected)
    write(OUT/'inputs/frozen_per_seed.csv',selected)
    ix={(r['method'][3:],r['view'],r['metric'],int(r['seed'])):float(r['percent']) for r in selected}
    summary=[];paired=[]
    for a in ARMS:
        for v in VIEWS:
            for metric in METRICS:
                summary.append(dict(structure=a,display_name=NAMES[a],view=v,metric=metric,**stats([ix[a,v,metric,s] for s in SEEDS])))
                if a!='full':
                    paired.append(dict(structure=a,display_name=NAMES[a],view=v,metric=metric,definition='variant_minus_full',**stats([ix[a,v,metric,s]-ix['full',v,metric,s] for s in SEEDS])))
    write(OUT/'tables/summary.csv',summary);write(OUT/'tables/variant_minus_full.csv',paired)
    all_strata=[r for r in rows(SOURCE/'tables/all_strata_per_seed.csv.gz') if r['method'] in keep and r['test']=='cross_both' and r['readout']=='raw']
    write(OUT/'tables/per_stratum_per_seed.csv.gz',all_strata)
    groups=defaultdict(dict);fields=('method','view','dimension','stratum','metric')
    for r in all_strata:
        key=tuple(r[k] for k in fields);seed=int(r['seed']);assert seed not in groups[key];groups[key][seed]=float(r['percent'])
    strata=[];strata_pairs=[]
    for key,values in groups.items():
        assert set(values)==set(SEEDS)
        strata.append(dict(zip(fields,key),**stats([values[s] for s in SEEDS])))
        if key[0]!='CE_full':strata_pairs.append(dict(zip(fields,key),definition='variant_minus_full',**stats([values[s]-groups[('CE_full',*key[1:])][s] for s in SEEDS])))
    write(OUT/'tables/per_stratum_summary.csv.gz',strata);write(OUT/'tables/per_stratum_variant_minus_full.csv.gz',strata_pairs)
    write(OUT/'tables/figure_accuracy.csv',[r for r in summary if r['metric']=='accuracy' and r['view'] in SCENES])
    write(OUT/'tables/figure_paired.csv',[r for r in paired if r['view'] in ('clean','six_practical_mean')])
    write(OUT/'tables/figure_scene_changes.csv',[r for r in paired if r['metric']=='accuracy' and r['view'] in SCENES])
    metadata=dict(report_date='2026-10-26',scope='Classification-only supervised PAMF, joint unseen receiver and earlier unseen date',model_seeds=list(SEEDS),structures=list(ARMS),models=18,source_filter={'protocol':'F0','test':'cross_both','readout':'raw','group':'strict_structure'},registered_classes=6,target_receivers=7,packets_per_scene=42000,scenes=list(SCENES),training_epochs=50,optimizer_updates_per_model=11100,default_head_joint=True,delta_definition='variant_minus_full',negative_means='Worse than full PAMF',positive_means='Better than full PAMF',summary_rows=len(summary),paired_rows=len(paired),stratum_seed_rows=len(all_strata),stratum_summary_rows=len(strata),stratum_paired_rows=len(strata_pairs),SIG_TIME_clarification={'NO_TIME':'Remove explicit time coordinate, retain descriptor branch','NO_SIGNATURE':'Remove full SIG-TIME path-descriptor branch; same experiment, not a second removal','SIG_TIME':'Historical single-view control; excluded from this report'},new_training=0,new_prediction=0,new_truth_access=0)
    (OUT/'inputs/scope.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False),flush=True)

def figures():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7,'axes.linewidth':.65,'lines.linewidth':1.15,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False,'savefig.facecolor':'white'})
    summary=list(rows(OUT/'tables/summary.csv'));paired=list(rows(OUT/'tables/variant_minus_full.csv'));manifest=[]
    def save(fig,name,caption,source):
        for ext in ('pdf','svg','png'):fig.savefig(OUT/'figures'/(name+'.'+ext),dpi=320)
        plt.close(fig);manifest.append(dict(name=name,caption=caption,source_csv='tables/'+source,style='IEEE double-column; editable Times New Roman text'))
    colors=['#233C57','#648EA5','#4D8C81','#BE695A','#B19656','#8A79A6']
    markers=['o','s','D','^','v','P'];lines=['-','--','-.',':','--','-.']
    fig,ax=plt.subplots(figsize=(7.16,2.45));fig.subplots_adjust(left=.085,right=.72,bottom=.21,top=.96)
    for a,color,marker,ls in zip(ARMS,colors,markers,lines):
        rr=[next(r for r in summary if r['structure']==a and r['view']==v and r['metric']=='accuracy') for v in SCENES]
        ax.plot(range(7),[float(r['mean']) for r in rr],label=EN[a],color=color,marker=marker,ms=3.5,lw=1.8 if a=='full' else 1.05,ls=ls)
    ax.set(xticks=range(7),xticklabels=['Clean','High S','Mid S','Low S','High U','Mid U','Low U'],ylabel='Accuracy (%)');ax.grid(axis='y',alpha=.18,lw=.5)
    ax.legend(loc='center left',bbox_to_anchor=(1.025,.52),frameon=False,handlelength=2.5,labelspacing=.95)
    save(fig,'seven_scene_accuracy','跨日跨接收机、PAMF直接识别的逐场景准确率。每点为3个训练种子均值。S为郊区，U为城市；不同线型和点形便于黑白打印。','figure_accuracy.csv')
    fig,axes=plt.subplots(1,2,sharey=True,figsize=(7.16,2.65));fig.subplots_adjust(left=.265,right=.985,bottom=.19,top=.83,wspace=.18)
    for ax,metric in zip(axes,METRICS):
        for i,a in enumerate(ARMS[1:]):
            for off,view,marker,fill in [(-.13,'clean','o',False),(.13,'six_practical_mean','s',True)]:
                r=next(r for r in paired if r['structure']==a and r['metric']==metric and r['view']==view)
                m=float(r['mean']);sd=float(r['sd']);color=LOSS if m<0 else GAIN if m>0 else NEUTRAL
                values=list(map(float,r['seed_values'].split(';')))
                ax.scatter(values,np.array([i+off-.035,i+off,i+off+.035]),s=7,color=[LOSS if value<0 else GAIN if value>0 else NEUTRAL for value in values],alpha=.36,zorder=2,edgecolors='none')
                ax.errorbar(m,i+off,xerr=sd,fmt=marker,ms=4,mfc=color if fill else 'white',mec=color,mew=.9,ecolor=color,elinewidth=.85,capsize=2,zorder=3)
        ax.axvline(0,color='#46515C',ls='--',lw=.75,zorder=1);ax.set(yticks=range(5),yticklabels=[EN[a] for a in ARMS[1:]],xlabel=('Δ Accuracy (pp)' if metric=='accuracy' else 'Δ Macro-F1 (pp)'));ax.grid(axis='x',alpha=.15,lw=.5)
    bounds=[(float(r['mean'])-float(r['sd']),float(r['mean'])+float(r['sd'])) for r in paired if r['view'] in ('clean','six_practical_mean')]
    limits=(math.floor(min(x[0] for x in bounds))-.25,math.ceil(max(x[1] for x in bounds))+.25)
    for ax in axes:ax.set_xlim(*limits);ax.set_xticks([-8,-4,0,4])
    axes[0].invert_yaxis()
    legend=[Line2D([],[],marker='o',color=NEUTRAL,ls='',mfc='white',label='Clean'),Line2D([],[],marker='s',color=NEUTRAL,ls='',label='Six-scene mean'),Line2D([],[],color=LOSS,lw=2,label='Negative: degradation'),Line2D([],[],color=GAIN,lw=2,label='Positive: improvement')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.58,1.0),ncol=2,frameon=False,columnspacing=1.6,handlelength=1.8)
    save(fig,'paired_changes','同种子配对差：变体减完整PAMF。误差线为3个配对差的样本标准差，浅色小点为每个种子的配对差；负值表示下降，正值表示提高。','figure_paired.csv')
    rr=[[next(r for r in paired if r['structure']==a and r['metric']=='accuracy' and r['view']==v) for v in SCENES] for a in ARMS[1:]]
    z=np.array([[float(r['mean']) for r in line] for line in rr]);limit=float(math.ceil(abs(z).max()))
    cmap=LinearSegmentedColormap.from_list('effect_direction',[LOSS,'#FCFCFC',GAIN])
    fig,ax=plt.subplots(figsize=(7.16,2.45));fig.subplots_adjust(left=.265,right=.86,bottom=.22,top=.96)
    im=ax.imshow(z,cmap=cmap,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),aspect='auto')
    ax.set(yticks=range(5),yticklabels=[EN[a] for a in ARMS[1:]],xticks=range(7),xticklabels=['Clean','High S','Mid S','Low S','High U','Mid U','Low U'])
    ax.set_xticks(np.arange(-.5,7,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True);ax.grid(which='minor',color='white',lw=.9);ax.tick_params(which='minor',length=0)
    for i in range(5):
        for j in range(7):ax.text(j,i,f'{z[i,j]:+.2f}',ha='center',va='center',fontsize=7.5,color='white' if abs(z[i,j])>.67*limit else '#20262C')
    ca=fig.add_axes([.89,.23,.015,.71]);cb=fig.colorbar(im,cax=ca);cb.set_label('Δ Accuracy (pp)',labelpad=6);cb.set_ticks([-limit,-limit/2,0,limit/2,limit])
    save(fig,'scene_changes','逐场景配对准确率变化，单位为百分点。暖红表示变体低于完整PAMF，青绿表示变体高于完整PAMF；色标以0为中心且对称。','figure_scene_changes.csv')
    (OUT/'figures/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(figures=3)),flush=True)

def tex(s):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','±':r'\ensuremath{\pm}','→':r'\ensuremath{\rightarrow}','−':'-'}
    return ''.join(mapping.get(c,c) for c in str(s))

def document():
    summary=list(rows(OUT/'tables/summary.csv'));paired=list(rows(OUT/'tables/variant_minus_full.csv'))
    def get(a,v='six_practical_mean',m='accuracy'):return next(r for r in summary if (r['structure'],r['view'],r['metric'])==(a,v,m))
    def pair(a,v='six_practical_mean',m='accuracy'):return next(r for r in paired if (r['structure'],r['view'],r['metric'])==(a,v,m))
    def fmt(r,signed=False,n=2):return f"{float(r['mean']):+.{n}f}±{float(r['sd']):.{n}f}" if signed else f"{float(r['mean']):.{n}f}±{float(r['sd']):.{n}f}"
    views_metrics=[('clean','accuracy'),('clean','macro_f1'),('six_practical_mean','accuracy'),('six_practical_mean','macro_f1')]
    definitions=[
        ('去除显式时间坐标','保留路径描述及共享骨干，只去除描述中的显式时间坐标；不是删除时域骨干。'),
        ('去除路径描述支路','完整移除SIG-TIME路径描述分支；基础时频表示继续训练。这就是“去SIG-TIME”，不能重复计为另一组。'),
        ('去除功放相关支路','移除功放相关计算和统计投影，其他结构继续按同预算训练。'),
        ('去除参考响应支路','移除公共参考响应及其投影，保留其他身份表示。'),
        ('打乱参考模板相位','固定扰乱公共模板频谱相位并保留幅度谱；训练与测试使用同一模板。')]
    paragraphs={
        'scope':'本报告只使用日期与接收机同时未见的测试群体，以及PAMF直接识别的冻结结果。标准六类划分中，训练使用源接收机的三天数据，测试使用7个目标接收机在另一较早日期的观测；每景42000包。该日期早于训练日期，不解释为未来日期外推。',
        'training':'识别器采用仅分类损失的监督控制配方，保留原输入与信道增强设置，既有默认分类头联合训练。完整网络与五个结构变体均使用相同划分和50轮预算，每组3个独立模型种子，共18个已完成模型。本文没有新增训练、预测或目标真值访问。',
        'statistics':'每个模型的全部声明场景决策冻结后，由独立CPU完成评分。六景均值不含clean，先在每个种子内对六个信道场景等权平均，再报告三个种子的均值与样本标准差。宏平均F1按全部六类计算。所有指定变体、三个种子和七个场景均保留。',
        'delta':'相对变化统一定义为变体减完整PAMF，同种子相减后再统计。负值表示性能下降，正值表示性能提高；±表示三个配对差的样本标准差，不是置信区间，也不是两组标准差相加。暖红与青绿分别标记下降和提高，图中同时使用正负号与点形，避免只依赖颜色。',
        'pa':'去除功放相关支路的六景准确率变化为'+fmt(pair('NO_PA'),True)+'个百分点，宏平均F1变化为'+fmt(pair('NO_PA',m='macro_f1'),True)+'个百分点。三个种子的六景准确率变化均为负，说明这一结构在当前监督配方与双重变化群体中有较一致的作用。clean变化的种子方向仍不完全一致，不能声称所有条件都同等受益。',
        'path':'去除路径描述支路的六景准确率变化为'+fmt(pair('NO_SIGNATURE'),True)+'个百分点，三个种子方向均为正；clean变化为'+fmt(pair('NO_SIGNATURE','clean'),True)+'个百分点。该结果不支持在本测试群体中声称完整SIG-TIME支路必然提高识别性能，也不能由此直接改变其他训练配方的结构。',
        'reference':'去除参考响应支路的六景变化为'+fmt(pair('NO_REFERENCE'),True)+'个百分点，三个种子均为负。打乱参考模板相位在clean下变化为'+fmt(pair('WRONG_REFERENCE','clean'),True)+'个百分点，但六景变化为'+fmt(pair('WRONG_REFERENCE'),True)+'个百分点，接近零。参考支路收益与“真实模板相位必不可少”不是同一个命题。',
        'time':'去除显式时间坐标的六景变化为'+fmt(pair('NO_TIME'),True)+'个百分点，两个种子下降、一个提高。该控制保留路径描述和骨干中的时域信息，不能把结果解释成移除了整个时间建模能力。',
        'limits':'本报告按用户指定范围事后筛选展示项，未依据成绩删除种子、重排训练或选择性重跑。结论只适用于当前六类划分、监督分类目标和跨日跨接收机测试群体；不能外推到其他群体或训练目标。三个种子不足以支撑普遍显著性结论；单支路改动的条件差异也不是可相加的独立因果贡献。'}
    title='PAMF网络分支消融：跨日跨接收机泛化'
    md=['# '+title,'','仅分类损失的监督训练 · IEEE 双栏版 · 2026-10-26','']
    def mdtable(head,data):
        md.extend(['| '+' | '.join(head)+' |','|'+'|'.join('---' for _ in head)+'|']);md.extend('| '+' | '.join(r)+' |' for r in data);md.append('')
    md.extend(['## 实验范围','',paragraphs['scope'],'',paragraphs['training'],'',paragraphs['statistics'],'','## 五项结构控制','']);mdtable(['结构变体','实际改动'],definitions)
    md.extend(['## 完整结果','']);heads=['网络结构','clean准确率（%）','clean宏平均F1（%）','六景准确率（%）','六景宏平均F1（%）']
    absolute=[[NAMES[a],*[fmt(get(a,v,m)) for v,m in views_metrics]] for a in ARMS];mdtable(heads,absolute)
    md.extend(['## 相对完整 PAMF 的变化','',paragraphs['delta'],'']);paired_data=[[NAMES[a],*[fmt(pair(a,v,m),True) for v,m in views_metrics]] for a in ARMS[1:]];mdtable(['结构变体','clean准确率Δ（百分点）','clean宏平均F1Δ（百分点）','六景准确率Δ（百分点）','六景宏平均F1Δ（百分点）'],paired_data)
    md.extend(['![同种子配对变化](figures/paired_changes.png)','','## 逐场景表现','']);mdtable(['网络结构',*SCENE_NAMES],[[NAMES[a],*[f"{float(get(a,v)['mean']):.2f}" for v in SCENES]] for a in ARMS]);md.extend(['![逐场景准确率](figures/seven_scene_accuracy.png)','','![逐场景配对变化](figures/scene_changes.png)','','## 结果解读',''])
    for k in ('pa','path','reference','time','limits'):md.extend([paragraphs[k],''])
    (OUT/'report.md').write_text('\n'.join(md).rstrip()+'\n',encoding='utf-8')
    write(OUT/'tables/论文表_完整结果.csv',[dict(zip(heads,r)) for r in absolute])
    write(OUT/'tables/论文表_相对变化.csv',[dict(zip(['结构变体','clean准确率Δ','clean宏平均F1Δ','六景准确率Δ','六景宏平均F1Δ'],r)) for r in paired_data])
    preamble=r'''\documentclass[10pt,journal]{IEEEtran}
\usepackage{fontspec,xeCJK}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Consolas}
\setCJKmainfont{SimSun}[BoldFont=SimHei,ItalicFont=KaiTi,BoldItalicFont=SimHei]
\usepackage{amsmath,amssymb,graphicx,booktabs,array,xcolor,placeins,stfloats,url}
\usepackage[hidelinks]{hyperref}
\definecolor{loss}{HTML}{B6574E}
\definecolor{gain}{HTML}{307F76}
\renewcommand{\abstractname}{摘要}
\renewcommand{\IEEEkeywordsname}{关键词}
\renewcommand{\tablename}{表}
\renewcommand{\figurename}{图}
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{4pt}
\setlength{\textfloatsep}{10pt plus 2pt minus 1pt}
\setlength{\dbltextfloatsep}{10pt plus 2pt minus 1pt}
\setlength{\floatsep}{8pt plus 2pt minus 1pt}
\renewcommand{\topfraction}{.95}
\renewcommand{\dbltopfraction}{.95}
\renewcommand{\textfraction}{.04}
\renewcommand{\floatpagefraction}{.8}
\renewcommand{\dblfloatpagefraction}{.8}
\setcounter{topnumber}{5}
\setcounter{dbltopnumber}{5}
\setcounter{totalnumber}{10}
\makeatletter
\setlength{\@fptop}{0pt}
\setlength{\@fpsep}{14pt}
\setlength{\@fpbot}{0pt plus 1fil}
\setlength{\@dblfptop}{0pt}
\setlength{\@dblfpsep}{14pt}
\setlength{\@dblfpbot}{0pt plus 1fil}
\makeatother
\clubpenalty=10000\widowpenalty=10000
\begin{document}
'''
    body=[r'\title{'+tex(title)+'}',r'\author{仅分类损失的监督训练\quad 实验结果报告\quad 2026年10月26日}',r'\maketitle',r'\begin{abstract}']
    abstract='本报告只比较PAMF直接识别在日期与接收机同时变化测试群体上的网络分支消融。完整网络与五项结构控制各含三个模型种子。相对变化采用“变体减完整网络”，负值表示下降。去除功放相关支路使六景准确率下降5.63±1.95个百分点，去除路径描述支路则提高0.85±0.25个百分点。结果按clean、六景均值及逐场景分别呈现，保留种子离散度与反向结果。'
    body.extend([tex(abstract),r'\end{abstract}',r'\begin{IEEEkeywords}射频指纹识别，PAMF，结构消融，跨日跨接收机泛化。\end{IEEEkeywords}',r'\section{实验设置与统计口径}',tex(paragraphs['scope']),tex(paragraphs['training']),tex(paragraphs['statistics']),r'\section{结构控制与差值定义}'])
    for name,description in definitions:body.append(r'\noindent\textbf{'+tex(name)+'}。'+tex(description)+r'\par')
    body.extend([tex(paragraphs['delta']),r'\begin{equation}\Delta_{s}=M_{s,\mathrm{variant}}-M_{s,\mathrm{full}},\quad \bar\Delta=\frac{1}{3}\sum_{s=1}^{3}\Delta_s.\end{equation}'])
    def tex_table(label,caption,data,is_paired=False):
        body.extend([r'\begin{table*}[!t]',r'\caption{'+tex(caption)+'}',r'\label{tab:'+label+'}',r'\centering\small',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}',r'\toprule',r'网络结构 & \multicolumn{2}{c}{clean} & \multicolumn{2}{c}{六景均值} \\',r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',r' & 准确率 & 宏平均F1 & 准确率 & 宏平均F1 \\',r'\midrule'])
        for a in data:
            values=[]
            for v,m in views_metrics:
                r=pair(a,v,m) if is_paired else get(a,v,m)
                cell=tex(fmt(r,is_paired))
                if is_paired:cell=r'\textcolor{'+('loss' if float(r['mean'])<0 else 'gain')+'}{'+cell+'}'
                elif a=='full':cell=r'\textbf{'+cell+'}'
                values.append(cell)
            body.append(tex(NAMES[a])+' & '+' & '.join(values)+r' \\')
        body.extend([r'\bottomrule',r'\end{tabular*}',r'\end{table*}'])
    tex_table('absolute','PAMF直接识别：跨日跨接收机测试结果（%，三种子均值±样本标准差）。',ARMS)
    tex_table('paired','同种子相对变化：结构变体减完整PAMF（百分点）。负值为下降，正值为提高；±为配对差的样本标准差。',ARMS[1:],True)
    body.extend([r'\section{结果与讨论}',r'表~\ref{tab:absolute}~给出绝对指标，表~\ref{tab:paired}~和图~\ref{fig:paired_changes}~给出同种子配对变化。',tex(paragraphs['pa']),tex(paragraphs['path']),tex(paragraphs['reference']),tex(paragraphs['time'])])
    captions={r['name']:r['caption'] for r in json.loads((OUT/'figures/manifest.json').read_text(encoding='utf-8'))}
    for name in ('paired_changes','seven_scene_accuracy','scene_changes'):
        body.extend([r'\begin{figure*}[!t]',r'\centering',r'\includegraphics[width=\textwidth]{../figures/'+name+'.pdf}',r'\caption{'+tex(captions[name])+'}',r'\label{fig:'+name+'}',r'\end{figure*}'])
    body.extend([r'图~\ref{fig:seven_scene_accuracy}~与图~\ref{fig:scene_changes}~按七个固定场景呈现结果。整体均值与逐场景结论并不等价，单个场景的正向或负向变化均应保留。',r'\section{结论与适用边界}',tex(paragraphs['limits']),r'\noindent\textbf{结论。} 当前监督控制中，功放相关支路的六景作用最明显；参考响应支路有较小且同种子方向一致的六景作用。完整路径描述和显式时间坐标的收益不能预设为普遍成立。原始指标、三个配对差及可编辑矢量图随本报告导出，便于论文使用与独立复算。',r'\FloatBarrier',r'\end{document}'])
    (OUT/'ieee/main.tex').write_text(preamble+'\n\n'.join(body)+'\n',encoding='utf-8')
    (OUT/'README.md').write_text('''# PAMF 网络分支消融：IEEE 聚焦版

仅分类损失的监督训练；只用日期与接收机同时变化测试群体，PAMF直接识别。完整模型＋五项变体，各3种子，共18模型。五项是去显式时间坐标、去路径描述、去功放相关、去参考响应、固定扰乱参考模板相位。删除完整SIG-TIME与删除路径描述是同一项；单视图控制不在本报告中。

差值一律为“变体减完整PAMF”，负值下降、正值提高，按种子配对。只包含性能指标；不含其他推理链、其他泛化群体、参数量或时延。

先读report.pdf或report.md；figures含3幅IEEE双栏宽度的PDF/SVG矢量图和PNG预览；ieee/main.tex可用XeLaTeX编译。tables包含总体／逐场景／逐接收机／逐发射机／逐日期指标及配对差。inputs只保存筛选后的既有冻结分数，不含IQ、权重或逐包真值。

解包后纯标准库复算：

```text
python -X utf8 -S reproduce/verify_report.py --root .
```

本次没有新增训练、预测或目标真值访问。新包r21，不覆盖历史报告。
''',encoding='utf-8')
    print(json.dumps(dict(status='IEEE_SOURCE_WRITTEN',tables=2,figures=3),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['extract','figures','document']);a=p.parse_args();{'extract':extract,'figures':figures,'document':document}[a.phase]()
