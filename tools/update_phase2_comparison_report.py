"""Refresh the existing Phase2 HTML from truth-last score snapshots.

The exported inputs contain metrics only, never IQ, prediction arrays or truth.
"""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'Microsoft YaHei','svg.fonttype':'none','axes.unicode_minus':False,
                     'font.size':10,'figure.facecolor':'white','axes.spines.top':False,
                     'axes.spines.right':False})
POINTS=((5,5),(5,10),(10,5),(10,10))
MIX3={'MIX01','MIX02','MIX03'}
SEEDS={2026092701,2026092702,2026092703}
LABELS={'mopc_hr':'原生 MoPC-HR','protonet_cda':'ProtoNet-CDA','mrior_sda':'MRIOR-SDA',
        'twostage':'Two-stage','poster_ft':'POSTER-FT','tweak_m1':'Tweak M1',
        'lora_triplet_knn':'原生 LoRa','orthogonal':'原生 Orthogonal'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def key(row):
    return row['capsule'],row['seed'],row.get('stage','C'),row['k'],row['n']


def ep(row):
    return row['capsule'].split('/')[2]


def verify_pair(a,b):
    assert a['split_id']==b['split_id']
    assert a['registered_classes']==b['registered_classes']
    assert a.get('count',a.get('query_count'))==b.get('count',b.get('query_count'))


def pct(value):
    return '—' if value is None else f'{100*value:.2f}%'


def average(values):
    return statistics.fmean(values) if values else None


def digest(rows,metric):
    vals=[r[metric] for r in rows if r.get(metric) is not None]
    return (average(vals),len(vals))


def make_figure(series,title,metric,path,baseline=None,ks=(5,10),ns=(5,10)):
    colors={1:'#408d60',5:'#1874a4',10:'#cf6f3d',20:'#8762a8'};fig,ax=plt.subplots(figsize=(6.5,4.0),layout='constrained')
    ax.set_title(title,loc='left',fontsize=11,fontweight='bold',pad=12)
    for k in ks:
        ys=[series[k,n][0]*100 if series[k,n][0] is not None else float('nan') for n in ns]
        ax.plot(ns,ys,marker='o',lw=2.1,markersize=5,color=colors[k],label=f'注册后 K={k}')
        if len(ks)<=2:
            for x,y,n in zip(ns,ys,ns):
                if y==y:ax.annotate(f'{y:.1f}  (n={series[k,n][1]})',(x,y),xytext=(0,-15 if k==5 else 8),textcoords='offset points',ha='center',fontsize=8,color=colors[k])
        if baseline and baseline[k][0] is not None:
            ax.axhline(baseline[k][0]*100,color=colors[k],lw=1.4,ls='--',alpha=.8,
                       label=f'注册前 B · K={k} (n={baseline[k][1]})')
    ax.set_xticks(ns);ax.set_xlabel('新增类别数 N');ax.set_ylabel(metric+'（%）')
    ax.set_ylim(0,110);ax.grid(axis='y',color='#dfe9ed',lw=.7)
    ax.legend(fontsize=7.8,frameon=False,ncol=2,loc='lower left');fig.savefig(path,format='svg');plt.close(fig)


def grouped_bars(data,title,path):
    fig,ax=plt.subplots(figsize=(10,max(4.6,len(data)*.42+1.4)),layout='constrained')
    y=list(range(len(data)));left=[v['baseline']*100 for v in data];right=[v['pamf']*100 for v in data]
    ax.barh([i+.18 for i in y],left,height=.32,color='#788fa0',label='对比方法 · 原始输入')
    ax.barh([i-.18 for i in y],right,height=.32,color='#167ca5',label='PAMF＋SCALR · 同任务原始输入')
    ax.set_yticks(y,[v['label'] for v in data]);ax.invert_yaxis();ax.set_xlim(0,119)
    ax.set_xlabel('注册后旧类与新类宏准确率的调和均值 H（%）')
    ax.set_title(title,loc='left',fontsize=12,fontweight='bold',pad=12);ax.grid(axis='x',color='#e5edf0');ax.set_axisbelow(True)
    for i,v in enumerate(data):
        ax.text(min(left[i]+.8,101),i+.18,f"{left[i]:.1f}  n={v['n']}",va='center',fontsize=7.2)
        ax.text(min(right[i]+.8,101),i-.18,f'{right[i]:.1f}',va='center',fontsize=7.2,color='#167ca5')
    ax.legend(frameon=False,loc='lower right',fontsize=8);fig.savefig(path,format='svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact-dir',required=True);p.add_argument('--html',required=True);a=p.parse_args()
    root=Path(a.artifact_dir);old=Path(a.html);figs=root/'phase2_figures_author_20261118';figs.mkdir(exist_ok=True)
    author=read(root/'phase2_author_report_evidence_20261118.json')
    pamfsat=read(root/'phase2_pamf_full16_report_20261118.json');base=read(root/'phase2_full_health_20261117.json')
    historical=read(root/'phase2_clean_satellite_paired_verified_20261117.json')
    assert not pamfsat['errors'] and pamfsat['tasks']==504 and len(pamfsat['paired_rows'])==24192 and not base['issues']
    runs=author['runs'];assert [(r['jobs'],r['scored'],len(r['failures'])) for r in runs]==[(6,6,0),(1134,1022,112),(6,6,0),(1128,1062,66)]
    assert len(runs[0]['rows'])+len(runs[1]['rows'])==4088
    assert len(runs[2]['rows'])+len(runs[3]['rows'])==4272
    pamf={}
    for condition,source in (('clean',author['pamf_clean']),('sat',pamfsat['paired_rows'])):
        for raw in source:
            if raw['seed'] not in SEEDS:continue
            row=dict(raw);row['count']=row.get('count',row.get('query_count'))
            row['stage']=row.get('stage');k=(condition,row['method'])+key(row)
            assert k not in pamf;pamf[k]=row
    author_rows={};failures=[]
    for i,r in enumerate(runs):
        condition='sat' if i<2 else 'clean'
        for item in r['rows']:
            assert item['input_condition']==('satellite_received' if condition=='sat' else 'clean_ground_pre_satellite_normalized')
            row=dict(item,stage='C');idx=(condition,row['profile'])+key(row)
            assert idx not in author_rows;author_rows[idx]=row
        failures += [dict(condition=condition,**f) for f in r['failures']]
    comparisons=defaultdict(dict)
    for (condition,profile,*_),row in author_rows.items():
        comparisons[(condition,('原生 MoPC-HR' if profile=='mopc_hr' else LABELS[profile]+' → MoPC-HR'))][key(row)]=row
    for row0 in base['paired_rows']:
        row=dict(row0,stage='C',count=row0['query_count'])
        if row['seed'] not in SEEDS or ep(row) not in MIX3 or (row['k'],row['n']) not in POINTS:continue
        if row['group']=='native_c' and row['profile'] in ('lora_triplet_knn','orthogonal'):
            label=LABELS[row['profile']]
        elif row['group']=='hybrid' and row['method'] in ('lora','orthogonal') and row['profile'] in ('protonet_cda','mrior_sda','twostage','poster_ft','tweak_m1'):
            label=LABELS[row['profile']]+' → '+('LoRa' if row['method']=='lora' else 'Orthogonal')
        else:continue
        idx=key(row);assert idx not in comparisons[('sat',label)];comparisons[('sat',label)][idx]=row
    # These historical rows were previously verified against split, registered classes and query count.
    for d in historical['details']:
        if d['group']!='native_cil' or d['stage']!='C' or d['method'] not in ('lora_triplet_knn','orthogonal'):continue
        cap='capsules/'+d['pool']+'/'+d['episode']+'/'+d['receiver']
        row=dict(capsule=cap,seed=d['seed'],stage='C',k=d['k'],n=d['n'],split_id=d['split_id'],h=d['clean'])
        comparisons[('clean',LABELS[d['method']])][key(row)]=row
    # Pair each competing method with PAMF on the identical split, not an unrelated overall mean.
    matched=defaultdict(list);mismatches=[]
    for (condition,label),items in comparisons.items():
        for ident,row in items.items():
            other=pamf.get((condition,'PAMF_clean')+ident)
            if other is None:continue
            try:
                assert row['split_id']==other['split_id']
                if 'registered_classes' in row:verify_pair(row,other)
            except AssertionError:
                mismatches.append((condition,label,ident));continue
            if row.get('h') is None or other.get('h') is None:continue
            matched[(condition,label,row['k'],row['n'])].append((row['h'],other['h']))
    assert not mismatches, f'paired identity mismatches: {mismatches[:5]}'
    sections=[]
    sections.append('<section id="author-current"><h2>新版 MoPC：完成情况与有效结果</h2>')
    sections.append('<div class="warning"><b>队列结束，结果仍有缺项。</b> 星地：1,134 个完整队列任务中 1,022 项完成评分、112 项隔离；另有 pilot 6 项的 K=5/N=5 已评分。clean：完整队列 1,062 项评分、66 项隔离，另有 pilot 6 项评分。星地隔离包括 65 项作者采样遗漏类别、47 项真实数值非有限；clean 的 66 项均为采样遗漏类别。隔离任务保留，不能算作成功。</div>')
    sections.append('<p>pilot 的 6 项 K=5/N=5 与完整队列中对应任务的其余 K/N 互补，不作为额外的 6 个独立任务。这里的 MoPC 固定作者仓库行为版：乱序和 drop-last，无梯度裁剪；IQ、模型接口及随机数按项目移植。历史裁剪版或论文公式版注册结果不并入。源 checkpoint 复用；MRIOR 与 Two-stage 的适应使用获授权的 L_s 源数据，Tweak 使用 checkpoint 中的源注册特征。clean 是实际地面信号，仍有地面信道和噪声。</p>')
    sections.append('<p>比较按相同数据池、MIX01–MIX03、接收机、模型种子、K、N 与 split 配对，MoPC 与当前 PAMF 的类别及 query 数也逐条核对。每个方法使用它自身能配对的任务，因此不同方法的 n 可能不同；缺失不作为零。Tweak→MoPC 的星地非有限任务较多，存活样本均值可能有选择偏差。原生注册方法没有额外的 B 阶段，与 PAMF 的旧类标签使用和训练预算不同。恢复版 PAMF 使用恢复后的 support 和 query，与其他方法的原始输入条件不同。</p>')
    sections.append('<p class="note">数据依据：<a href="phase2_author_comparison_summary_20261118.json">各图表逐组数值、任务数与隔离清单</a> · <a href="phase2_author_report_evidence_20261118.json">新版 MoPC 及 clean PAMF 的逐任务评分摘录</a>。文件不含原始 IQ、预测或 query 真值。</p>')
    for k,n in POINTS:
        graph=[]
        for (condition,label,kk,nn),values in sorted(matched.items()):
            if condition=='sat' and (kk,nn)==(k,n) and values:
                graph.append(dict(label=label,baseline=average(v[0] for v in values),pamf=average(v[1] for v in values),n=len(values)))
        graph.sort(key=lambda x:x['baseline'],reverse=True)
        path=figs/f'paired_sat_k{k}_n{n}.svg';grouped_bars(graph,f'星地原始输入 · K={k}，N={n}',path)
        sections.append(f'<figure><img src="{figs.name}/{path.name}" alt="星地 K={k} N={n} 同任务各方法与 PAMF 的宏 H 柱状图" style="width:100%;height:auto"><figcaption>每行的 PAMF 均用该方法对应的有效任务重新计算；n 为配对任务数，不同方法之间 n 可能不同。恢复版不在此原始输入图中。</figcaption></figure>')
    for condition,title in [('clean','地面 clean 原始输入'),('sat','星地原始输入')]:
        labels=sorted({label for c,label,_,_ in matched if c==condition})
        sections.append(f'<h3>{title}：逐方法、逐 K/N 的配对宏 H</h3><div class="table-wrap"><table><thead><tr><th>方法</th>'+''.join(f'<th>K={k}，N={n}</th>' for k,n in POINTS)+'</tr></thead><tbody>')
        for label in labels:
            cells=[]
            for k,n in POINTS:
                vals=matched.get((condition,label,k,n),[])
                if not vals:cells.append('<td>未找到匹配的有效评分</td>');continue
                m=average(x for x,_ in vals);p=average(y for _,y in vals)
                cells.append(f'<td><b>{pct(m)}</b> / PAMF {pct(p)}<span class="detail">配对 n={len(vals)} · 差 {(m-p)*100:+.2f} pp</span></td>')
            sections.append('<tr><th>'+html.escape(label)+'</th>'+''.join(cells)+'</tr>')
        sections.append('</tbody></table></div>')
    sections.append('<p class="note">clean 未列出部分混合 LoRa／Orthogonal 组合：本次可核对快照中没有对应 clean 评分，并非零分。原始输入 PAMF 为公平比较参照；恢复版结果见下方独立曲线。</p></section>')
    pamf_html=['<section id="pamf"><h2>一、PAMF＋SCALR：注册前后与新类数量</h2>',
        '<p>星地实验实际已完成 K=1、5、10、20 × N=2、5、10、20 的完整矩阵；此前报告仅导出了四格，是报告遗漏，本次直接复用已评分结果，没有重复运行。每张图固定数据口径、输入路径和指标。横轴是新增类别数 N，颜色对应每类支持样本数 K，不将不同 K 或 N 混合平均。星地八场景展示 PAMF 自身的完整覆盖；三场景展示用于与对比方法匹配的范围。clean 快照目前只展示 K=5、10 × N=5、10。</p>',
        '<p class="note"><a href="phase2_pamf_full16_report_20261118.json">查看星地完整 16 组 K/N 逐任务评分摘录</a>；每个八场景配置均有 504 个任务、每个三场景配置有 189 个任务。数据不含原始 IQ、预测或 query 真值。</p>',
        '<p>实线为注册后 C。旧类图的虚线为注册前 B（已使用旧类 support 适应）；A0 是不使用目标 support 的适应前源模型，见下方表格。B 只评估旧类，C 包含新旧类，两个阶段的 query 构成可能不同，不直接相减为遗忘率。图上的 n 是参与均值的任务数，不是独立受试者数。</p>']
    contexts=[('sat','PAMF_clean','星地八场景 · 原始 support / 原始 query',set(f'MIX{i:02}' for i in range(1,9))),
              ('sat','PAMF_恢复','星地八场景 · 恢复 support / 恢复 query',set(f'MIX{i:02}' for i in range(1,9))),
              ('sat','PAMF_clean','星地三场景 · 原始 support / 原始 query',MIX3),
              ('sat','PAMF_恢复','星地三场景 · 恢复 support / 恢复 query',MIX3),
              ('clean','PAMF_clean','地面 clean · 原始 support / 原始 query',MIX3),
              ('clean','PAMF_恢复','地面 clean · 恢复 support / 恢复 query',MIX3)]
    derived=[]
    for idx,(condition,method,title,episodes) in enumerate(contexts):
        ks=(1,5,10,20) if condition=='sat' else (5,10)
        ns=(2,5,10,20) if condition=='sat' else (5,10)
        rr=[row for (c,m,*_),row in pamf.items() if c==condition and m==method and ep(row) in episodes]
        before={};after={}
        for k in ks:
            before[k]={stage:[r for r in rr if r['stage']==stage and r['k']==k and r['n']==0] for stage in ('A0','B')}
            for n in ns:after[k,n]=[r for r in rr if r['stage']=='C' and r['k']==k and r['n']==n]
        if condition=='sat':
            expected_n=504 if len(episodes)==8 else 189
            assert all(len(after[k,n])==expected_n for k in ks for n in ns)
            assert all(len(before[k][stage])==expected_n for k in ks for stage in ('A0','B'))
        pamf_html.append(f'<h3>{title}</h3><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(410px,1fr));gap:12px">')
        for metric,desc in [('old','旧类宏准确率'),('new','新类宏准确率'),('h','宏 H')]:
            series={(k,n):digest(after[k,n],metric) for k in ks for n in ns}
            baselines={k:digest(before[k]['B'],'macro_accuracy') for k in ks} if metric=='old' else None
            file=figs/f'pamf_{idx}_{metric}.svg';make_figure(series,title+' · '+desc,desc,file,baselines,ks,ns)
            pamf_html.append(f'<figure><img src="{figs.name}/{file.name}" alt="{title} {desc}，按 K 分线、N 为横轴" style="width:100%;height:auto"><figcaption><a href="{figs.name}/{file.name}">下载可编辑 SVG</a></figcaption></figure>')
        pamf_html.append('</div><div class="table-wrap"><table><thead><tr><th>K</th><th>阶段</th><th>N</th><th>旧类宏准确率</th><th>新类宏准确率</th><th>全部类宏准确率</th><th>宏 H</th><th>宏 F1</th><th>任务数</th></tr></thead><tbody>')
        for k in ks:
            for stage,n,rs in [('A0',0,before[k]['A0']),('B',0,before[k]['B'])]+[('C',n,after[k,n]) for n in ns]:
                vals=[digest(rs,m)[0] for m in ('old','new','macro_accuracy','h','f1')]
                pamf_html.append(f'<tr><td>{k}</td><td>{stage}</td><td>{n if n else "—"}</td>'+''.join('<td>'+pct(v)+'</td>' for v in vals)+f'<td>{len(rs)}</td></tr>')
                derived.append(dict(condition=condition,method=method,scope=title,stage=stage,k=k,n=n,count=len(rs),
                                    old=vals[0],new=vals[1],macro_accuracy=vals[2],h=vals[3],f1=vals[4]))
        pamf_html.append('</tbody></table></div>')
    pamf_html.append('</section>')
    definition='''<section id="metric-definitions"><h2>指标定义与比较口径</h2>
<p>设第 i 类 query 的正确数为 TPᵢ、总数为 Tᵢ、被预测为该类的总数为 Pᵢ。类别准确率（召回率）rᵢ = TPᵢ / Tᵢ；仅对该任务中有 query 的类别计算。</p>
<ul><li><b>宏准确率：</b>所有已注册且有 query 的类别的 rᵢ 算术平均，各类等权，不按类样本数加权。</li>
<li><b>旧类／新类宏准确率：</b>分别对旧类别和本次注册的新类别取 rᵢ 的算术平均。注册前没有新类，因此新类项为“不适用”。</li>
<li><b>宏 H：</b>H = 2 × 旧类宏准确率 × 新类宏准确率 /（旧类宏准确率 + 新类宏准确率），两者均为 0 时记 0。先对每个任务算 H，再对匹配任务求均值；不等于先平均两类准确率再算 H。</li>
<li><b>总体准确率：</b>所有 query 中预测正确的比例，按样本计数；它与各类等权的宏准确率不同。</li>
<li><b>宏 F1：</b>各类别 F1ᵢ = 2TPᵢ /（Tᵢ + Pᵢ）后，对有 query 的类别等权平均；<b>类别下限</b>为这些类别准确率的最小值。</li></ul>
<p>每个表格数值均为该固定 K、N、阶段和输入条件下的逐任务算术平均。三池之间物理数据互斥，但同池重采样任务并不独立；模型种子、接收机与场景均在任务数 n 中。目标数据此前已暴露，本报告不称独立盲测或证明性能差异的因果来源。</p></section>'''
    content=old.read_text(encoding='utf8')
    content=re.sub(r'<!-- BEGIN 20261118 AUTHOR -->.*?<!-- END 20261118 AUTHOR -->','',content,flags=re.S)
    content=content.replace('<section id="overview">','<!-- BEGIN 20261118 AUTHOR -->'+''.join(sections)+definition+'<!-- END 20261118 AUTHOR --><section id="overview">',1)
    if '<!-- BEGIN 20261118 PAMF -->' in content:
        content=re.sub(r'<!-- BEGIN 20261118 PAMF -->.*?<!-- END 20261118 PAMF -->','',content,flags=re.S)
    else:
        content=re.sub(r'<section id="pamf">.*?(?=<section id="adapt">)','',content,flags=re.S)
    content=content.replace('<!-- BEGIN 20261118 AUTHOR -->','<!-- BEGIN 20261118 PAMF -->'+''.join(pamf_html)+'<!-- END 20261118 PAMF --><!-- BEGIN 20261118 AUTHOR -->',1)
    if '<a href="#author-current">' not in content:
        content=content.replace('<nav>','<nav><a href="#author-current">新版 MoPC 与对比</a><a href="#metric-definitions">指标定义</a>',1)
    content=content.replace('PHASE 2 · 固定任务配对 · 2026-11-17','PHASE 2 · 固定任务配对 · 更新至 2026-11-18',1)
    content=content.replace('clean 与星地：保留方法的历史结果','clean 与星地：PAMF 与对比方法结果',1)
    content=content.replace('依据 2026-11-17 的已评分快照生成；所有数值均为描述性统计，不使用测试结果反向调整模型。','历史对照基于 2026-11-17 已评分快照；作者版 MoPC 更新至 2026-11-18。所有数值均为描述性统计，不使用测试结果反向调整模型。')
    content=content.replace('<p>同一 K/N 的配对任务中，PAMF 原始输入为 94.30% → 60.85%；双端恢复为 93.88% → 73.68%。恢复输入与其他方法的原始输入是不同条件，需要分开解读。</p>',
        '<p>下方保留 2026-11-17 历史快照。新版 MoPC 的最新结果及隔离情况见本页开头；PAMF 的各 K、N 独立结果见下方折线图。</p>')
    content=content.replace('<span>已扫描两轮运行日志中的非有限等错误标记</span>','<span>历史两轮日志中的错误标记；不包含新版 MoPC</span>')
    old.write_text(content,encoding='utf8')
    out=dict(source_runs=[dict(run=r['run'],status=r['status'],jobs=r['jobs'],scored=r['scored'],
                  quarantined=len(r['failures']),rows=len(r['rows'])) for r in runs],pamf=derived,
             matched=[dict(condition=c,method=l,k=k,n=n,pairs=len(v),method_mean=average(x for x,_ in v),
                           pamf_mean=average(y for _,y in v)) for (c,l,k,n),v in sorted(matched.items())],
             failures=[dict(condition=x['condition'],profile=x['profile'],capsule=x['capsule'],seed=x['seed'],status=x['status']) for x in failures],
             note='Means within fixed K/N and method; scores only, no prediction or truth arrays.')
    (root/'phase2_author_comparison_summary_20261118.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(pamf_groups=len(derived),paired_cells=len(out['matched']),failures=len(failures),
                          figures=len(list(figs.glob('*.svg'))),report=str(old)),ensure_ascii=False))


if __name__=='__main__':main()
